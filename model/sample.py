import argparse
import torch
import numpy as np
import pandas as pd
import os
import json
import time
from mlp import MLPDiffusion
from diffusion import GaussianMultinomialDiffusion


def sample_all(self, num_samples, batch_size, ddim=False, steps=1000):
    if ddim:
        print('Sample using DDIM.')
        sample_fn = self.sample_ddim
    else:
        sample_fn = self.sample

    b = batch_size

    all_samples = []
    num_generated = 0
    while num_generated < num_samples:
        if not ddim:
            sample = sample_fn(b)
        else:
            sample = sample_fn(b, steps=steps)
        mask_nan = torch.any(sample.isnan(), dim=1)
        sample = sample[~mask_nan]

        all_samples.append(sample)

        if sample.shape[0] != b:
            raise FoundNANsError
        num_generated += sample.shape[0]

    x_gen = torch.cat(all_samples, dim=0)[:num_samples]

    return x_gen


@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, conditional_columns=[]):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    syn_num = num_inverse(syn_num).astype(np.float32)
    syn_cat = cat_inverse(syn_cat)

    return syn_num, syn_cat

    # if info['task_type'] == 'regression':
    #     syn_target = syn_num[:, :len(target_col_idx)]
    #     syn_num = syn_num[:, len(target_col_idx):]

    # else:
    #     print(syn_cat.shape)
    #     syn_target = syn_cat[:, :len(target_col_idx)]
    #     syn_cat = syn_cat[:, len(target_col_idx):]

    # return syn_num, syn_cat, syn_target


def recover_data(syn_num, syn_cat, syn_target, info, conditional_columns=[]):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    print(cat_col_idx)
    for i in cat_col_idx:
        for j in conditional_columns:
            print('idx', cat_col_idx[j])
            if i > cat_col_idx[j]:
                idx_mapping[i] = idx_mapping[i]-1

    conds = list(map(lambda x: cat_col_idx[x], conditional_columns))

    for i in conditional_columns:
        cat_col_idx.pop(i)

    print('map', idx_mapping, len(idx_mapping))
    print('map vals', idx_mapping.values())

    syn_df = pd.DataFrame()
    print(syn_num.shape, syn_cat.shape)

    print(idx_mapping)

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in conditional_columns:
                continue

            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            # else:
            #     syn_df[i] = syn_target[:, idx_mapping[i] -
            #                            len(num_col_idx) - len(cat_col_idx)]

    else:
        for i in range(len(num_col_idx) + len(cat_col_idx)):
            # if i in conditional_columns:
            #     continue
            print('col num', i)
            if i in conds:
                print('skip', i)
                continue

            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                print(i, idx_mapping[i] - len(num_col_idx),
                      idx_mapping[i], len(num_col_idx))
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            # else:
            #     print('target', i)
            #     print(idx_mapping[i] - len(num_col_idx) - len(cat_col_idx))
            #     syn_df[i] = syn_target[:, idx_mapping[i] -
            #                            len(num_col_idx) - len(cat_col_idx) - 1]

    return syn_df


def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
):
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model


def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


def sample(
    model_save_path,
    sample_save_path,
    real_data_path,
    batch_size=2000,
    num_samples=0,
    task_type='binclass',
    model_type='mlp',
    model_params=None,
    num_timesteps=1000,
    gaussian_loss_type='mse',
    scheduler='cosine',
    T_dict=None,
    num_numerical_features=0,
    disbalance=None,
    device=torch.device('cuda:0'),
    change_val=False,
    ddim=False,
    steps=1000,
    conditional_columns=[],
):

    T = Transformations(**T_dict)

    D = make_dataset(
        real_data_path,
        T,
        task_type=task_type,
        change_val=False,
        conditional_columns=conditional_columns
    )

    K = np.array(D.get_category_sizes('train'))
    num_classes_cond = np.array(D.get_category_sizes_cond('train'))

    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    print(np.sum(num_classes_cond))
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)

    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )
    cond_cols = list(map(lambda x: str(x), conditional_columns))
    # model_path = f'{model_save_path}/model_cond_cols_'+'_'.join(cond_cols)+'_best_loss.pt'
    model_path = f'/content/drive/MyDrive/results/models/adult/model_cond_cols__best_loss.pt'

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler,
        device=device,
        cond_features=num_classes_cond

    )

    diffusion.to(device)
    diffusion.eval()

    start_time = time.time()
    x_cond = torch.from_numpy(D.X_cond['train']).float()
    display(x_cond)
    if not ddim:
        x_gen = diffusion.sample_all(
            num_samples, batch_size, ddim=False, x_cond=x_cond)
    else:
        x_gen = diffusion.sample_all(
            num_samples, batch_size, ddim=True, steps=steps, x_cond=x_cond)

    print('Shape', x_gen.shape)
    display(x_gen)
    # for i in range(8):
    #   print(len(np.unique(x_gen[:,i+6])))

    syn_data = x_gen
    num_inverse = D.num_transform.inverse_transform
    cat_inverse = D.cat_transform.inverse_transform

    info_path = f'{real_data_path}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)
    print(syn_data.shape)
    # syn_num, syn_cat, syn_target = split_num_cat_target(
    #     syn_data, info, num_inverse, cat_inverse)

    syn_num, syn_cat = split_num_cat_target(
        syn_data, info, num_inverse, cat_inverse)

    print(syn_num.shape, syn_cat.shape)
    # print(len(np.unique(syn_target)))
    syn_df = recover_data(syn_num, syn_cat, [], info, conditional_columns)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key,
                        value in idx_name_mapping.items()}

    syn_df.rename(columns=idx_name_mapping, inplace=True)
    end_time = time.time()

    print('Sampling time:', end_time - start_time)

    save_path = sample_save_path

    syn_df['income'] = pd.read_csv(
        '/content/drive/MyDrive/results/synthetic/real.csv')['income']
    syn_df.to_csv(save_path, index=False)


def sample_main(gpu=1, steps=1000, ddim=False, conditional_columns=[]):
    dataname = 'adult'
    device = 'cpu'

    curr_dir = 'model'
    config_path = '/content/drive/MyDrive/adult/info/adult.toml'
    model_save_path = '/content/drive/MyDrive/results/models/adult'
    real_data_path = '/content/drive/MyDrive/adult'

    cond_cols = list(map(lambda x: str(x), conditional_columns))
    sample_save_path = '/content/drive/MyDrive/results/synthetic/synthetic_cond_cols_' + \
        '_'.join(cond_cols) + '.csv'

    train = True

    raw_config = load_config(config_path)

    '''
    Modification of configs
    '''
    print('START SAMPLING')

    sample(
        num_samples=raw_config['sample']['num_samples'],
        # num_samples = 20000,
        batch_size=raw_config['sample']['batch_size'],
        # batch_size = 6000,
        disbalance=raw_config['sample'].get('disbalance', None),
        **raw_config['diffusion_params'],
        model_save_path=model_save_path,
        sample_save_path=sample_save_path,
        real_data_path=real_data_path,
        task_type=raw_config['task_type'],
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device='cuda',
        ddim=ddim,
        steps=steps,
        conditional_columns=[]
    )
