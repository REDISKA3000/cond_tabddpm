import numpy as np
import pandas as pd
import os
import sys
import json

from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

pd.options.mode.chained_assignment = None


def Alpha_Beta(model='model', syn_path='synthetic/synthetic.csv', real_path=f'synthetic/real.csv', data_dir=f'/content/drive/MyDrive/adult', dataname='adult', conditional_columns=[]):
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    syn_data.drop('income', axis=1, inplace=True)

    real_data = pd.read_csv(real_path)
    real_data = real_data[syn_data.columns.tolist()]

    ''' Special treatment for default dataset and CoDi model '''

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    elif conditional_columns == []:
        cat_col_idx = cat_col_idx
    else:
        cat_col_idx += target_col_idx

    print(conditional_columns)
    print('cat idx init', cat_col_idx)
    print('num idx init', num_col_idx)
    print('target idx init', target_col_idx)

    for i in conditional_columns:
        for j in range(len(cat_col_idx)):
            if cat_col_idx[i] < cat_col_idx[j]:
                cat_col_idx[j] = cat_col_idx[j] - 1

        for j in range(len(num_col_idx)):
            if cat_col_idx[i] < num_col_idx[j]:
                num_col_idx[j] = num_col_idx[j] - 1

        for j in range(len(target_col_idx)):
            if cat_col_idx[i] < target_col_idx[j]:
                target_col_idx[j] = target_col_idx[j] - 1

        cat_col_idx.pop(i)

    print('cat idx', cat_col_idx)
    print('num idx', num_col_idx)
    print('target idx', target_col_idx)

    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype('str')

    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]

    num_syn_data_np = num_syn_data.to_numpy()

    cat_syn_data_np = cat_syn_data.to_numpy().astype('str')

    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)

    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    le_real_data = pd.DataFrame(np.concatenate(
        (num_real_data_np, cat_real_data_oh), axis=1)).astype(float)
    le_real_num = pd.DataFrame(num_real_data_np).astype(float)
    le_real_cat = pd.DataFrame(cat_real_data_oh).astype(float)

    le_syn_data = pd.DataFrame(np.concatenate(
        (num_syn_data_np, cat_syn_data_oh), axis=1)).astype(float)
    le_syn_num = pd.DataFrame(num_syn_data_np).astype(float)
    le_syn_cat = pd.DataFrame(cat_syn_data_oh).astype(float)

    np.set_printoptions(precision=4)

    result = []

    print(' All Features ')
    print('Data shape: ', le_syn_data.shape)

    X_syn_loader = GenericDataLoader(le_syn_data)
    X_real_loader = GenericDataLoader(le_real_data)

    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {
        k: v for (k, v) in qual_res.items() if "naive" in k
    }
    qual_score = np.mean(list(qual_res.values()))

    print('alpha precision: {:.6f}, beta recall: {:.6f}'.format(
        qual_res['delta_precision_alpha_naive'], qual_res['delta_coverage_beta_naive']))

    Alpha_Precision_all = qual_res['delta_precision_alpha_naive']
    Beta_Recall_all = qual_res['delta_coverage_beta_naive']

    save_dir = f'/content/drive/MyDrive/results/metrcis/alpha_beta'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}/{model}.txt', 'w') as f:
        f.write(f'{Alpha_Precision_all}\n')
        f.write(f'{Beta_Recall_all}\n')
