import os
from urllib import request
import numpy as np
import pandas as pd
import zipfile
import json
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import numpy as np
from dataclasses import astuple, dataclass, replace
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List
import torch
from pathlib import Path
import enum
from torch import nn
import sklearn
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import torch
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor
import math
import kaleido
import plotly
import tomli
from mlp import MLPDiffusion
from diffusion import GaussianMultinomialDiffusion
from dataloader import FastTensorDataLoader


def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
):
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model


class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, model_save_path, device=torch.device('cuda:1'), conditional_columns=[]):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(
            columns=['step', 'mloss', 'gloss', 'loss'])
        self.model_save_path = model_save_path
        self.num_cond_columns = len(conditional_columns)
        self.conditional_columns = conditional_columns

        columns = list(np.arange(5)*200)
        columns[0] = 1
        columns = ['step'] + columns

        self.log_every = 50
        self.print_every = 1
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, x_cond):
        x = x.to(self.device)
        x_cond = x_cond.to(self.device)

        self.optimizer.zero_grad()

        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, x_cond=x_cond)

        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        self.print_every = 1
        self.log_every = 1

        best_loss = np.inf
        print('Steps: ', self.steps)
        while step < self.steps:
            start_time = time.time()
            x = next(self.train_iter)[0]
            # print(x.shape)
            # display(x)
            # conditional_columns = list(map(lambda x: x+6, self.conditional_columns))
            # x_cond = x[:, conditional_columns]
            # x_non_cond = np.delete(x, conditional_columns, axis= 1)
            if self.num_cond_columns == 0:
                self.num_cond_columns += 1
            # print('num ond cols', self.num_cond_columns)

            # x = [x_num, x_cat, x_cond, y]
            x_non_cond = x[:, :(-1) * self.num_cond_columns]
            x_cond = x[:, (-1) * self.num_cond_columns:]
            # print(x_cond.shape, len(np.unique(x_cond)))
            # print(len(np.unique(x_cond)))
            # for i in range(len(x_non_cond)):
            #   print(len(np.unique(x_non_cond[:,i])))
            # display(x_cond)
            # display(x_non_cond)
            # display(x[:,6:])
            # display(x[:,:6])
            # break

            batch_loss_multi, batch_loss_gauss = self._run_step(
                x_non_cond, x_cond)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if np.isnan(gloss):
                    print('Finding Nan')
                    break

                if (step + 1) % self.print_every == 0:
                    print(
                        f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1, mloss, gloss, mloss + gloss]

                np.set_printoptions(suppress=True)

                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

                if mloss + gloss < best_loss:
                    best_loss = mloss + gloss
                    cond_cols = list(
                        map(lambda x: str(x), self.conditional_columns))
                    model_name = 'model_cond_cols_' + \
                        '_'.join(cond_cols)+'_best_loss.pt'
                    torch.save(self.diffusion._denoise_fn.state_dict(),
                               os.path.join(self.model_save_path, model_name))

                if (step + 1) % 10000 == 0:
                    cond_cols = list(
                        map(lambda x: str(x), self.conditional_columns))
                    model_name = 'model_cond_cols_' + \
                        '_'.join(cond_cols)+f'_{step+1}.pt'
                    torch.save(self.diffusion._denoise_fn.state_dict(),
                               os.path.join(self.model_save_path, model_name))

            # update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1

            # end_time = time.time()
            # print('Time: ', end_time - start_time)


def train(
    model_save_path,
    real_data_path,
    steps=1000,
    lr=0.002,
    weight_decay=1e-4,
    batch_size=1024,
    task_type='binclass',
    model_type='mlp',
    model_params=None,
    num_timesteps=1000,
    gaussian_loss_type='mse',
    scheduler='cosine',
    T_dict=None,
    num_numerical_features=0,
    device=torch.device('cuda:0'),
    seed=0,
    change_val=False,
    conditional_columns=[]
):
    real_data_path = os.path.normpath(real_data_path)

    # zero.improve_reproducibility(seed)

    T = Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        task_type=task_type,
        change_val=False,
        conditional_columns=conditional_columns
    )
    print('dataset done')
    # display(np.unique(dataset.X_cond['train']))
    K = np.array(dataset.get_category_sizes('train'))
    num_classes_cond = np.array(dataset.get_category_sizes_cond('train'))
    print(K, num_classes_cond)

    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_classes = K  # it as a vector [K1, K2, ..., Km]
    num_classes_expanded = torch.from_numpy(
        np.concatenate([num_classes[i].repeat(num_classes[i])
                       for i in range(len(num_classes))])
    )

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    print(' d in', d_in)
    model_params['d_in'] = d_in

    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)
    print('model done')
    train_loader = prepare_fast_dataloader(
        dataset, split='train', batch_size=batch_size)

    print('loader done')

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device,
        cond_features=num_classes_cond
    )

    num_params = sum(p.numel() for p in diffusion.parameters())

    diffusion.to(device)

    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=100000,
        model_save_path=model_save_path,
        device=device,
        conditional_columns=conditional_columns
    )
    trainer.run_loop()
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    cond_cols = list(map(lambda x: str(x), conditional_columns))
    model_name = 'model_cond_cols_'+'_'.join(cond_cols)+'.pt'
    model_ema_name = 'model_ema_cond_cols_'+'_'.join(cond_cols)+'.pt'
    loss_name = 'loss_cond_cols_' + '_'.join(cond_cols)+'.csv'

    torch.save(diffusion._denoise_fn.state_dict(),
               os.path.join(model_save_path, model_name))

    # print('saved1')

    torch.save(trainer.ema_model.state_dict(),
               os.path.join(model_save_path, model_ema_name))

    # print('saved2')

    trainer.loss_history.to_csv(os.path.join(
        model_save_path, loss_name), index=False)
