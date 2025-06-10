
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


def DCR(dataname=dataname, model=model, syn_path=syn_path, real_path=real_path, test_path=test_path, data_dir=data_dir, conditional_columns=[]):
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)
    test_data = pd.read_csv(test_path)

    real_data = real_data[syn_data.columns.tolist()]
    test_data = test_data[syn_data.columns.tolist()]

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

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

    num_ranges = []

    real_data.columns = list(np.arange(len(real_data.columns)))
    syn_data.columns = list(np.arange(len(real_data.columns)))
    test_data.columns = list(np.arange(len(real_data.columns)))
    for i in num_col_idx:
        num_ranges.append(real_data[i].max() - real_data[i].min())

    num_ranges = np.array(num_ranges)

    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]
    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]
    num_test_data = test_data[num_col_idx]
    cat_test_data = test_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype('str')
    num_syn_data_np = num_syn_data.to_numpy()
    cat_syn_data_np = cat_syn_data.to_numpy().astype('str')
    num_test_data_np = num_test_data.to_numpy()
    cat_test_data_np = cat_test_data.to_numpy().astype('str')

    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)

    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()
    cat_test_data_oh = encoder.transform(cat_test_data_np).toarray()

    num_real_data_np = num_real_data_np / num_ranges
    num_syn_data_np = num_syn_data_np / num_ranges
    num_test_data_np = num_test_data_np / num_ranges

    real_data_np = np.concatenate([num_real_data_np, cat_real_data_oh], axis=1)
    syn_data_np = np.concatenate([num_syn_data_np, cat_syn_data_oh], axis=1)
    test_data_np = np.concatenate([num_test_data_np, cat_test_data_oh], axis=1)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    real_data_th = torch.tensor(real_data_np).to(device)
    syn_data_th = torch.tensor(syn_data_np).to(device)
    test_data_th = torch.tensor(test_data_np).to(device)

    dcrs_real = []
    dcrs_test = []
    batch_size = 100

    batch_syn_data_np = syn_data_np[i*batch_size: (i+1) * batch_size]

    for i in range((syn_data_th.shape[0] // batch_size) + 1):
        if i != (syn_data_th.shape[0] // batch_size):
            batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
        else:
            batch_syn_data_th = syn_data_th[i*batch_size:]

        dcr_real = (batch_syn_data_th[:, None] -
                    real_data_th).abs().sum(dim=2).min(dim=1).values
        dcr_test = (batch_syn_data_th[:, None] -
                    test_data_th).abs().sum(dim=2).min(dim=1).values
        dcrs_real.append(dcr_real)
        dcrs_test.append(dcr_test)

    dcrs_real = torch.cat(dcrs_real)
    dcrs_test = torch.cat(dcrs_test)

    score = (dcrs_real < dcrs_test).nonzero().shape[0] / dcrs_real.shape[0]

    print(f'{dataname}-{model}, DCR Score = {score}')
