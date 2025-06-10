from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
import numpy as np
import torch
import pandas as pd
import os
import sys

import json
import pickle

# Metrics
from sdmetrics import load_demo
from sdmetrics.single_table import LogisticDetection

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# Metrics


def reorder(real_data, syn_data, info, conditional_columns=[]):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    elif conditional_columns == []:
        cat_col_idx = cat_col_idx
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

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]

    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    metadata = info['metadata']

    columns = metadata['columns']
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']

    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]

    return new_real_data, new_syn_data, metadata


def detection(dataname=dataname, model=model, syn_path=syn_path, real_path=real_path, test_path=test_path, data_dir=data_dir):
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    syn_data.drop('income', axis=1, inplace=True)
    real_data = pd.read_csv(real_path)

    real_data = real_data[syn_data.columns.tolist()]

    save_dir = f'eval/density/{dataname}/{model}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    metadata = info['metadata']
    metadata['columns'] = {int(key): value for key,
                           value in metadata['columns'].items()}

    new_real_data, new_syn_data, metadata = reorder(
        real_data, syn_data, info, conditional_columns=[])

    # qual_report.generate(new_real_data, new_syn_data, metadata)
    # display(new_real_data)
    # display(new_syn_data)

    score = LogisticDetection.compute(
        real_data=new_real_data,
        synthetic_data=new_syn_data,
        metadata=metadata
    )

    print(f'{dataname}, {model}: {score}')
