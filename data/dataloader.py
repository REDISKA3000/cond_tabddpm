import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from dataset import Dataset

from copy import deepcopy

import sklearn.metrics as skm


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False, num_cond_columns=0):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_cond_columns = num_cond_columns

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def prepare_fast_dataloader(
    D: Dataset,
    split: str,
    batch_size: int
):

    X = torch.from_numpy(np.concatenate(
        [D.X_num[split], D.X_cat[split], D.X_cond[split]], axis=1)).float()
    dataloader = FastTensorDataLoader(X, batch_size=batch_size, shuffle=(
        split == 'train'), num_cond_columns=len(D.X_cond[split]))
    while True:
        yield from dataloader
