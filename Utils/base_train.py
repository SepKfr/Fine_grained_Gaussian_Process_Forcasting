# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import torch
import numpy as np
from Utils import utils, base
import pandas as pd
import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

InputTypes = base.InputTypes


def sample_train_val_test(ddf, max_samples, time_steps, num_encoder_steps, pred_len, column_definition, tgt_all=False):

    id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definition)
    time_col = utils.get_single_col_by_input_type(InputTypes.TIME, column_definition)
    target_col = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definition)
    enc_input_cols = [
        tup[0]
        for tup in column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    valid_sampling_locations = []
    split_data_map = {}

    for identifier, df in ddf.groupby(id_col):
        num_entries = len(df)
        if num_entries >= time_steps:
            valid_sampling_locations += [
                (identifier, time_steps + i)
                for i in range(num_entries - time_steps + 1)
            ]

            split_data_map[identifier] = df

    if 0 < max_samples < len(valid_sampling_locations):
        ranges = [
            valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), max_samples, replace=False)
        ]
    else:
        print("maximum samples exceeds {}".format(len(valid_sampling_locations)))
        ranges = [
            valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), len(valid_sampling_locations), replace=False)
        ]

    input_size = len(enc_input_cols)
    inputs = np.zeros((max_samples, time_steps, input_size))
    enc_inputs = np.zeros((max_samples, num_encoder_steps, input_size))
    dec_inputs = np.zeros((max_samples, time_steps - num_encoder_steps - pred_len, input_size))
    outputs = np.zeros((max_samples, time_steps, 1))
    time = np.empty((max_samples, time_steps, 1), dtype=object)
    identifiers = np.empty((max_samples, time_steps, 1), dtype=object)

    for i, tup in enumerate(ranges):
        if (i + 1 % 1000) == 0:
            print(i + 1, 'of', max_samples, 'samples done...')
        identifier, start_idx = tup
        sliced = split_data_map[identifier].iloc[start_idx -
                                                 time_steps:start_idx]
        enc_inputs[i, :, :] = sliced[enc_input_cols].iloc[:num_encoder_steps]
        dec_inputs[i, :, :] = sliced[enc_input_cols].iloc[num_encoder_steps:-pred_len]
        inputs[i, :, :] = sliced[enc_input_cols]
        outputs[i, :, :] = sliced[[target_col]]
        time[i, :, 0] = sliced[time_col]
        identifiers[i, :, 0] = sliced[id_col]

    sampled_data = {
        'inputs': inputs,
        'enc_inputs': enc_inputs,
        'dec_inputs': dec_inputs,
        'outputs': outputs[:, -pred_len:, :],
        'input_arima': outputs[:, :-pred_len, :],
        'active_entries': np.ones_like(outputs[:, num_encoder_steps:, :]),
        'time': time,
        'identifier': identifiers
    }

    return sampled_data


def batch_sampled_data(data, train_percent, max_samples, time_steps,
                       num_encoder_steps, pred_len,
                       column_definition, batch_size, tgt_all=False):
    """Samples segments into a compatible format.
    Args:
      seed:
      column_definition:
      pred_len:
      num_encoder_steps:
      time_steps:
      data: Sources data_set to sample and batch
      max_samples: Maximum number of samples in batch
    Returns:
      Dictionary of batched data_set with the maximum samples specified.
    """

    np.random.seed(2436)
    random.seed(2436)

    time_col = utils.get_single_col_by_input_type(InputTypes.TIME, column_definition)
    id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definition)

    data.sort_values(by=[id_col, time_col], inplace=True)

    train_len = int(len(data) * train_percent)
    valid_len = int((len(data) - train_len) / 2)

    train = data[:train_len]
    valid = data[train_len:-valid_len]
    test = data

    train_max, valid_max = max_samples

    sample_train = sample_train_val_test(train, train_max, time_steps, num_encoder_steps, pred_len, column_definition)
    sample_valid = sample_train_val_test(valid, valid_max, time_steps, num_encoder_steps, pred_len, column_definition)
    sample_test = sample_train_val_test(test, valid_max, time_steps, num_encoder_steps, pred_len, column_definition, tgt_all)

    train_data = TensorDataset(torch.FloatTensor(sample_train['enc_inputs']),
                               torch.FloatTensor(sample_train['dec_inputs']),
                               torch.FloatTensor(sample_train['outputs']))

    valid_data = TensorDataset(torch.FloatTensor(sample_valid['enc_inputs']),
                               torch.FloatTensor(sample_valid['dec_inputs']),
                               torch.FloatTensor(sample_valid['outputs']))

    test_data = TensorDataset(torch.FloatTensor(sample_test['enc_inputs']),
                              torch.FloatTensor(sample_test['dec_inputs']),
                              torch.FloatTensor(sample_test['outputs']))

    train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last=True)
    valid_data = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, drop_last=True)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=batch_size, drop_last=True)

    return train_data, valid_data, test_data

