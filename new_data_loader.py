import itertools
import random
import numpy as np
import pandas as pd
import torch
from optuna.integration import sklearn
from pytorch_forecasting import GroupNormalizer
from torch.utils.data import BatchSampler, TensorDataset
from pytorch_forecasting.data import TimeSeriesDataSet, TorchNormalizer
from data import traffic, electricity, air_quality, solar




class DataLoader:
    def __init__(self, exp_name,
                 max_encoder_length,
                 pred_len,
                 target_col,
                 max_train_sample,
                 max_test_sample,
                 batch_size):

        data_formatter = {"traffic": traffic.TrafficFormatter,
                          "electricity": electricity.ElectricityFormatter,
                          "solar": solar.SolarFormatter,
                          "air_quality": air_quality.AirQualityFormatter}

        self.max_encoder_length = max_encoder_length
        self.pred_len = pred_len
        self.max_train_sample = max_train_sample
        self.max_test_sample = max_test_sample
        self.batch_size = batch_size
        seed = 1234
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        data_csv_path = "{}.csv".format(exp_name)
        data = pd.read_csv(data_csv_path, dtype={'date': str})
        data.sort_values(by=["id", "hours_from_start"], inplace=True)
        data = data_formatter[exp_name]().transform_data(data)

        total_batches = int(len(data) / self.batch_size)
        train_len = int(total_batches * batch_size * 0.8)
        valid_len = int(total_batches * batch_size * 0.1)
        test_len = int(total_batches * batch_size * 0.1)

        train = data[:train_len]
        valid = data[train_len:train_len+valid_len]
        test = data[train_len+valid_len:train_len+valid_len+test_len]

        train_data = pd.DataFrame(
            dict(
                value=train[target_col],
                group=train["id"],
                time_idx=np.arange(train_len),
            )
        )

        valid_data = pd.DataFrame(
            dict(
                value=valid[target_col],
                group=valid["id"],
                time_idx=np.arange(train_len, train_len+valid_len),
            )
        )

        test_data = pd.DataFrame(
            dict(
                value=test[target_col],
                group=test["id"],
                time_idx=np.arange(train_len+valid_len, train_len+valid_len+test_len),
            )
        )
        self.train_loader = self.create_dataloader(train_data, num_samples=self.max_train_sample)
        self.valid_loader = self.create_dataloader(valid_data, num_samples=self.max_test_sample)
        self.test_loader = self.create_dataloader(test_data, num_samples=self.max_test_sample)

    def create_time_series_dataset(self, data):
        return TimeSeriesDataSet(
            data,
            group_ids=["group"],
            target="value",
            time_idx="time_idx",
            min_encoder_length=1,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.pred_len,
        )

    def create_dataloader(self, data, num_samples):
        batch_sampler = BatchSampler(
            sampler=torch.utils.data.RandomSampler(data, num_samples=num_samples),
            batch_size=self.batch_size,
            drop_last=False,
        )
        data_loader = self.create_time_series_dataset(data).to_dataloader(batch_sampler=batch_sampler)
        x_enc_list = []
        x_dec_list = []
        y_list = []
        for x, y in data_loader:
            x_enc_list.append(x["encoder_target"][:, :-self.pred_len].unsqueeze(-1))
            x_dec_list.append(x["encoder_target"][:, -self.pred_len:].unsqueeze(-1))
            y_list.append(y[0].unsqueeze(-1))

        x_enc = torch.stack(list(itertools.chain.from_iterable(x_enc_list)))
        x_dec = torch.stack(list(itertools.chain.from_iterable(x_dec_list)))
        y = torch.stack(list(itertools.chain.from_iterable(y_list)))

        tensor_dataset = TensorDataset(x_enc,
                                       x_dec,
                                       y)

        return torch.utils.data.DataLoader(tensor_dataset, batch_size=self.batch_size)