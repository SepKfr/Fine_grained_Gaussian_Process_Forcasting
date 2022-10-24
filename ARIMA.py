import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

from Utils import utils, base
from Utils.base_train import batch_sampled_data, sample_train_val_test
from data.data_loader import ExperimentConfig
InputTypes = base.InputTypes


def Train(data, args, pred_len):

    config = ExperimentConfig(pred_len, args.exp_name)
    formatter = config.make_data_formatter()
    params = formatter.get_experiment_params()

    time_steps = params['total_time_steps']
    num_encoder_steps = params['num_encoder_steps']
    column_definition = params["column_definition"]

    data = formatter.transform_data(data)

    train_max, valid_max = formatter.get_num_samples_for_calibration()
    max_samples = (train_max, valid_max)

    np.random.seed(2436)
    random.seed(2436)

    time_col = utils.get_single_col_by_input_type(InputTypes.TIME, column_definition)
    id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definition)

    data.sort_values(by=[id_col, time_col], inplace=True)

    train_len = int(len(data) * 0.8)
    valid_len = int((len(data) - train_len) / 2)

    test = data[-valid_len:]

    train_max, valid_max = max_samples

    sample_test = sample_train_val_test(test, valid_max, time_steps, num_encoder_steps, pred_len, column_definition)

    y_true = np.squeeze(sample_test["outputs"], axis=-1)
    test = np.squeeze(sample_test["input_arima"], axis=-1)

    ls_outer = []

    for i in tqdm(range(len(test))):
        tmp = test[i]
        ls_inner = []
        for j in range(pred_len):
            arima = ARIMA(tmp, order=(1, 1, 0))
            model = arima.fit()
            pred = model.forecast()[0]
            tmp = np.append(tmp, pred)
            ls_inner.append(pred)
        ls_outer.append(ls_inner)

    predictions = torch.from_numpy(np.array(ls_outer))
    targets_all = torch.from_numpy(y_true)

    criterion = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    mse_loss = criterion(predictions, targets_all).item()
    normaliser = targets_all.abs().mean()
    mse_loss = mse_loss / normaliser

    mae_loss = mae_loss(predictions, targets_all).item()
    normaliser = targets_all.abs().mean()
    mae_loss = mae_loss / normaliser

    erros = dict()

    print("test loss {:.4f}".format(mse_loss))

    erros["{}".format(args.name)] = list()
    erros["{}".format(args.name)].append(float("{:.5f}".format(mse_loss)))
    erros["{}".format(args.name)].append(float("{:.5f}".format(mae_loss)))

    error_path = "new_Errors_{}_{}.json".format(args.exp_name, pred_len)

    if os.path.exists(error_path):
        with open(error_path) as json_file:
            json_dat = json.load(json_file)
            if json_dat.get("{}".format(args.name)) is None:
                json_dat["{}".format(args.name)] = list()
            json_dat["{}".format(args.name)].append(float("{:.5f}".format(mse_loss)))
            json_dat["{}".format(args.name)].append(float("{:.5f}".format(mae_loss)))

        with open(error_path, "w") as json_file:
            json.dump(json_dat, json_file)
    else:
        with open(error_path, "w") as json_file:
            json.dump(erros, json_file)


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--exp_name", type=str, default='traffic')
    parser.add_argument("--name", type=str, default='arima')
    args = parser.parse_args()

    data_csv_path = "{}.csv".format(args.exp_name)
    raw_data = pd.read_csv(data_csv_path)

    for pred_len in [24, 48, 72, 96]:
        Train(raw_data, args, pred_len)


if __name__ == '__main__':
    main()