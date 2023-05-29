import argparse
import json
import random
from collections import OrderedDict

import gpytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from Utils.base_train import batch_sampled_data
from data_loader import ExperimentConfig
from forecast_denoising import Forecast_denoising

parser = argparse.ArgumentParser(description="preprocess argument parser")
parser.add_argument("--attn_type", type=str, default='KittyCat')
parser.add_argument("--name", type=str, default='KittyCat')
parser.add_argument("--exp_name", type=str, default='traffic')
parser.add_argument("--cuda", type=str, default="cuda:0")
parser.add_argument("--pred_len", type=int, default=24)
parser.add_argument("--dae", type=str, default="False")
parser.add_argument("--gp", type=str, default="False")
parser.add_argument("--no_noise", type=str, default="False")
parser.add_argument("--residual", type=str, default="False")
parser.add_argument("--input_corrupt", type=str, default="False")
parser.add_argument("--input_corrupt_iso", type=str, default="False")

args = parser.parse_args()

kernel = [9]
n_heads = 8
d_model = [32]
batch_size = 256

config = ExperimentConfig(args.pred_len, args.exp_name)

formatter = config.make_data_formatter()
params = formatter.get_experiment_params()
total_time_steps = params['total_time_steps']
num_encoder_steps = params['num_encoder_steps']
column_definition = params["column_definition"]
pred_len = args.pred_len

device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
data_csv_path = "{}.csv".format(args.exp_name)
raw_data = pd.read_csv(data_csv_path)

data = formatter.transform_data(raw_data)
train_max, valid_max = formatter.get_num_samples_for_calibration(num_train=batch_size)
max_samples = (train_max, valid_max)

_, _, test = batch_sampled_data(data, 0.8, max_samples, params['total_time_steps'],
                                        params['num_encoder_steps'], pred_len,
                                        params["column_definition"],
                                        batch_size)

test_enc, test_dec, test_y = next(iter(test))
total_b = len(list(iter(test)))

model_path = "models_{}_{}".format(args.exp_name, pred_len)
model_params = formatter.get_default_model_params()

src_input_size = test_enc.shape[2]
tgt_input_size = test_dec.shape[2]

predictions = np.zeros((3, total_b, test_y.shape[0], pred_len))
test_y_tot = torch.zeros((total_b, test_y.shape[0], pred_len))
n_batches_test = test_enc.shape[0]


mse = nn.MSELoss()
mae = nn.L1Loss()
stack_size = [2, 1]
denoising = True if args.dae == "True" else False
gp = True if args.gp == "True" else False
no_noise = True if args.no_noise == "True" else False
residual = True if args.residual == "True" else False
input_corrupt = True if args.input_corrupt == "True" else False
input_corrupt_iso = True if args.input_corrupt_iso == "True" else False


for i, seed in enumerate([7631, 9873, 5249]):
    for d in d_model:
        for layer in stack_size:
            try:
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)

                d_k = int(d / n_heads)

                config = src_input_size, tgt_input_size, d, n_heads, d_k, layer

                model = Forecast_denoising(model_name=args.name,
                                           config=config,
                                           gp=gp,
                                           denoise=denoising,
                                           device=device,
                                           seed=seed,
                                           pred_len=pred_len,
                                           attn_type=args.attn_type,
                                           no_noise=no_noise,
                                           residual=residual,
                                           input_corrupt=input_corrupt,
                                           input_corrupt_iso=input_corrupt_iso).to(device)
                model.to(device)

                checkpoint = torch.load(os.path.join("models_{}_{}".format(args.exp_name, pred_len),
                                        "{}_{}".format(args.name, seed)), map_location=device)

                if gp:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint['model_state_dict']
                    new_state_dict = OrderedDict()
                    for key, value in state_dict.items():

                        if "deep_gp" not in key:
                            if "mean_proj" not in key:
                                if "proj_up" not in key:
                                    new_state_dict[key] = value

                    state_dict = new_state_dict

                model.load_state_dict(state_dict)

                model.eval()

                j = 0
                for test_enc, test_dec, test_y in test:
                    if gp:
                        with gpytorch.settings.num_likelihood_samples(1):
                             output, _ = model(test_enc.to(device), test_dec.to(device))
                    else:
                        output, _ = model(test_enc.to(device), test_dec.to(device))

                    predictions[i, j] = output[:, -pred_len:, :].squeeze(-1).cpu().detach().numpy()
                    if i == 0:
                        test_y_tot[j] = test_y[:, -pred_len:, :].squeeze(-1).cpu().detach()
                    j += 1

            except RuntimeError as e:
                pass


predictions_mean = torch.from_numpy(np.mean(predictions, axis=0))
predictions = torch.from_numpy(predictions)
mse_std = torch.zeros(3, args.pred_len)
mae_std = torch.zeros(3, args.pred_len)

for i in range(3):
    for j in range(args.pred_len):
        mse_std[i, j] = mse(predictions[i, :, :, j], test_y_tot[:, :, j]).item()
        mae_std[i, j] = mae(predictions[i, :, :, j], test_y_tot[:, :, j]).item()

normaliser = test_y_tot.abs().mean()

mse_mean = mse_std.mean(dim=0)
m_mse_men = torch.mean(mse_mean).item() / normaliser
mae_mean = mae_std.mean(dim=0)
m_mae_men = torch.mean(mae_mean).item() / normaliser
mse_std = torch.mean(mse_std.std(dim=0)).item() / np.sqrt(pred_len)
mae_std = torch.mean(mae_std.std(dim=0)).item() / np.sqrt(pred_len)

results = torch.zeros(4, args.pred_len)


test_loss = mse(predictions_mean, test_y_tot).item() / normaliser
mae_loss = mae(predictions_mean, test_y_tot).item() / normaliser

for j in range(args.pred_len):

    results[0, j] = mse(predictions_mean[:, :, j], test_y_tot[:, :, j]).item()
    results[1, j] = mae(predictions_mean[:, :, j], test_y_tot[:, :, j]).item()
    results[2] = mse_mean
    results[2] = mae_mean

df = pd.DataFrame(results.detach().cpu().numpy())
if not os.path.exists("predictions"):
    os.makedirs("predictions")

df.to_csv(os.path.join("predictions", "{}_{}_{}.csv".format(args.exp_name, args.name, args.pred_len)))

erros = dict()
erros["{}".format(args.name)] = list()
erros["{}".format(args.name)].append(float("{:.5f}".format(test_loss)))
erros["{}".format(args.name)].append(float("{:.5f}".format(mae_loss)))
erros["{}".format(args.name)].append(float("{:.5f}".format(m_mse_men)))
erros["{}".format(args.name)].append(float("{:.5f}".format(m_mae_men)))
erros["{}".format(args.name)].append(float("{:.5f}".format(mse_std)))
erros["{}".format(args.name)].append(float("{:.5f}".format(mae_std)))

error_path = "final_final_errors_{}_{}.json".format(args.exp_name, pred_len)

if os.path.exists(error_path):
    with open(error_path) as json_file:
        json_dat = json.load(json_file)
        if json_dat.get("{}".format(args.name)) is None:
            json_dat["{}".format(args.name)] = list()
        json_dat["{}".format(args.name)].append(float("{:.5f}".format(test_loss)))
        json_dat["{}".format(args.name)].append(float("{:.5f}".format(mae_loss)))
        json_dat["{}".format(args.name)].append(float("{:.5f}".format(m_mse_men)))
        json_dat["{}".format(args.name)].append(float("{:.5f}".format(m_mae_men)))
        json_dat["{}".format(args.name)].append(float("{:.5f}".format(mse_std)))
        json_dat["{}".format(args.name)].append(float("{:.5f}".format(mae_std)))

    with open(error_path, "w") as json_file:
        json.dump(json_dat, json_file)
else:
    with open(error_path, "w") as json_file:
        json.dump(erros, json_file)
