import argparse
import json
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from Utils.base_train import batching, ModelData, batch_sampled_data
from data.data_loader import ExperimentConfig
from models.eff_acat import Transformer
from models.rnn import RNN

parser = argparse.ArgumentParser(description="preprocess argument parser")
parser.add_argument("--attn_type", type=str, default='KittyCat')
parser.add_argument("--name", type=str, default='KittyCat')
parser.add_argument("--exp_name", type=str, default='traffic')
parser.add_argument("--cuda", type=str, default="cuda:0")
parser.add_argument("--pred_len", type=int, default=24)
parser.add_argument("--dae", type=str, default="False")
parser.add_argument("--gp", type=str, default="False")

args = parser.parse_args()

kernel = [1, 3, 6, 9]
n_heads = 8
d_model = [16, 32]
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

device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
model_path = "models_{}_{}".format(args.exp_name, pred_len)
model_params = formatter.get_default_model_params()

src_input_size = test_enc.shape[2]
tgt_input_size = test_dec.shape[2]

predictions = np.zeros((3, total_b, test_y.shape[0], test_y.shape[1]))
test_y_tot = torch.zeros((total_b, test_y.shape[0], test_y.shape[1]))
n_batches_test = test_enc.shape[0]


mse = nn.MSELoss()
mae = nn.L1Loss()
stack_size = 1


for i, seed in enumerate([4293, 1692, 3029]):
    for d in d_model:
        for k in kernel:
            try:
                d_k = int(d / n_heads)

                if args.name == "lstm":
                    p_model = False
                    model = RNN(n_layers=stack_size,
                                hidden_size=d,
                                src_input_size=src_input_size,
                                tgt_input_size=tgt_input_size,
                                rnn_type="lstm",
                                device=device,
                                d_r=0,
                                seed=seed,
                                pred_len=pred_len)
                else:

                    p_model = True if args.dae == "True" else False
                    gp = True if args.gp == "True" else False
                    model = Transformer(src_input_size=src_input_size,
                                        tgt_input_size=tgt_input_size,
                                        pred_len=pred_len,
                                        d_model=d,
                                        d_ff=d * 4,
                                        d_k=d_k, d_v=d_k, n_heads=n_heads,
                                        n_layers=stack_size, src_pad_index=0,
                                        tgt_pad_index=0, device=device,
                                        attn_type=args.attn_type,
                                        seed=seed, kernel=k,
                                        p_model=p_model, gp=gp)

                checkpoint = torch.load(os.path.join("models_{}_{}".format(args.exp_name, args.pred_len),
                                        "{}_{}".format(args.name, seed)))
                state_dict = checkpoint['model_state_dict']
                new_state_dict = OrderedDict()

                if not p_model:
                    for key, value in state_dict.items():
                        if 'process' not in key:
                            new_state_dict[key] = value

                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])

                model.eval()
                model.to(device)

                j = 0
                for test_enc, test_dec, test_y in test:
                    if p_model:
                        output, _ = model(test_enc.to(device), test_dec.to(device))
                    else:
                        output = model(test_enc.to(device), test_dec.to(device))

                    predictions[i, j] = output.squeeze(-1).cpu().detach().numpy()
                    if i == 0:
                        test_y_tot[j] = test_y.squeeze(-1).cpu().detach()
                    j += 1

            except RuntimeError as e:
                pass

predictions = torch.from_numpy(np.mean(predictions, axis=0))

results = torch.zeros(2, args.pred_len)
normaliser = test_y_tot.abs().mean()

test_loss = mse(predictions, test_y_tot).item() / normaliser
mae_loss = mae(predictions, test_y_tot).item() / normaliser

for j in range(args.pred_len):

    results[0, j] = mse(predictions[:, :, j], test_y_tot[:, :, j]).item()
    results[1, j] = mae(predictions[:, :, j], test_y_tot[:, :, j]).item()

df = pd.DataFrame(results.detach().cpu().numpy())
df.to_csv("{}_{}_{}.csv".format(args.exp_name, args.name, args.pred_len))

erros = dict()
erros["{}".format(args.name)] = list()
erros["{}".format(args.name)].append(float("{:.5f}".format(test_loss)))
erros["{}".format(args.name)].append(float("{:.5f}".format(mae_loss)))

error_path = "final_errors_{}_{}.json".format(args.exp_name, pred_len)

if os.path.exists(error_path):
    with open(error_path) as json_file:
        json_dat = json.load(json_file)
        if json_dat.get("{}".format(args.name)) is None:
            json_dat["{}".format(args.name)] = list()
        json_dat["{}".format(args.name)].append(float("{:.5f}".format(test_loss)))
        json_dat["{}".format(args.name)].append(float("{:.5f}".format(mae_loss)))

    with open(error_path, "w") as json_file:
        json.dump(json_dat, json_file)
else:
    with open(error_path, "w") as json_file:
        json.dump(erros, json_file)
