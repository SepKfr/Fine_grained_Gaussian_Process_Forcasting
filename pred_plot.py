import argparse
import random

import matplotlib.pyplot as plt
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

plt.rc('font', size=8)
plt.rc('axes', titlesize=8)

parser = argparse.ArgumentParser(description="preprocess argument parser")
parser.add_argument("--attn_type", type=str, default='autoformer')
parser.add_argument("--name", type=str, default='autoformer')
parser.add_argument("--exp_name", type=str, default='traffic')
parser.add_argument("--cuda", type=str, default="cuda:0")
parser.add_argument("--pred_len", type=int, default=96)
parser.add_argument("--dae", type=str, default="False")
parser.add_argument("--gp", type=str, default="False")

args = parser.parse_args()

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
                                        batch_size, True)

test_enc, test_dec, test_y = next(iter(test))
total_b = len(list(iter(test)))

device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
model_path = "models_{}_{}".format(args.exp_name, pred_len)
model_params = formatter.get_default_model_params()

src_input_size = test_enc.shape[2]
tgt_input_size = test_dec.shape[2]

predictions = np.zeros((3, total_b, test_y.shape[0], pred_len))
test_y_tot = torch.zeros((total_b, test_y.shape[0], test_y.shape[1]))
total_steps = test_y.shape[1]
n_batches_test = test_enc.shape[0]


mse = nn.MSELoss()
mae = nn.L1Loss()
stack_size = 1


def get_pred_tgt(p_model, gp, name):

    for i, seed in enumerate([4293, 1692, 3029]):
        for d in d_model:
            try:
                d_k = int(d / n_heads)

                if "LSTM" in name:

                    model = RNN(n_layers=stack_size,
                                hidden_size=d,
                                src_input_size=src_input_size,
                                device=device,
                                d_r=0,
                                seed=seed,
                                pred_len=pred_len,
                                dae=p_model,
                                gp=gp)
                else:

                    model = Transformer(src_input_size=src_input_size,
                                        tgt_input_size=tgt_input_size,
                                        pred_len=pred_len,
                                        d_model=d,
                                        d_ff=d * 4,
                                        d_k=d_k, d_v=d_k, n_heads=n_heads,
                                        n_layers=stack_size, src_pad_index=0,
                                        tgt_pad_index=0, device=device,
                                        attn_type=args.attn_type,
                                        seed=seed, p_model=p_model, gp=gp)

                checkpoint = torch.load(os.path.join("models_{}_{}".format(args.exp_name, args.pred_len),
                                        "{}_{}".format(name, seed)))
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

        preds = torch.from_numpy(np.mean(predictions, axis=0))
        return preds.reshape(total_b*batch_size, -1), test_y_tot.reshape(total_b*batch_size, -1)


preds_gp, tgt = get_pred_tgt(True, True, "{}_gp".format(args.name))
preds_random, _ = get_pred_tgt(True, False, "{}_random".format(args.name))
preds, _ = get_pred_tgt(False, False, "{}".format(args.name))


diff_1 = 0
inds = []
mses = dict()
best_loss = 1e10

for j in range(total_b*batch_size):

    gp_loss = mse(preds_gp[j], tgt[j, -pred_len:]).item()
    random_loss = mse(preds_random[j], tgt[j, -pred_len:]).item()
    pred_loss = mse(preds[j], tgt[j, -pred_len:]).item()

    if gp_loss < random_loss and gp_loss < pred_loss:
        if gp_loss < best_loss:
            best_loss = gp_loss
            losses = [gp_loss, random_loss, pred_loss]
            mses[j] = losses
            inds.append(j)


inds.sort(reverse=True)

n = min(5, len(inds))

for i in range(0, n):

    loss_tuple = mses.get(inds[i])
    plt.plot(np.arange(total_steps), tgt[inds[i]], color="gray")
    plt.plot(np.arange(total_steps-pred_len, total_steps), preds[inds[i]], color="lime")
    plt.plot(np.arange(total_steps-pred_len, total_steps), preds_random[inds[i]], color="orchid")
    plt.plot(np.arange(total_steps-pred_len, total_steps), preds_gp[inds[i]], color="darkblue")

    plt.axvline(x=total_steps-pred_len, color="black")
    plt.legend(["ground-truth", "Prediction:MSE={:.3f}".format(loss_tuple[-1]),
                "Isotropic Denoised:MSE={:.3f}".format(loss_tuple[1]),
                "GP Denoised:MSE={:.3f}".format(loss_tuple[-1])])
    direc = os.path.join("prediction_plots", "{}_{}".format(args.exp_name, pred_len), "{}".format(args.name))
    if not os.path.exists(direc):
        os.makedirs(direc)
    plt.savefig(os.path.join(direc, "{}.pdf".format(i+1)), dpi=1000)
    plt.close()

