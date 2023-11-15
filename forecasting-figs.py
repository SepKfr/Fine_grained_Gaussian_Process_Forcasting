import argparse
import random
import gpytorch
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from Utils.base_train import batch_sampled_data
from data_loader import ExperimentConfig
from forecast_denoising import Forecast_denoising

plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
plt.rcParams["figure.figsize"] = (12, 8)

parser = argparse.ArgumentParser(description="preprocess argument parser")
parser.add_argument("--attn_type", type=str, default='autoformer')
parser.add_argument("--model_name", type=str, default='autoformer')
parser.add_argument("--exp_name", type=str, default='traffic')
parser.add_argument("--cuda", type=str, default="cuda:0")
parser.add_argument("--pred_len", type=int, default=96)


args = parser.parse_args()

kernel = [9]
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
total_steps = test_y.shape[1]
y = test_y


model_path = "models_{}_{}".format(args.exp_name, pred_len)
model_params = formatter.get_default_model_params()


src_input_size = test_enc.shape[2]
tgt_input_size = test_dec.shape[2]


n_batches_test = test_enc.shape[0]


mse = nn.MSELoss()
mae = nn.L1Loss()
stack_size = [2, 1]


def get_pred_tgt(denoise, gp, iso):

    predictions = torch.zeros((total_b, y.shape[0], pred_len))
    test_y_tot = torch.zeros((total_b, y.shape[0], y.shape[1]))

    for i, seed in enumerate([8220]):
        model_name = "{}_{}_{}_{}{}{}{}".format(args.model_name, args.exp_name, pred_len, seed,
                                                    "_denoise" if denoise else "",
                                                    "_gp" if gp else "",
                                                    "_iso" if iso else "")
        print(model_name)

        for d in d_model:
            for layer in stack_size:
                try:
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)

                    d_k = int(d / n_heads)

                    config = src_input_size, tgt_input_size, d, n_heads, d_k, layer

                    model = Forecast_denoising(model_name=model_name,
                                               config=config,
                                               gp=gp,
                                               denoise=denoise,
                                               device=device,
                                               seed=seed,
                                               pred_len=pred_len,
                                               attn_type=args.attn_type,
                                               no_noise=False,
                                               residual=False,
                                               input_corrupt=False).to(device)
                    model.to(device)

                    checkpoint = torch.load(os.path.join("models_{}_{}".format(args.exp_name, pred_len),
                                                         "{}".format(model_name)), map_location=device)

                    state_dict = checkpoint['model_state_dict']

                    model.load_state_dict(state_dict)

                    model.eval()

                    print("Successful...")

                    j = 0
                    for test_enc, test_dec, test_y in test:
                        if gp:
                            with gpytorch.settings.num_likelihood_samples(1):
                                output, _, _ = model(test_enc.to(device), test_dec.to(device))
                        else:
                            output, _, _ = model(test_enc.to(device), test_dec.to(device))

                        predictions[j] = output[:, -pred_len:, :].squeeze(-1).cpu().detach()
                        if i == 0:
                            test_y_tot[j] = test_y.squeeze(-1).cpu().detach()
                        j += 1

                except RuntimeError as e:
                    pass

        return predictions.reshape(total_b*batch_size, -1), test_y_tot.reshape(total_b*batch_size, -1)


preds_gp, tgt = get_pred_tgt(True, True, False)
preds_random, _ = get_pred_tgt(True, False, True)
preds, _ = get_pred_tgt(False, False, False)

diff_1 = 0
diff_2 = 0
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


mses = dict(sorted(mses.items(), key=lambda item: item[1][0]))
print(len(mses))

direc = os.path.join("prediction_plots_5", "{}_{}".format(args.exp_name, pred_len), "{}".format(args.model_name))
if not os.path.exists(direc):
    os.makedirs(direc)
for key in mses.keys():

    loss_tuple = mses.get(key)

    plt.plot(np.arange(total_steps - pred_len), tgt[key][:-pred_len], color="black")
    plt.plot(np.arange(total_steps), tgt[key][-pred_len:], color="gray")
    plt.plot(np.arange(total_steps - pred_len, total_steps), preds[key], color="lime", alpha=0.5)
    plt.axvline(x=total_steps - pred_len, color="black", linestyle='--')
    plt.legend(["X", "Y", "Autoformer:MSE={:.3f}".format(loss_tuple[-1])])
    plt.tight_layout()
    plt.savefig(os.path.join(direc, "{}_{}.pdf".format(key, "Autoformer")), dpi=1000)
    plt.close()

    plt.plot(np.arange(total_steps - pred_len), tgt[key][:-pred_len], color="black")
    plt.plot(np.arange(total_steps), tgt[key][-pred_len:], color="gray")
    plt.plot(np.arange(total_steps - pred_len, total_steps), preds_random[key], color="orchid", alpha=0.5)
    plt.axvline(x=total_steps - pred_len, color="black", linestyle='--')
    plt.legend(["X", "Y", "AutoDI:MSE={:.3f}".format(loss_tuple[1])])
    plt.tight_layout()
    plt.savefig(os.path.join(direc, "{}_{}.pdf".format(key, "AutoDI")), dpi=1000)
    plt.close()

    plt.plot(np.arange(total_steps - pred_len), tgt[key][:-pred_len], color="black")
    plt.plot(np.arange(total_steps), tgt[key][-pred_len:], color="gray")
    plt.plot(np.arange(total_steps - pred_len, total_steps), preds_gp[key], color="darkblue", alpha=0.5)
    plt.axvline(x=total_steps - pred_len, color="black", linestyle='--')
    plt.legend(["X", "Y", "AutoGP:MSE={:.3f}".format(loss_tuple[0])])
    plt.tight_layout()
    plt.savefig(os.path.join(direc, "{}_{}.pdf".format(key, "AutoGP")), dpi=1000)
    plt.close()

