import argparse
import random
import gpytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from matplotlib import pyplot as plt

from Utils.base_train import batch_sampled_data
from data_loader import ExperimentConfig
from forecast_denoising import Forecast_denoising

parser = argparse.ArgumentParser(description="preprocess argument parser")
parser.add_argument("--attn_type", type=str, default='autoformer')
parser.add_argument("--model_name", type=str, default='autoformer')
parser.add_argument("--exp_name", type=str, default='autoformer')
parser.add_argument("--cuda", type=str, default="cuda:0")
parser.add_argument("--pred_len", type=int, default=24)
parser.add_argument("--denoising", type=str, default="False")
parser.add_argument("--gp", type=str, default="False")
parser.add_argument("--no-noise", type=str, default="False")
parser.add_argument("--residual", type=str, default="False")
parser.add_argument("--iso", type=str, default="False")
parser.add_argument("--input_corrupt_training", type=str, default="False")

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

model_path = "models_{}_{}".format(args.exp_name, pred_len)
model_params = formatter.get_default_model_params()


src_input_size = test_enc.shape[2]
tgt_input_size = test_dec.shape[2]

predictions = np.zeros((2, total_b, test_y.shape[0], pred_len))
test_y_tot = torch.zeros((total_b, test_y.shape[0], pred_len))
n_batches_test = test_enc.shape[0]


mse = nn.MSELoss()
mae = nn.L1Loss()
stack_size = [1, 2, 3]
denoising = True if args.denoising == "True" else False
gp = True if args.gp == "True" else False
no_noise = True if args.no_noise == "True" else False
residual = True if args.residual == "True" else False
iso = True if args.iso == "True" else False
input_corrupt = True if args.input_corrupt_training == "True" else False


for seed in [8220]:
    for i, m_n in enumerate(["basic", "ATA"]):
        try:
            model_name = "{}_{}_{}_{}{}{}{}{}{}".format(m_n, args.exp_name, pred_len, seed,
                                                        "_denoise" if denoising else "",
                                                        "_gp" if gp else "",
                                                        "_predictions" if no_noise else "",
                                                        "_iso" if iso else "",
                                                        "_residual" if residual else "",
                                                        "_input_corrupt" if input_corrupt else "")

            for d in d_model:

                for layer in stack_size:

                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)

                    d_k = int(d / n_heads)

                    config = src_input_size, tgt_input_size, d, n_heads, d_k, layer

                    model = Forecast_denoising(model_name=model_name,
                                               config=config,
                                               gp=gp,
                                               denoise=denoising,
                                               device=device,
                                               seed=seed,
                                               pred_len=pred_len,
                                               attn_type=args.attn_type,
                                               no_noise=no_noise,
                                               residual=residual,
                                               input_corrupt=input_corrupt).to(device)
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

                        predictions[i, j] = output[:, -pred_len:, :].squeeze(-1).cpu().detach().numpy()
                        if i == 0:
                            test_y_tot[j] = test_y[:, -pred_len:, :].squeeze(-1).cpu().detach()
                        j += 1

        except RuntimeError as e:
            print(e)


mse_std_mean = torch.zeros(2, pred_len)
mae_std_mean = torch.zeros(2, pred_len)

normaliser = test_y_tot.abs().mean()
predictions = torch.from_numpy(predictions)
predictions_mean = torch.mean(predictions, dim=0)

for i in range(2):

    for j in range(pred_len):

        mse_std_mean[i, j] = mse(predictions[i, :, j, :], test_y_tot[:, j, :])
        mae_std_mean[i, j] = mae(predictions[i, :, j, :], test_y_tot[:, j, :])


mae_loss_basic = mae_std_mean[0]
mse_loss_acat = mse_std_mean[1]

bar_width = 0.4

# Positions of the bars on the x-axis
r1 = np.arange(pred_len)
r2 = [x + bar_width for x in r1]

# Plot the data
plt.figure(figsize=(12, 6))
plt.bar(r1, mae_loss_basic, color='blue', width=bar_width, edgecolor='grey', label='Basic Attention')
plt.bar(r2, mse_loss_acat, color='red', width=bar_width, edgecolor='grey', label='ATA')

# Add title and labels
plt.title('MSE Loss')
plt.xlabel('Time Step')
plt.ylabel('Value')

# Add xticks on the middle of the group bars
plt.xticks([r + bar_width/2 for r in range(pred_len)], r1)

# Add legend
plt.legend()
plt.tight_layout()
plt.savefig("{}_basic_vs_ATA.png".format(args.exp_name), dpi=300)