import gpytorch
from forecast_denoising import Forecast_denoising
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
import argparse
import os
import torch.nn.functional as F
import random
import pandas as pd
import optuna
from optuna.trial import TrialState
from modules.opt_model import NoamOpt
from new_data_loader import DataLoader

torch.autograd.set_detect_anomaly(True)

with gpytorch.settings.num_likelihood_samples(1):

    class Train:
        def __init__(self, exp_name, args, pred_len, seed):

            self.denoising = True if args.denoising == "True" else False
            self.gp = True if args.gp == "True" else False
            self.iso = True if args.iso == "True" else False
            self.no_noise = True if args.no_noise == "True" else False
            self.residual = True if args.residual == "True" else False
            self.input_corrupt_training = True if args.input_corrupt_training == "True" else False

            self.seed = seed
            self.pred_len = pred_len
            target_col = {"traffic": "values",
                          "electricity": "power_usage",
                          "exchange": "value",
                          "solar": "Power(MW)",
                          "air_quality": "NO2"}

            self.dataloader_obj = DataLoader(exp_name,
                                             max_encoder_length=96 + 2 * pred_len,
                                             target_col=target_col[exp_name],
                                             pred_len=pred_len,
                                             max_train_sample=6400,
                                             max_test_sample=640,
                                             batch_size=64)

            self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
            self.model_path = "models_{}_{}".format(args.exp_name, pred_len)
            self.attn_type = args.attn_type
            self.criterion = nn.MSELoss()
            self.mae_loss = nn.L1Loss()
            self.num_epochs = args.num_epochs
            self.exp_name = args.exp_name
            self.model_name = "{}_{}_{}_{}{}{}{}{}{}{}".format(args.model_name,
                                                               self.exp_name,
                                                               self.pred_len,
                                                               self.seed,
                                                               "_denoise" if self.denoising
                                                               else "",
                                                               "_iso" if self.iso
                                                               else "",
                                                               "_predictions" if self.no_noise
                                                               else "",
                                                               "_residual" if self.residual
                                                               else "",
                                                               "_gp" if self.gp
                                                               else "",
                                                               "_add_noise_only_training" if self.input_corrupt_training
                                                               else "")
            self.best_val = 1e10
            self.param_history = []
            self.errors = dict()

            self.best_model = nn.Module()

            self.run_optuna(args)
            self.predictions, self.test_y = self.evaluate()

        def run_optuna(self, args):

            study = optuna.create_study(direction="minimize",
                                        pruner=optuna.pruners.HyperbandPruner())

            study.optimize(self.objective, n_trials=args.n_trials, n_jobs=4)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

        def objective(self, trial):

            src_input_size = 1
            tgt_input_size = 1

            if not os.path.exists(self.model_path):
                try:
                    os.makedirs(self.model_path)
                except FileExistsError:
                    pass

            # hyperparameters

            d_model = trial.suggest_categorical("d_model", [8])
            w_steps = trial.suggest_categorical("w_steps", [4000])
            n_heads = trial.suggest_categorical("n_heads", [8])
            stack_size = trial.suggest_categorical("stack_size", [2])
            nu = trial.suggest_categorical("nu", [0])

            if [d_model, stack_size, w_steps, n_heads, nu] in self.param_history:
                raise optuna.exceptions.TrialPruned()
            self.param_history.append([d_model, stack_size, w_steps, n_heads, nu])

            d_k = int(d_model / n_heads)

            assert d_model % d_k == 0

            config = src_input_size, tgt_input_size, d_model, n_heads, d_k, stack_size

            model = Forecast_denoising(model_name=self.model_name,
                                       config=config,
                                       gp=self.gp,
                                       denoise=self.denoising,
                                       device=self.device,
                                       seed=self.seed,
                                       pred_len=self.pred_len,
                                       attn_type=self.attn_type,
                                       no_noise=self.no_noise,
                                       residual=self.residual,
                                       input_corrupt=self.input_corrupt_training,
                                       nu=nu).to(self.device)

            optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, w_steps)

            val_loss = 1e10

            print("Start Training...")

            for epoch in range(self.num_epochs):

                total_loss = 0
                model.train()

                for train_enc, train_dec, train_y in self.dataloader_obj.train_loader:
                    output_fore_den, loss_train = \
                            model(train_enc.to(self.device), train_dec.to(self.device), train_y.to(self.device))

                    total_loss += loss_train.item()
                    optimizer.zero_grad()
                    loss_train.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step_and_update_lr()

                model.eval()
                test_loss = 0
                for valid_enc, valid_dec, valid_y in self.dataloader_obj.valid_loader:

                    output, loss_eval = \
                            model(valid_enc.to(self.device), valid_dec.to(self.device), valid_y.to(self.device))

                    test_loss += loss_eval.item()

                if epoch % 5 == 0:
                    print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))
                    print("val loss: {:.4f}".format(test_loss))

                if test_loss < val_loss:
                    val_loss = test_loss
                    if val_loss < self.best_val:
                        self.best_val = val_loss
                        self.best_model = model
                        torch.save({'model_state_dict': self.best_model.state_dict()},
                                   os.path.join(self.model_path, "{}_{}".format(self.model_name, self.seed)))

            return val_loss

        def write_to_file(self, mse_loss, mae_loss, mse_std=None, mae_std=None):

            errors = {self.model_name: {'MSE': f"{mse_loss:.3f}", 'MAE': f"{mae_loss: .3f}"}}
            if mse_std is not None:
                errors = {self.model_name: {'MSE': f"{mse_loss:.3f}", 'MAE': f"{mae_loss: .3f}",
                          'MSE_std': f"{mse_std:.3f}", 'MAE_std': f"{mae_std:.3f}"}}

            print(errors)
            error_path = "ICLR_errors_{}.csv".format(self.exp_name)

            df = pd.DataFrame.from_dict(errors, orient='index')

            if os.path.exists(error_path):

                df_old = pd.read_csv(error_path)
                df_new = pd.concat([df_old, df], axis=0)
                df_new.to_csv(error_path)
            else:
                df.to_csv(error_path)

        def evaluate(self):

            self.best_model.eval()
            total_b = len(self.dataloader_obj.test_loader)
            _, _, test_y = next(iter(self.dataloader_obj.test_loader))

            predictions = np.zeros((total_b, test_y.shape[0], self.pred_len))
            test_y_tot = np.zeros((total_b, test_y.shape[0], self.pred_len))

            j = 0

            for test_enc, test_dec, test_y in self.dataloader_obj.test_loader:

                output, _ = self.best_model(test_enc.to(self.device), test_dec.to(self.device))
                predictions[j] = output.squeeze(-1).cpu().detach().numpy()
                test_y_tot[j] = test_y[:, -self.pred_len:].squeeze(-1).cpu().detach().numpy()
                j += 1

            predictions = torch.from_numpy(predictions)
            test_y = torch.from_numpy(test_y_tot)

            mse_loss_mean = nn.MSELoss()(predictions, test_y).item()

            mae_loss_mean = nn.L1Loss()(predictions, test_y).item()

            mse_loss = nn.MSELoss(reduction='none')(predictions, test_y)
            mae_loss = nn.L1Loss(reduction='none')(predictions, test_y)

            mse_loss_std = np.sqrt(mse_loss.std().item()) / np.sqrt(self.pred_len)
            mae_loss_std = np.sqrt(mae_loss.std().item()) / np.sqrt(self.pred_len)

            self.write_to_file(mse_loss_mean, mae_loss_mean, mse_loss_std, mae_loss_std)

            return predictions, test_y


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--attn_type", type=str, default='autoformer')
    parser.add_argument("--model_name", type=str, default="autoformer")
    parser.add_argument("--exp_name", type=str, default='traffic')
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--denoising", type=str, default="True")
    parser.add_argument("--gp", type=str, default="True")
    parser.add_argument("--residual", type=str, default="False")
    parser.add_argument("--no-noise", type=str, default="False")
    parser.add_argument("--iso", type=str, default="False")
    parser.add_argument("--input_corrupt_training", type=str, default="False")
    parser.add_argument("--num_epochs", type=int, default=1)

    args = parser.parse_args()

    random.seed(1234)
    seeds = [random.randint(1000, 9999) for _ in range(1)]
    pred_lens = [96, 192, 336, 720]
    pred_lens.reverse()

    for pred_len in pred_lens:

        for seed in seeds:

            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            Train(args.exp_name, args, pred_len, seed)


if __name__ == '__main__':
    main()
