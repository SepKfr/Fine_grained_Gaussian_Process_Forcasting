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
from data_loader import ExperimentConfig
from Utils.base_train import batch_sampled_data
from modules.opt_model import NoamOpt

torch.autograd.set_detect_anomaly(True)

with gpytorch.settings.num_likelihood_samples(1):
    class Train:
        def __init__(self, data, args, pred_len, seed):

            config = ExperimentConfig(pred_len, args.exp_name)
            self.input_corrupt = args.input_corrupt_training
            self.denoising = args.denoising if not self.input_corrupt else False
            self.gp = args.gp
            self.no_noise = args.no_noise
            self.residual = args.residual
            self.iso = args.iso
            self.data = data
            self.len_data = len(data)
            self.formatter = config.make_data_formatter()
            self.params = self.formatter.get_experiment_params()
            self.total_time_steps = self.params['total_time_steps']
            self.num_encoder_steps = self.params['num_encoder_steps']
            self.column_definition = self.params["column_definition"]
            self.pred_len = pred_len
            self.seed = seed
            self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
            self.model_path = "models_{}_{}".format(args.exp_name, pred_len)
            self.model_params = self.formatter.get_default_model_params()
            self.batch_size = self.model_params['minibatch_size'][0]
            self.attn_type = args.attn_type
            self.criterion = nn.MSELoss()
            self.mae_loss = nn.L1Loss()
            self.num_epochs = args.num_epochs
            self.model_name = "{}_{}_{}_{}{}{}{}{}{}{}".format(args.model_name, args.exp_name, pred_len, seed,
                                                              "_denoise" if self.denoising else "",
                                                              "_gp" if self.gp else "",
                                                              "_predictions" if self.no_noise else "",
                                                              "_iso" if self.iso else "",
                                                              "_residual" if self.residual else "",
                                                              "_input_corrupt" if self.input_corrupt else "")
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.best_val = 1e10
            self.param_history = []
            self.exp_name = args.exp_name
            self.best_model = nn.Module()
            self.train, self.valid, self.test, self.n_batches = self.split_data()
            self.run_optuna(args)
            self.evaluate()

        def split_data(self):

            data = self.formatter.transform_data(self.data)

            train_max, valid_max = self.formatter.get_num_samples_for_calibration()
            n_batches = int(train_max / self.batch_size)
            max_samples = (train_max, valid_max)

            train, valid, test = batch_sampled_data(data, 0.8 if not self.exp_name == "exchange" else 0.4,
                                                    max_samples, self.params['total_time_steps'],
                                                    self.params['num_encoder_steps'], self.pred_len,
                                                    self.params["column_definition"],
                                                    self.batch_size)

            return train, valid, test, n_batches

        def run_optuna(self, args):

            study = optuna.create_study(study_name=args.model_name,
                                        direction="minimize")
            study.optimize(self.objective, n_trials=args.n_trials, n_jobs=1)

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

            train_enc, train_dec, train_y = next(iter(self.train))

            src_input_size = train_enc.shape[2]
            tgt_input_size = train_dec.shape[2]

            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

            # hyperparameters

            d_model = trial.suggest_categorical("d_model", [8, 16])
            w_steps = trial.suggest_categorical("w_steps", [1000, 8000])
            stack_size = trial.suggest_categorical("stack_size", [1, 2])

            n_heads = self.model_params['num_heads']

            if [d_model, stack_size, w_steps] in self.param_history:
                raise optuna.exceptions.TrialPruned()
            self.param_history.append([d_model, stack_size, w_steps])

            d_k = int(d_model / n_heads)

            assert d_model % d_k == 0

            config = src_input_size, tgt_input_size, d_model, n_heads, d_k, stack_size

            train_enc, train_dec, _ = next(iter(self.train))

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
                                       input_corrupt=self.input_corrupt).to(self.device)

            optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, w_steps)

            val_loss = 1e10
            mse_losses_train = []
            mse_losses_valid = []
            for epoch in range(self.num_epochs):

                total_loss = 0
                total_loss_mse = 0
                model.train()

                for train_enc, train_dec, train_y in self.train:

                    output_fore_den, loss, mse_loss_train = model(train_enc.to(self.device), train_dec.to(self.device),
                                                  train_y.to(self.device))
                    total_loss += loss.item()
                    total_loss_mse += mse_loss_train.item()
                    mse_losses_train.append(total_loss_mse)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step_and_update_lr()

                model.eval()
                test_loss = 0
                test_loss_mse = 0
                for valid_enc, valid_dec, valid_y in self.valid:
                    output, loss, mse_loss_val = model(valid_enc.to(self.device), valid_dec.to(self.device), valid_y.to(self.device))
                    test_loss += loss.item()
                    test_loss_mse += mse_loss_val.item()
                    mse_losses_valid.append(test_loss_mse)
                if epoch % 5 == 0:
                    print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))
                    print("val loss: {:.4f}".format(test_loss))

                if test_loss < val_loss:
                    val_loss = test_loss
                    if val_loss < self.best_val:
                        self.best_val = val_loss
                        self.best_model = model
                        torch.save({'model_state_dict': self.best_model.state_dict()},
                                   os.path.join(self.model_path, "{}".format(self.model_name)))
                path_t_losses = "losses_lists"
                if not os.path.exists(path_t_losses):
                    os.makedirs(path_t_losses)

                np.save(os.path.join(path_t_losses, "{}_mse_losses_train.npy".format(self.model_name)), mse_losses_train)
                np.save(os.path.join(path_t_losses, "{}_mse_losses_valid.npy".format(self.model_name)), mse_losses_valid)

            return val_loss

        def evaluate(self):

            self.best_model.eval()

            _, _, test_y = next(iter(self.test))
            total_b = len(list(iter(self.test)))

            predictions = np.zeros((total_b, test_y.shape[0], self.pred_len))
            test_y_tot = np.zeros((total_b, test_y.shape[0], self.pred_len))

            j = 0

            for test_enc, test_dec, test_y in self.test:
                output, _, _ = self.best_model(test_enc.to(self.device), test_dec.to(self.device))
                predictions[j] = output.squeeze(-1).cpu().detach().numpy()
                test_y_tot[j] = test_y[:, -self.pred_len:, :].squeeze(-1).cpu().detach().numpy()
                j += 1

            predictions = torch.from_numpy(predictions.reshape(-1, 1))
            test_y = torch.from_numpy(test_y_tot.reshape(-1, 1))
            normaliser = test_y.abs().mean()

            test_loss = F.mse_loss(predictions, test_y).item() / normaliser
            mse_loss = test_loss

            mae_loss = F.l1_loss(predictions, test_y).item() / normaliser
            mae_loss = mae_loss

            errors = {self.model_name: {'MSE': f"{mse_loss:.3f}", 'MAE': f"{mae_loss: .3f}"}}
            print(errors)

            error_path = "Long_horizon_Previous_set_up_Final_errors_{}.csv".format(self.exp_name)

            df = pd.DataFrame.from_dict(errors, orient='index')

            if os.path.exists(error_path):

                df_old = pd.read_csv(error_path)
                df_new = pd.concat([df_old, df], axis=0)
                df_new.to_csv(error_path)
            else:
                df.to_csv(error_path)

    def main():

        parser = argparse.ArgumentParser(description="preprocess argument parser")
        parser.add_argument("--attn_type", type=str, default='fedformer')
        parser.add_argument("--model_name", type=str, default="fedformer")
        parser.add_argument("--exp_name", type=str, default='exchange')
        parser.add_argument("--cuda", type=str, default="cuda:0")
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument("--n_trials", type=int, default=50)
        parser.add_argument("--denoising", type=lambda x: str(x).lower() == "true", default="True")
        parser.add_argument("--gp", type=lambda x: str(x).lower() == "true", default="True")
        parser.add_argument("--residual", type=lambda x: str(x).lower() == "true", default="False")
        parser.add_argument("--no-noise", type=lambda x: str(x).lower() == "true", default="False")
        parser.add_argument("--input_corrupt_training", type=lambda x: str(x).lower() == "true", default="False")
        parser.add_argument("--iso", type=lambda x: str(x).lower() == "true", default="False")
        parser.add_argument("--num_epochs", type=int, default=3)

        args = parser.parse_args()

        data_csv_path = "{}.csv".format(args.exp_name)
        raw_data = pd.read_csv(data_csv_path, dtype={'date': str})

        random.seed(1234)

        seeds = [random.randint(1000, 9999) for _ in range(3)]
        for seed in seeds:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            for pred_len in [24, 48, 72, 96]:
                Train(raw_data, args, pred_len, seed)


    if __name__ == '__main__':
        main()
