import gpytorch
from gpytorch.mlls import PredictiveLogLikelihood, DeepApproximateMLL, VariationalELBO
from denoising_model.GPModel import ExactGPModel
from forecast_denoising import Forecast_denoising
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
import argparse
import json
import os
import torch.nn.functional as F
import random
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from data_loader import ExperimentConfig
from Utils.base_train import batch_sampled_data
from modules.opt_model import NoamOpt
from sdtw_cuda_loss import SoftDTW

torch.autograd.set_detect_anomaly(True)


class Train:
    def __init__(self, data, args, pred_len):

        config = ExperimentConfig(pred_len, args.exp_name)
        self.denoising = True if args.denoising == "True" else False
        self.gp = True if args.gp == "True" else False
        self.no_noise = True if args.no_noise == "True" else False
        self.residual = True if args.residual == "True" else False
        self.data = data
        self.len_data = len(data)
        self.formatter = config.make_data_formatter()
        self.params = self.formatter.get_experiment_params()
        self.total_time_steps = self.params['total_time_steps']
        self.num_encoder_steps = self.params['num_encoder_steps']
        self.column_definition = self.params["column_definition"]
        self.pred_len = pred_len
        self.seed = args.seed
        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        self.model_path = "models_{}_{}".format(args.exp_name, pred_len)
        self.model_params = self.formatter.get_default_model_params()
        self.batch_size = self.model_params['minibatch_size'][0]
        self.attn_type = args.attn_type
        self.criterion = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.num_epochs = args.num_epochs
        self.model_name = args.model_name
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.best_val = 1e10
        self.param_history = []
        self.erros = dict()
        self.exp_name = args.exp_name
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        self.best_model = nn.Module()
        self.train, self.valid, self.test, self.n_batches = self.split_data()
        self.run_optuna(args)
        self.evaluate()

    def split_data(self):

        data = self.formatter.transform_data(self.data)

        train_max, valid_max = self.formatter.get_num_samples_for_calibration()
        n_batches = int(train_max / self.batch_size)
        max_samples = (train_max, valid_max)

        train, valid, test = batch_sampled_data(data, 0.8, max_samples, self.params['total_time_steps'],
                                                self.params['num_encoder_steps'], self.pred_len,
                                                self.params["column_definition"],
                                                self.batch_size)

        return train, valid, test, n_batches

    def run_optuna(self, args):

        study = optuna.create_study(study_name=args.model_name,
                                    direction="minimize", pruner=optuna.pruners.HyperbandPruner(),
                                    sampler=TPESampler(seed=1234))
        study.optimize(self.objective, n_trials=args.n_trials)

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

        d_model = trial.suggest_categorical("d_model", [32])

        w_steps = trial.suggest_categorical("w_steps", [1000])
        stack_size = trial.suggest_categorical("stack_size", [1])

        n_heads = self.model_params['num_heads']

        if [d_model, stack_size, w_steps] in self.param_history:
            raise optuna.exceptions.TrialPruned()
        self.param_history.append([d_model, stack_size, w_steps])

        d_k = int(d_model / n_heads)

        assert d_model % d_k == 0

        config = src_input_size, tgt_input_size, d_model, n_heads, d_k, stack_size

        train_enc, train_dec, _ = next(iter(self.train))
        train_x = torch.cat([train_enc, train_dec], dim=1)

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
                                   train_x_shape=[train_x[0].shape[0], train_x[0].shape[1], d_model]).to(self.device)

        if self.gp:
            mll = DeepApproximateMLL(VariationalELBO(model.de_model.deep_gp.likelihood, model.de_model.deep_gp, train_x.shape[-2]))
        optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, w_steps)

        val_loss = 1e10
        for epoch in range(self.num_epochs):

            total_loss = 0
            model.train()

            for train_enc, train_dec, train_y in self.train:

                if self.gp:
                    with gpytorch.settings.num_likelihood_samples(1):
                        output_fore_den, dist = model(train_enc.to(self.device), train_dec.to(self.device))
                else:
                    output_fore_den, dist = model(train_enc.to(self.device), train_dec.to(self.device))
                if dist is not None:
                    mll_error = -mll(dist, train_y[:, :-self.pred_len, :].to(self.device).permute(2, 0, 1)).mean()
                else:
                    mll_error = 0

                #loss_train = nn.MSELoss()(output_fore_den, train_y[:, -self.pred_len:, :].to(self.device)) + 0.01 * mll_error
                loss_train = self.sdtw(output_fore_den, train_y[:, -self.pred_len:, :].to(self.device)) + 0.01 * mll_error

                total_loss += loss_train.item()
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step_and_update_lr()

            model.eval()
            test_loss = 0
            for valid_enc, valid_dec, valid_y in self.valid:
                if self.gp:
                    with gpytorch.settings.num_likelihood_samples(1):
                        output, _ = model(valid_enc.to(self.device), valid_dec.to(self.device))
                else:
                    output, _ = model(valid_enc.to(self.device), valid_dec.to(self.device))
                loss_eval = nn.MSELoss()(output, valid_y[:, -self.pred_len:, :].to(self.device))

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

    def evaluate(self):

        self.best_model.eval()

        _, _, test_y = next(iter(self.test))
        total_b = len(list(iter(self.test)))

        predictions = np.zeros((total_b, test_y.shape[0], self.pred_len))
        test_y_tot = np.zeros((total_b, test_y.shape[0], self.pred_len))

        j = 0

        for test_enc, test_dec, test_y in self.test:
            if self.gp:
                with gpytorch.settings.num_likelihood_samples(1):
                    output, _ = self.best_model(test_enc.to(self.device), test_dec.to(self.device))
            else:
                output, _ = self.best_model(test_enc.to(self.device), test_dec.to(self.device))
            predictions[j] = output.squeeze(-1).cpu().detach().numpy()
            test_y_tot[j] = test_y[:, -self.pred_len:, :].squeeze(-1).cpu().detach().numpy()
            j += 1

        predictions = torch.from_numpy(predictions.reshape(-1, 1))
        test_y = torch.from_numpy(test_y_tot.reshape(-1, 1))
        normaliser = test_y.abs().mean()

        test_loss = F.mse_loss(predictions, test_y).item() / normaliser
        test_loss = test_loss

        mae_loss = F.l1_loss(predictions, test_y).item() / normaliser
        mae_loss = mae_loss

        print("test loss {:.4f}".format(test_loss))

        self.erros["{}_{}".format(self.model_name, self.seed)] = list()
        self.erros["{}_{}".format(self.model_name, self.seed)].append(float("{:.5f}".format(test_loss)))
        self.erros["{}_{}".format(self.model_name, self.seed)].append(float("{:.5f}".format(mae_loss)))

        error_path = "errors_{}_{}.json".format(self.exp_name, self.pred_len)

        if os.path.exists(error_path):
            with open(error_path) as json_file:
                json_dat = json.load(json_file)
                if json_dat.get("{}_{}".format(self.model_name, self.seed)) is None:
                    json_dat["{}_{}".format(self.model_name, self.seed)] = list()
                json_dat["{}_{}".format(self.model_name, self.seed)].append(float("{:.5f}".format(test_loss)))
                json_dat["{}_{}".format(self.model_name, self.seed)].append(float("{:.5f}".format(mae_loss)))

            with open(error_path, "w") as json_file:
                json.dump(json_dat, json_file)
        else:
            with open(error_path, "w") as json_file:
                json.dump(self.erros, json_file)


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--attn_type", type=str, default='ATA')
    parser.add_argument("--model_name", type=str, default="ATA")
    parser.add_argument("--exp_name", type=str, default='exchange')
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--denoising", type=str, default="True")
    parser.add_argument("--gp", type=str, default="True")
    parser.add_argument("--residual", type=str, default="False")
    parser.add_argument("--no-noise", type=str, default="False")
    parser.add_argument("--num_epochs", type=int, default=5)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_csv_path = "{}.csv".format(args.exp_name)
    raw_data = pd.read_csv(data_csv_path, dtype={'date': str})

    for pred_len in [96, 72, 48, 24]:
        Train(raw_data, args, pred_len)


if __name__ == '__main__':
    main()
