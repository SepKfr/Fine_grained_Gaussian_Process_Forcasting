from models.eff_acat import Transformer
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
import argparse
import json
import os
import itertools
import random
import pandas as pd
import math
import optuna
import torch.nn.functional as F
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from data.data_loader import ExperimentConfig
from Utils.base_train import batching, batch_sampled_data, inverse_output
from models.rnn import RNN
from models.deepar import DeepAr


class ModelData:

    def __init__(self, enc, dec, y_true, y_id, device):
        self.enc = enc.to(device)
        self.dec = dec.to(device)
        self.y_true = y_true.to(device)
        self.y_id = y_id


class Train:
    def __init__(self, data, args, pred_len):

        config = ExperimentConfig(pred_len, args.exp_name)
        self.data = data
        self.len_data = len(data)
        self.formatter = config.make_data_formatter()
        self.params = self.formatter.get_experiment_params()
        self.total_time_steps = self.params['total_time_steps']
        self.num_encoder_steps = self.params['num_encoder_steps']
        self.column_definition = self.params["column_definition"]
        self.pred_len = pred_len
        self.seed = args.seed
        self.pr = 0.8
        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        self.model_path = "models_{}_{}".format(args.exp_name, pred_len)
        self.model_params = self.formatter.get_default_model_params()
        self.batch_size = self.model_params['minibatch_size'][0]
        self.criterion = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.num_epochs = self.params['num_epochs']
        self.name = args.name
        self.param_history = []
        self.n_distinct_trial = 0
        self.erros = dict()
        self.exp_name = args.exp_name
        self.best_model = nn.Module()
        self.train, self.valid, self.test = self.split_data()
        self.run_optuna(args)
        self.evaluate()

    def split_data(self):

        data = self.formatter.transform_data(self.data)

        train_max, valid_max = self.formatter.get_num_samples_for_calibration()
        max_samples = (train_max, valid_max)

        train, valid, test = batch_sampled_data(data, self.pr, max_samples, self.params['total_time_steps'],
                                                self.params['num_encoder_steps'], self.pred_len,
                                                self.params["column_definition"],
                                                self.batch_size)

        return train, valid, test

    def run_optuna(self, args):

        study = optuna.create_study(study_name=args.name,
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

        val_loss = 1e10
        train_enc, train_dec, _ = next(iter(self.train))
        src_input_size = train_enc.shape[2]
        tgt_input_size = train_dec.shape[2]

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        d_model = trial.suggest_categorical("d_model", [16, 32])
        stack_size = self.model_params['stack_size'][0]

        if d_model in self.param_history or self.n_distinct_trial > 4:
            raise optuna.exceptions.TrialPruned()
        else:
            self.n_distinct_trial += 1
        self.param_history.append(d_model)

        if self.name == "DeepAr":
            model = DeepAr(n_layers=stack_size,
                           hidden_size=d_model,
                           src_input_size=src_input_size,
                           device=self.device,
                           d_r=0,
                           seed=self.seed,
                           pred_len=self.pred_len)
        else:
            model = RNN(n_layers=stack_size,
                        hidden_size=d_model,
                        src_input_size=src_input_size,
                        device=self.device,
                        d_r=0,
                        seed=self.seed,
                        pred_len=self.pred_len)

        model.to(self.device)

        optimizer = Adam(model.parameters(), lr=1e-4)

        epoch_start = 0

        val_inner_loss = 1e10

        for epoch in range(epoch_start, self.num_epochs, 1):

            total_loss = 0
            model.train()
            for train_enc, train_dec, train_y in self.train:

                if self.name == "DeepAr":

                    output, mu, sigma = model(train_enc.to(self.device), train_dec.to(self.device))
                    distribution = torch.distributions.normal.Normal(mu, sigma)
                    likelihood = distribution.log_prob(train_y.to(self.device))
                    loss = -torch.mean(likelihood)

                else:

                    output = model(train_enc.to(self.device), train_dec.to(self.device))
                    loss = self.criterion(output, train_y.to(self.device))

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

            model.eval()
            test_loss = 0
            for valid_enc, valid_dec, valid_y in self.valid:

                if self.name == "DeepAr":

                    output, mu, sigma = model(valid_enc.to(self.device), valid_dec.to(self.device))
                    distribution = torch.distributions.normal.Normal(mu, sigma)
                    likelihood = distribution.log_prob(valid_y.to(self.device))
                    loss = -torch.mean(likelihood)

                else:
                    output = model(train_enc.to(self.device), train_dec.to(self.device))
                    loss = self.criterion(output, valid_y.to(self.device))

                test_loss += loss.item()

            print("val loss: {:.4f}".format(test_loss))

            if test_loss < val_inner_loss:
                val_inner_loss = test_loss
                if val_inner_loss < val_loss:
                    val_loss = val_inner_loss
                    self.best_model = model
                    torch.save({'model_state_dict': model.state_dict()},
                               os.path.join(self.model_path, "{}_{}".format(self.name, self.seed)))

        return val_loss

    def evaluate(self):

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        self.best_model.eval()
        _, _, test_y = next(iter(self.test))
        total_b = len(list(iter(self.test)))
        predictions = np.zeros((total_b, test_y.shape[0], test_y.shape[1]))
        test_y_tot = np.zeros((total_b, test_y.shape[0], test_y.shape[1]))
        j = 0

        for test_enc, test_dec, test_y in self.test:
            if self.name == "DeepAr":
                output, _, _ = self.best_model(test_enc.to(self.device), test_dec.to(self.device))
            else:
                output = self.best_model(test_enc.to(self.device), test_dec.to(self.device))
            predictions[j, :output.shape[0], :] = output.squeeze(-1).cpu().detach().numpy()
            test_y_tot[j, :test_y.shape[0], :] = test_y.squeeze(-1).cpu().detach().numpy()
            j += 1

        predictions = torch.from_numpy(predictions)
        test_y = torch.from_numpy(test_y_tot)
        normaliser = test_y.abs().mean()

        test_loss = self.criterion(predictions, test_y).item()
        test_loss = test_loss / normaliser

        mae_loss = self.mae_loss(predictions, test_y).item()
        mae_loss = mae_loss / normaliser

        print("test loss {:.4f}".format(test_loss))

        self.erros["{}_{}".format(self.name, self.seed)] = list()
        self.erros["{}_{}".format(self.name, self.seed)].append(float("{:.5f}".format(test_loss)))
        self.erros["{}_{}".format(self.name, self.seed)].append(float("{:.5f}".format(mae_loss)))

        error_path = "new_Errors_{}_{}.json".format(self.exp_name, self.pred_len)

        if os.path.exists(error_path):
            with open(error_path) as json_file:
                json_dat = json.load(json_file)
                if json_dat.get("{}_{}".format(self.name, self.seed)) is None:
                    json_dat["{}_{}".format(self.name, self.seed)] = list()
                json_dat["{}_{}".format(self.name, self.seed)].append(float("{:.5f}".format(test_loss)))
                json_dat["{}_{}".format(self.name, self.seed)].append(float("{:.5f}".format(mae_loss)))

            with open(error_path, "w") as json_file:
                json.dump(json_dat, json_file)
        else:
            with open(error_path, "w") as json_file:
                json.dump(self.erros, json_file)


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--name", type=str, default="DeepAr")
    parser.add_argument("--exp_name", type=str, default='traffic')
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--DataParallel", type=bool, default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_csv_path = "{}.csv".format(args.exp_name)
    raw_data = pd.read_csv(data_csv_path)

    for pred_len in [24, 48, 72, 96]:
        Train(raw_data, args, pred_len)


if __name__ == '__main__':
    main()