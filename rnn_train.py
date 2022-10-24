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


class ModelData:

    def __init__(self, enc, dec, y_true, y_id, device):
        self.enc = enc.to(device)
        self.dec = dec.to(device)
        self.y_true = y_true.to(device)
        self.y_id = y_id


def create_config(hyper_parameters):
    prod = list(itertools.product(*hyper_parameters))
    return list(random.sample(set(prod), len(prod)))


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
        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        self.model_path = "models_{}_{}".format(args.exp_name, pred_len)
        self.model_params = self.formatter.get_default_model_params()
        self.batch_size = self.model_params['minibatch_size'][0]
        self.attn_type = args.attn_type
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

    def define_model(self, d_model, stack_size, src_input_size, tgt_input_size):

        stack_size, d_model = stack_size, d_model

        model = RNN(n_layers=stack_size,
                    hidden_size=d_model,
                    src_input_size=src_input_size,
                    tgt_input_size=tgt_input_size,
                    rnn_type="lstm",
                    device=self.device,
                    d_r=0,
                    seed=self.seed,
                    pred_len=self.pred_len)
        model.to(self.device)
        return model

    def sample_data(self, max_samples, data):

        sample_data = batch_sampled_data(data, max_samples, self.params['total_time_steps'],
                               self.params['num_encoder_steps'], self.pred_len, self.params["column_definition"],
                                         self.seed)
        sample_data = ModelData(torch.from_numpy(sample_data['enc_inputs']),
                                                 torch.from_numpy(sample_data['dec_inputs']),
                                                 torch.from_numpy(sample_data['outputs']),
                                                 sample_data['identifier'], self.device)
        return sample_data

    def split_data(self):

        data = self.formatter.transform_data(self.data)

        train_max, valid_max = self.formatter.get_num_samples_for_calibration()
        max_samples = (train_max, valid_max)

        train, valid, test = batch_sampled_data(data, 0.8, max_samples, self.params['total_time_steps'],
                                                self.params['num_encoder_steps'], self.pred_len,
                                                self.params["column_definition"],
                                                self.device)

        trn_batching = batching(self.batch_size, train.enc, train.dec, train.y_true, train.y_id)
        valid_batching = batching(self.batch_size, valid.enc, valid.dec, valid.y_true, valid.y_id)
        test_batching = batching(self.batch_size, test.enc, test.dec, test.y_true, test.y_id)

        trn = ModelData(trn_batching[0], trn_batching[1], trn_batching[2], trn_batching[3], self.device)
        valid = ModelData(valid_batching[0], valid_batching[1], valid_batching[2], valid_batching[3], self.device)
        test = ModelData(test_batching[0], test_batching[1], test_batching[2], test_batching[3], self.device)

        return trn, valid, test

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
        src_input_size = self.train.enc.shape[3]
        tgt_input_size = self.train.dec.shape[3]
        n_batches_train = self.train.enc.shape[0]
        n_batches_valid = self.valid.enc.shape[0]

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        d_model = trial.suggest_categorical("d_model", [16, 32])
        stack_size = self.model_params['stack_size'][0]

        if d_model in self.param_history or self.n_distinct_trial > 4:
            raise optuna.exceptions.TrialPruned()
        else:
            self.n_distinct_trial += 1
        self.param_history.append(d_model)

        model = RNN(n_layers=stack_size,
                    hidden_size=d_model,
                    src_input_size=src_input_size,
                    tgt_input_size=tgt_input_size,
                    rnn_type="lstm",
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
            for batch_id in range(n_batches_train):

                output = model(self.train.enc[batch_id], self.train.dec[batch_id])
                loss = self.criterion(output, self.train.y_true[batch_id]) + self.mae_loss(output, self.train.y_true[batch_id])

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))

            model.eval()
            test_loss = 0
            for j in range(n_batches_valid):

                outputs = model(self.valid.enc[j], self.valid.dec[j])
                loss = self.criterion(outputs, self.valid.y_true[j]) + self.mae_loss(outputs, self.valid.y_true[j])
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
        predictions = torch.zeros(self.test.y_true.shape[0], self.test.y_true.shape[1], self.test.y_true.shape[2])
        targets_all = torch.zeros(self.test.y_true.shape[0], self.test.y_true.shape[1], self.test.y_true.shape[2])
        n_batches_test = self.test.enc.shape[0]

        for j in range(n_batches_test):

            output = self.best_model(self.test.enc[j], self.test.dec[j])
            predictions[j] = output.squeeze(-1)
            targets_all[j] = self.test.y_true[j].squeeze(-1)
            '''output_map = inverse_output(output, self.test.y_true[j], self.test.y_id[j])
            p = self.formatter.format_predictions(output_map["predictions"])
            if p is not None:
                if self.exp_name == "covid":
                    tp = 'int'
                else:
                    tp = 'float'
                forecast = torch.from_numpy(extract_numerical_data(p).to_numpy().astype(tp)).to(self.device)

                predictions[j, :forecast.shape[0], :] = forecast

                targets = torch.from_numpy(extract_numerical_data(
                    self.formatter.format_predictions(output_map["targets"])).to_numpy().astype(tp)).to(self.device)

                targets_all[j, :targets.shape[0], :] = targets'''

        predictions = predictions.to('cpu')
        targets_all = targets_all.to('cpu')
        test_loss = self.criterion(predictions, targets_all).item()
        normaliser = targets_all.abs().mean()
        test_loss = test_loss / normaliser

        mae_loss = self.mae_loss(predictions, targets_all).item()
        normaliser = targets_all.abs().mean()
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
    parser.add_argument("--attn_type", type=str, default='lstm')
    parser.add_argument("--name", type=str, default="lstm")
    parser.add_argument("--exp_name", type=str, default='covid')
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