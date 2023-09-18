import os
import random

import pandas as pd
import torch.nn.functional as F
import numpy as np
import optuna
import torch
from neuralforecast import NeuralForecast
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from neuralforecast.models import NBEATS, NHITS, DeepAR
import argparse
from torch import nn
from torch.optim import Adam

from modules.opt_model import NoamOpt
from new_data_loader import DataLoader


class DeepARParams:
    def __init__(self, num_class,
                 embedding_dim,
                 cov_dim,
                 lstm_hidden_dim,
                 lstm_layers,
                 lstm_dropout,
                 sample_times,
                 predict_steps,
                 device):
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.cov_dim = cov_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.sample_times = sample_times
        self.predict_steps = predict_steps
        self.device = device


class Baselines:
    def __init__(self, args, pred_len):

        target_col = {"traffic": "values",
                      "electricity": "power_usage",
                      "exchange": "value",
                      "solar": "Power(MW)",
                      "air_quality": "NO2"
                      }

        self.model_id = args.model_name
        self.exp_name = args.exp_name
        self.seed = args.seed
        self.pred_len = pred_len
        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

        self.best_val = 1e10
        self.best_model = nn.Module()

        self.errors = dict()

        self.dataloader_obj = DataLoader(self.exp_name,
                                         max_encoder_length=96 + 2 * pred_len,
                                         target_col=target_col[self.exp_name],
                                         pred_len=pred_len,
                                         max_train_sample=8,
                                         max_test_sample=8,
                                         batch_size=8)

        self.param_history = []
        self.model_path = "models_{}_{}".format(args.exp_name, pred_len)
        self.model_name = "{}_{}_{}".format(args.model_name,
                                            self.exp_name,
                                            self.seed)
        self.num_epochs = args.num_epochs
        self.run_optuna(args)
        self.evaluate()

    def run_optuna(self, args):

        study = optuna.create_study(study_name=args.model_name,
                                    direction="minimize", pruner=optuna.pruners.HyperbandPruner(),
                                    sampler=TPESampler(seed=1234))
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

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # hyperparameters

        d_model = trial.suggest_categorical("d_model", [32, 64])
        w_steps = trial.suggest_categorical("w_steps", [4000])
        stack_size = trial.suggest_categorical("stack_size", [1, 2] if self.model_id != "NBeats" else [1])

        if [d_model, stack_size, w_steps] in self.param_history:
            raise optuna.exceptions.TrialPruned()
        self.param_history.append([d_model, stack_size, w_steps])

        if self.model_id == "DeepAR":
            model = DeepAR(h=self.pred_len, lstm_n_layers=stack_size, lstm_hidden_size=d_model)
            nf = NeuralForecast(models=[model], freq='M')

        elif self.model_id == "NBeats":
            model = NBEATS(h=self.pred_len, input_size=1)
            nf = NeuralForecast(models=[model], freq='M')
        else:
            model = NHITS(h=self.pred_len, input_size=1)
            nf = NeuralForecast(models=[model], freq='M')

        #optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, w_steps)

        val_loss = 1e10

        print("Start Training...")

        for epoch in range(self.num_epochs):

            total_loss = 0
            #model.train()

            for x_enc, x_dec, y in self.dataloader_obj.train_loader:

                x_enc = x_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                y = y.to(self.device)

                x = torch.cat([x_enc, x_dec], dim=1)
                outputs = nf.fit(x)


                total_loss += loss_train.item()


            model.eval()
            test_loss = 0

            for valid_x_enc, valid_x_dec, valid_y in self.dataloader_obj.valid_loader:
                valid_x_enc = valid_x_enc.to(self.device)
                valid_x_dec = valid_x_dec.to(self.device)
                valid_y = valid_y.to(self.device)
                valid_x = torch.cat([valid_x_enc, valid_x_dec], dim=1)
                _, outputs = model(valid_x)
                outputs = outputs.unsqueeze(-1)
                loss_eval = nn.MSELoss()(valid_y, outputs)

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
        total_b = len(self.dataloader_obj.test_loader)
        _, _, test_y = next(iter(self.dataloader_obj.test_loader))

        predictions = np.zeros((total_b, test_y.shape[0], self.pred_len))
        test_y_tot = np.zeros((total_b, test_y.shape[0], self.pred_len))

        j = 0

        for x_enc, x_dec, y in self.dataloader_obj.test_loader:

            x = torch.cat([x_enc, x_dec], dim=1).to(self.device)

            _, output = self.best_model(x)
            predictions[j] = output.cpu().detach().numpy()

            test_y_tot[j] = y.squeeze(-1).cpu().detach().numpy()
            j += 1

        predictions = torch.from_numpy(predictions.reshape(-1, 1))
        test_y = torch.from_numpy(test_y_tot.reshape(-1, 1))
        normaliser = test_y.abs().mean()

        mse_loss = F.mse_loss(predictions, test_y).item() / normaliser

        mae_loss = F.l1_loss(predictions, test_y).item() / normaliser

        errors = {self.model_name: {'MSE': mse_loss.item(), 'MAE': mae_loss.item()}}
        print(errors)

        error_path = "Final_errors_{}.csv".format(self.exp_name)

        df = pd.DataFrame.from_dict(errors, orient='index')

        if os.path.exists(error_path):

            df_old = pd.read_csv(error_path)
            df_new = pd.concat([df_old, df], axis=0)
            df_new.to_csv(error_path)
        else:
            df.to_csv(error_path)


parser = argparse.ArgumentParser(description="preprocess argument parser")
parser.add_argument("--exp_name", type=str, default='traffic')
parser.add_argument("--model_name", type=str, default='Nhits')
parser.add_argument("--cuda", type=str, default="cuda:0")
parser.add_argument("--n_trials", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=2021)
args = parser.parse_args()

random.seed(1992)
seed = random.randint(1000, 9999)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

for pred_len in [96, 192]:
    baseline = Baselines(args, pred_len)