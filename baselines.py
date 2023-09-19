import os

import pandas as pd
import torch.nn.functional as F
import numpy as np
import optuna
import pytorch_forecasting
import torch
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from pytorch_forecasting import TemporalFusionTransformer, DeepAR, NHiTS, NBeats

import argparse

from torch import nn
from torch.optim import Adam

from modules.opt_model import NoamOpt
from new_data_loader import DataLoader


class Baselines:
    def __init__(self, args, pred_len):

        target_col = {"traffic": "values",
                      "electricity": "power_usage",
                      "exchange": "value",
                      "solar": "Power(MW)",
                      "air_quality": "NO2"
                      }

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
                                         max_train_sample=12800,
                                         max_test_sample=1280,
                                         batch_size=128)

        self.param_history = []
        self.model_path = "models_{}_{}".format(args.exp_name, pred_len)
        self.model_name = "{}_{}_{}".format(args.model_name,
                                            self.exp_name,
                                            self.seed)
        self.num_epochs = args.num_epochs
        self.run_optuna(args)
        self.evaluate()

    def get_model(self, d_model):

        if "NBeats" in self.model_name:
            model = NBeats.from_dataset(self.dataloader_obj.train_dataset,
                                        learning_rate=3e-2,
                                        weight_decay=1e-2,
                                        widths=[32, 512],
                                        backcast_loss_ratio=0.1).to(self.device)
        else:
            model = NHiTS.from_dataset(
                          self.dataloader_obj.train_dataset,
                          hidden_size=d_model).to(self.device)
        return model

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

        src_input_size = 1
        tgt_input_size = 1

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # hyperparameters

        d_model = trial.suggest_categorical("d_model", [32, 64])
        w_steps = trial.suggest_categorical("w_steps", [4000])

        if [d_model] in self.param_history:
            raise optuna.exceptions.TrialPruned()
        self.param_history.append([d_model])

        model = self.get_model(d_model)

        optimizer = NoamOpt(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 2, d_model, w_steps)

        val_loss = 1e10

        print("Start Training...")

        for epoch in range(self.num_epochs):

            total_loss = 0
            model.train()

            for x, y in self.dataloader_obj.train_loader2:

                x = {key: value.to(self.device) for key, value in x.items()}
                x["target_scale"] = torch.ones_like(x["target_scale"])
                outputs = model(x)
                if len(outputs["prediction"].shape) > 2:
                    outputs = outputs["prediction"][:, :, -1]
                else:
                    outputs = outputs["prediction"]
                loss_train = nn.MSELoss()(y[0].to(self.device),
                                          outputs.to(self.device))

                total_loss += loss_train.item()
                optimizer.zero_grad()
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step_and_update_lr()

            model.eval()
            test_loss = 0

            for valid_x, valid_y in self.dataloader_obj.valid_loader2:
                valid_x = {key: value.to(self.device) for key, value in valid_x.items()}
                valid_x["target_scale"] = torch.ones_like(valid_x["target_scale"])
                outputs = model(valid_x)
                if len(outputs["prediction"].shape) > 2:
                    outputs = outputs["prediction"][:, :, -1]
                else:
                    outputs = outputs["prediction"]
                loss_eval = nn.MSELoss()(valid_y[0].to(self.device),
                                         outputs.to(self.device))

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

        for x, y in self.dataloader_obj.test_loader2:
            x = {key: value.to(self.device) for key, value in x.items()}
            x["target_scale"] = torch.ones_like(x["target_scale"])

            outputs = self.best_model(x)
            if len(outputs["prediction"].shape) > 2:
                outputs = outputs["prediction"][:, :, -1]
            else:
                outputs = outputs["prediction"]
            predictions[j] = outputs.cpu().detach().numpy()
            test_y_tot[j] = y[0][:, -self.pred_len:].cpu().detach().numpy()
            j += 1

        predictions = torch.from_numpy(predictions.reshape(-1, 1))
        test_y = torch.from_numpy(test_y_tot.reshape(-1, 1))
        normaliser = test_y.abs().mean()

        mse_loss = F.mse_loss(predictions, test_y).item() / normaliser

        mae_loss = F.l1_loss(predictions, test_y).item() / normaliser

        errors = {self.model_name: {'MSE': mse_loss.item(), 'MAE': mae_loss.item()}}
        print(errors)

        error_path = "Final_errors-{}.csv".format(self.exp_name)

        df = pd.DataFrame.from_dict(errors, orient='index')

        if os.path.exists(error_path):

            df_old = pd.read_csv(error_path)
            df_new = pd.concat([df_old, df], axis=0)
            df_new.to_csv(error_path)
        else:
            df.to_csv(error_path)


parser = argparse.ArgumentParser(description="preprocess argument parser")
parser.add_argument("--exp_name", type=str, default='traffic')
parser.add_argument("--model_name", type=str, default='NBeats')
parser.add_argument("--cuda", type=str, default="cuda:0")
parser.add_argument("--n_trials", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=2021)
args = parser.parse_args()

for pred_len in [96, 192]:
    baseline = Baselines(args, pred_len)