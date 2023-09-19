import os

import pandas as pd
import torch.nn.functional as F
import numpy as np
import optuna
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from pytorch_forecasting import TemporalFusionTransformer, DeepAR, NHiTS, NBeats, MQF2DistributionLoss

import argparse

from torch import nn
from torch.optim import Adam

from modules.opt_model import NoamOpt
from new_data_loader import DataLoader
print(pl.__version__)
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
                                         max_train_sample=1,
                                         max_test_sample=1,
                                         batch_size=1)

        self.param_history = []
        self.model_path = "models_{}_{}".format(args.exp_name, pred_len)
        self.model_name = "{}_{}_{}".format(args.model_name,
                                            self.exp_name,
                                            self.seed)
        self.num_epochs = args.num_epochs
        self.train_model(32)

    def train_model(self, d_model):

        if "NBeats" in self.model_name:
            model = NBeats.from_dataset(self.dataloader_obj.train_dataset,
                                        learning_rate=3e-2,
                                        weight_decay=1e-2,
                                        widths=[32, 512],
                                        backcast_loss_ratio=0.1).to(self.device)

        else:
            model = NHiTS.from_dataset(
                          self.dataloader_obj.train_dataset,
                            learning_rate=5e-3,
                            log_interval=10,
                            log_val_interval=1,
                            weight_decay=1e-2,
                            backcast_loss_ratio=0.0,
                            hidden_size=64,
                            optimizer="AdamW",

            ).to(self.device)

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        trainer = pl.Trainer(
            max_epochs=5,
            accelerator="cpu",
            enable_model_summary=True,
            gradient_clip_val=1.0,
            callbacks=[early_stop_callback],
            limit_train_batches=30,
            enable_checkpointing=True,
        )

        trainer.fit(
            model,
            train_dataloaders=self.dataloader_obj.train_loader2,
            val_dataloaders=self.dataloader_obj.valid_loader2,
        )

        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model = NBeats.load_from_checkpoint(best_model_path)

        actuals = torch.cat([y[0] for x, y in iter(self.dataloader_obj.test_loader2)])
        predictions = best_model.predict(self.dataloader_obj.test_loader2)

        predictions = torch.from_numpy(predictions.reshape(-1, 1))
        test_y = torch.from_numpy(actuals.reshape(-1, 1))

        mse_loss = F.mse_loss(predictions, test_y).item()

        mae_loss = F.l1_loss(predictions, test_y).item()

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

        return model


parser = argparse.ArgumentParser(description="preprocess argument parser")
parser.add_argument("--exp_name", type=str, default='traffic')
parser.add_argument("--model_name", type=str, default='NHiTS')
parser.add_argument("--cuda", type=str, default="cuda:0")
parser.add_argument("--n_trials", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=2021)
args = parser.parse_args()

for pred_len in [96, 192]:
    baseline = Baselines(args, pred_len)