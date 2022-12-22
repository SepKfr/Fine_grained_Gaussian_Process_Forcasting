from torch.optim.lr_scheduler import OneCycleLR

from models.ACAT_diffusion.ACAT_diff_network import ACATTrainingNetwork
from models.eff_acat import Transformer
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch
import argparse
import json
import os
import torch.nn.functional as F
import itertools
import random
import pandas as pd
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from data.data_loader import ExperimentConfig
from Utils.base_train import batching, batch_sampled_data, inverse_output, ModelData


class NoamOpt:

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class Train:
    def __init__(self, data, args, pred_len):

        config = ExperimentConfig(pred_len, args.exp_name)
        self.diff_model = True if args.diff_model == "True" else False
        self.gp = True if args.gp == "True" else False
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
        self.best_val = 1e10
        self.pr = args.pr
        self.param_history = []
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

        train_enc, train_dec, _ = next(iter(self.train))

        src_input_size = train_enc.shape[2]
        tgt_input_size = train_dec.shape[2]

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        d_model = trial.suggest_categorical("d_model", [16, 32])
        w_steps = trial.suggest_categorical("w_steps", [1000])
        stack_size = trial.suggest_categorical("stack_size", [1])

        n_heads = self.model_params['num_heads']

        kernel = [1, 3, 6, 9] if self.attn_type == "conv_attn" else [1]
        kernel = trial.suggest_categorical("kernel", kernel)

        if [d_model, kernel, stack_size, w_steps] in self.param_history:
            raise optuna.exceptions.TrialPruned()
        self.param_history.append([d_model, kernel, stack_size, w_steps])

        d_k = int(d_model / n_heads)
        assert d_model % d_k == 0

        if self.diff_model:
            model = ACATTrainingNetwork(src_input_size=src_input_size,
                                        tgt_input_size=tgt_input_size,
                                        pred_len=self.pred_len,
                                        d_model=d_model,
                                        d_k=d_k, n_heads=n_heads,
                                        stack_size=stack_size, device=self.device,
                                        attn_type=self.attn_type,
                                        seed=self.seed,
                                        gp=self.gp)
        else:
            model = Transformer(src_input_size=src_input_size,
                                tgt_input_size=tgt_input_size,
                                pred_len=self.pred_len,
                                d_model=d_model,
                                d_ff=d_model * 4,
                                d_k=d_k, d_v=d_k, n_heads=n_heads,
                                n_layers=stack_size, src_pad_index=0,
                                tgt_pad_index=0, device=self.device,
                                attn_type=self.attn_type,
                                seed=self.seed, kernel=kernel)
        model.to(self.device)

        optimizer = Adam(model.parameters())

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-2,
            steps_per_epoch=125,
            epochs=50,
        )

        val_loss = 1e10
        for epoch in range(self.num_epochs):

            total_loss = 0
            model.train()
            for train_enc, train_dec, train_y in self.train:

                if self.diff_model:
                    output = model(train_enc.to(self.device), train_dec.to(self.device), train_y.to(self.device))
                    loss = output
                else:
                    output = model(train_enc.to(self.device), train_dec.to(self.device))
                    loss = F.mse_loss(output, train_y.to(self.device))
                    loss = loss.mean()

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            model.eval()
            test_loss = 0
            for valid_enc, valid_dec, valid_y in self.valid:

                if self.diff_model:
                    output = model(valid_enc.to(self.device), valid_dec.to(self.device), valid_y.to(self.device))
                    loss = output
                else:
                    output = model(valid_enc.to(self.device), valid_dec.to(self.device))
                    loss = F.mse_loss(output, valid_y.to(self.device))
                    loss = loss.mean()

                test_loss += loss.item()

            if epoch % 5 == 0:
                print("Train epoch: {}, loss: {:.4f}".format(epoch, total_loss))
                print("val loss: {:.4f}".format(test_loss))

            if test_loss < val_loss:
                val_loss = test_loss
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self.best_model = model
                    torch.save({'model_state_dict': self.best_model.state_dict()},
                               os.path.join(self.model_path, "{}_{}".format(self.name, self.seed)))

        return val_loss

    def evaluate(self):

        self.best_model.eval()

        _, _, test_y = next(iter(self.test))
        total_b = len(list(iter(self.test)))

        predictions = np.zeros((total_b, test_y.shape[0], test_y.shape[1]))
        test_y_tot = np.zeros((total_b, test_y.shape[0], test_y.shape[1]))

        j = 0

        for test_enc, test_dec, test_y in self.test:

            if self.diff_model:
                output = self.best_model.predict(test_enc.to(self.device), test_dec.to(self.device))
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

        error_path = "errors_{}_{}.json".format(self.exp_name, self.pred_len)

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
    parser.add_argument("--attn_type", type=str, default='autoformer')
    parser.add_argument("--name", type=str, default="autoformer")
    parser.add_argument("--exp_name", type=str, default='traffic')
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--pr", type=float, default=0.8)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--DataParallel", type=bool, default=True)
    parser.add_argument("--diff_model", type=str, default="True")
    parser.add_argument("--gp", type=str, default="False")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_csv_path = "{}.csv".format(args.exp_name)
    raw_data = pd.read_csv(data_csv_path, dtype={'date': str})

    for pred_len in [24, 48, 72, 96]:
        Train(raw_data, args, pred_len)


if __name__ == '__main__':
    main()
