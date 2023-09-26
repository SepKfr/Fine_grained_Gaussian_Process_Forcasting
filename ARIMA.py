import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from new_data_loader import DataLoader
import statsmodels.api as sm


def run_ARIMA(exp_name, pred_len):

    target_col = {"traffic": "values",
                  "electricity": "power_usage",
                  "exchange": "value",
                  "solar": "Power(MW)",
                  "air_quality": "NO2"
                  }

    dataloader_obj = DataLoader(exp_name,
                             max_encoder_length=8 * 24,
                             target_col=target_col[exp_name],
                             pred_len=pred_len,
                             max_train_sample=32000,
                             max_test_sample=3840,
                             batch_size=256)

    total_b = len(dataloader_obj.test_loader)
    _, _, test_y = next(iter(dataloader_obj.test_loader))

    predictions = np.zeros((total_b*256, pred_len))
    test_y_tot = np.zeros((total_b*256, pred_len))

    j = 0

    for test_enc, test_dec, test_y in dataloader_obj.test_loader:

        xes = torch.cat([test_enc, test_dec], dim=1).squeeze(-1).detach().numpy()
        yes = test_y.squeeze(-1).detach().numpy()
        for x, y in zip(xes, yes):
            arima_model = sm.tsa.ARIMA(x, order=(1, 1, 1))
            arima_results = arima_model.fit()
            forecasts = arima_results.forecast(steps=pred_len)
            predictions[j] = forecasts[0]
            test_y_tot[j] = y[-pred_len:]
            j += 1

    predictions = torch.from_numpy(predictions.reshape(-1, 1))
    test_y = torch.from_numpy(test_y_tot.reshape(-1, 1))

    mse_loss = F.mse_loss(predictions, test_y).item()

    mae_loss = F.l1_loss(predictions, test_y).item()

    errors = {"ARIMA_{}".format(pred_len): {'MSE': f"{mse_loss:.3f}", 'MAE': f"{mae_loss: .3f}"}}
    print(errors)

    error_path = "Final_errors_{}.csv".format(exp_name)

    df = pd.DataFrame.from_dict(errors, orient='index')

    if os.path.exists(error_path):

        df_old = pd.read_csv(error_path)
        df_new = pd.concat([df_old, df], axis=0)
        df_new.to_csv(error_path)
    else:
        df.to_csv(error_path)


for data_set in ["traffic", "electricity", "solar", "air_quality"]:
    for pred_len in [24, 48, 96, 192]:
        run_ARIMA(data_set, pred_len)