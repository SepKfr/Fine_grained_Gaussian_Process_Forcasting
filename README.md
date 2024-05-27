# Coarse and Fine-grained Forecasting Via Gaussian Process Blurring Effect

This repository contains the code and repoducibility instructions of our ```Coarse and Fine Grained Forecassting Via Gaussian Process Blurring Effect``` paper submitted to TMLR.

## Abstract 

Time series forecasting is a challenging task due to the existence of complex and dynamic temporal dependencies, leading to inaccurate predictions even by the most advanced models. While increasing training data is a common approach to enhance accuracy, it is often a limitted source. In contrast, we are building on successful denoising approaches for image generation by proposing an end-to-end forecast-blur-denoise framework. By training the parameters of the blur model for best end-to-end performance, we advocate for a clear division of tasks between the forecasting and denoising models. This encourages the forecasting model to learn the coarse-grained behavior, while the denoising model is filling in the blurred fine-grained details.

## Requirements

```
python >= 3.10.4
torch >= 1.13.0
optuna >= 3.0.4
gpytorch >= 1.9.0
numpy >= 1.23.5
```

## Data Loading 

```
python new_data_loader.py --expt_name solar
```

After running the above python script, a csv file containing the solar dataset is created. In order to generate csv files regarding our other datasets, simply change the expt_name to the desired datatset. You can choose from ```{traffic, electricity, solar}```.

## How to run:
```
Command line arguments:

exp_name: str    the name of the dataset
model_name:str   name of the end-to-end forecasting model (for saving model purpose)
attn_type:str    the type of the attention model (ATA, autofomer, informer, conv_attn)
denoising:bool   whether to use denoising
gp:bool          whether to use our proposed GP noise model 
seed:int         random seed value
cuda:str         which GPU

# one example with traffic dataset and Autoformer forecasting model when apply corruption and denoising with our proposed GP model 

python train.py --exp_name solar --model_name AutoDG--attn_type autofromer --denoising True --gp True --seed 4293 --cuda cuda:0
```

## 
