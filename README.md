# Corruption-resilient Forecasting Models

This repository contains the code and repoducibility instructions of our ```Corruption-resilient Forecasting Models``` paper submitted to Neurips 2023.

## Abstract 

Time series forecasting is challenging due to complex temporal dependencies and unobserved external factors, which can lead to incorrect predictions by even the best forecasting models. Using more training data is one way to improve the accuracy, but this source is often limited. In contrast, we are building on successful denoising approaches for image generation. When a time series is corrupted by the common isotropic Gaussian noise, it yields unnaturally behaving time series. To avoid generating unnaturally behaving time series that do not represent the true error mode in modern forecasting models, we propose to employ Gaussian Processes to generate smoothly-correlated corrupted time series. However, instead of directly corrupting the training data, we propose a joint forecast-corrupt-denoise model to encourage the forecasting model to focus on accurately predicting coarse-grained behavior, while the denoising model focuses on capturing fine-grained behavior. All three parts are interacting via a corruption model which enforces the model to be resilient.
Our extensive experiments demonstrate that our proposed corruption-resilient forecasting approach is able to improve the forecasting accuracy of several state-of-the-art forecasting models as well as several other denoising approaches. 

## Supplementary Explanations on How We Obtain the Final Predictions

Please refer to [Supplementary_Explanation](https://github.com/SepKfr/Corruption-resilient-Forecasting-Models/blob/master/Supplementary_Explanation.pdf) for a detailed explanation on how we obtain the final predictions.

## Supplementary of Main Results

Please refer to [Supplementary_Results](https://github.com/SepKfr/Corruption-resilient-Forecasting-Models/blob/master/Supplementary_Results.pdf) for supplementary of main results. 

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
python data_loader.py --expt_name solar
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

python train.py --exp_name solar --model_name autoformer_corrupt_denoise_gp --attn_type autofromer --denoising True --gp True --seed 4293 --cuda cuda:0
```

The notebook file [example_run.ipynb](https://github.com/SepKfr/Corruption-resilient-Forecasting-Models/blob/master/example_run.ipynb) is an example of how to load data as well as training and evaluating three different corruption models 1. GP: our Gaussian-Process-based corruption model 2. Iso: isotropic Gaussian corruptio 3. no: no corruption (only forecasting).

## 
