# Corruption-resilient Forecasting Models

This repository contains the code and repoducibility instructions of our ```Corruption-resilient Forecasting Models``` paper submitted to ICML 2023.

## Abstract 

Challenges in time series forecasting arise due to the existence of complex dependencies on different time scales and the effect of unseen external factors. Despite the recent advances, the existence of erroneous predictions is inevitable, and can only be reduced with additional training data (which is a limited resource). We are building on the recent success of denoising approaches for image generation, and explore the utility of such approaches for time series forecasting. 
Most commonly, scaled isotropic Gaussian noise is used as a corruption process. However, when applied to time series isotropic noise yields unnaturally behaving time series (which are easily disguised) which do not represent the error modes of modern time series methods. 

Instead, we propose to substitute the isotropic Gaussian noise with a Gaussian Process for generating corrupted time series that are smoothly-correlated across time. We hypothesize that a model that is able to denoise such smooth, yet erroneous behavior will be a more resilient forecasting model.
Our extensive experiments demonstrate that our proposed corruption-resilient forecasting approach is able to improve the forecasting accuracy of several state-of-the-art time series forecasting models in 74\% of the cases. 



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

After running the above python script, a csv file containing the solar dataset is created. In order to generate csv files regarding our other datasets, simply change the expt_name to the desired datatset. You can choose from ```{traffic, electricity, air_quality, solar, watershed}```.

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

# one example with traffic dataset and ATA forecasting model when apply corruption and denoising with our proposed GP model 

python train.py --exp_name solar --model_name ATA_gp --attn_type ATA --denoising True --gp True --seed 4293 --cuda cuda:0
```

The notebook file [example_run.ipynb](https://github.com/SepKfr/Corruption-resilient-Forecasting-Models/blob/master/example_run.ipynb) is an example of how to load data as well as training and evaluating three different corruption models 1. GP: our Gaussian-Process-based corruption model 2. Iso: isotropic Gaussian corruptio 3. no: no corruption (only forecasting).

## 
