# Corruption-resilient Forecasting Models

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
