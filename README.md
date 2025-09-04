<div align="center">
<img src="docs/src/assets/logo.png"/>
</div>

# Durbyn.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://taf-society.github.io/Durbyn.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://taf-society.github.io/Durbyn.jl/dev/) [![Build Status](https://github.com/taf-society/Durbyn.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/taf-society/Durbyn.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/taf-society/Durbyn.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/taf-society/Durbyn.jl)

## About

**Durbyn** is a Julia package that implements the functionality of the R **forecast** package, providing tools for time series forecasting. The name "Durbyn" comes from the Kurdish word for binoculars—where *Dur* means "far" and *Byn* means "to see."

This package is currently under development and will be part of the **TAFS Forecasting Ecosystem**, an open-source initiative.

## About TAFS

**TAFS (Time Series Analysis and Forecasting Society)** is a non-profit association registered as a **"Verein"** in Vienna, Austria. The organization connects a global audience of academics, experts, practitioners, and students to engage, share, learn, and innovate in the fields of data science and artificial intelligence, with a particular focus on time-series analysis, forecasting, and decision science. [TAFS](https://taf-society.org/)

TAFS's mission includes:

-   **Connecting**: Hosting events and discussion groups to establish connections and build a community of like-minded individuals.
-   **Learning**: Providing a platform to learn about the latest research, real-world problems, and applications.
-   **Sharing**: Inviting experts, academics, practitioners, and others to present and discuss problems, research, and solutions.
-   **Innovating**: Supporting the transfer of research into solutions and helping to drive innovations.

As a registered non-profit association under Austrian law, TAFS ensures that all contributions remain fully open source and cannot be privatized or commercialized. [TAFS](https://taf-society.org/)

## License

The Durbyn package is licensed under the **MIT License**, allowing for open-source distribution and collaboration.

## Installation

Durbyn is still in development. Once it is officially released, you will be able to install it using Julia’s package manager:

For the latest development version, you can install directly from GitHub:

``` julia
Pkg.add(url="https://github.com/taf-society/Durbyn.jl")
```

## Usage

### Forecasting using Exponential Smoothing.

``` julia

using Durbyn
using Durbyn.ExponentialSmoothing

ap = air_passengers();
fit_ets = ets(ap, 12, "ZZZ")
fc_ets = forecast(fit_ets, h = 12)
plot(fc_ets)


ses_fit = ses(ap, 12)
ses_fc = forecast(ses_fit, h = 12)
plot(ses_fc)


holt_fit = holt(ap, 12)
holt_fc = forecast(holt_fit, h = 12)
plot(holt_fc)


hw_fit = holt_winters(ap, 12)
hw_fc = forecast(hw_fit, h = 12)
plot(hw_fc)
```

### Forecasting Intermittent Demand Data

``` julia
data = [6, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 
0, 0, 0, 0, 0];

# Based on Shenstone, L., and Hyndman, R.J. (2005)
m = 1
fit_crst =croston(data, m)
fc_crst = forecast(fit_crst, 12)
plot(fc_crst)

# this module is based on Kourentzes (2014)
using Durbyn.IntermittentDemand

# Classical Croston Method based Croston, J. (1972) 
crst1 =croston_classic(data)
fc1 = forecast(crst1, h = 12)

residuals(crst1)
residuals(fc1)

fitted(crst1)
fitted(fc1)

plot(fc1, show_fitted = true)

# Croston Method with Syntetos-Boylan Approximation
crst2 =croston_sba(data)
fc2 = forecast(crst2, h = 12)

residuals(crst2)
residuals(fc2)

fitted(crst2)
fitted(fc2)

plot(fc2, show_fitted = true)

# Croston-Shale-Boylan-Johnston Bias Correction Method
crst3 =croston_sbj(data)
fc3 = forecast(crst3, h = 12)

residuals(crst3)
residuals(fc3)

fitted(crst3)
fitted(fc3)

plot(fc3, show_fitted = true)
```

### Forecasting using Arima

``` julia
using Durbyn.Arima
# Fit an arima model
fit = arima(ap, 12, order = PDQ(2,1,1), seasonal=PDQ(0,1,0))

## Generate a forecast
fc = forecast(fit, h = 12)

# Fit an auto arima model
fit = auto_arima(ap, 12, d=1, D=1)

## Generate a forecast
fc = forecast(fit, h = 12)
# Plot the forecast
plot(fc)
```

### Forecasting using Ararma and Arar models

``` julia
# Ararma module

using Durbyn
using Durbyn.Ararma

ap = air_passengers();

# basing arar model
fit = arar(ap, max_ar_depth = 13)
fc = forecast(fit, h = 12)
plot(fc)

# arar model
fit = ararma(ap, p = 0, q = 1)
fc = forecast(fit, h = 12)
plot(fc)

# auto arar model
fit = auto_ararma(ap)
fc = forecast(fit, h = 12)
plot(fc)
```

Durbyn will introduce a **DataFrame-based interface** (tidy forecasting) similar to the **R fable** package, allowing for a more intuitive workflow when working with time series data.