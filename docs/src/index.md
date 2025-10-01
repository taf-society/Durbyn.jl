# Durbyn.jl

![Durbyn.jl logo](assets/logo.png)

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://taf-society.github.io/Durbyn.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://taf-society.github.io/Durbyn.jl/dev/) [![Build Status](https://github.com/taf-society/Durbyn.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/taf-society/Durbyn.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/taf-society/Durbyn.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/taf-society/Durbyn.jl)

**Durbyn** is a Julia package that implements functionality of the R **forecast** package, providing tools for time-series forecasting.

The name Durbyn derives from Kurdish: *Dur* (far) + *Byn* (to see) = binoculars—symbolizing our ability to see far into the future through mathematical precision.

> This site documents the development version. After your first tagged release, see **stable** docs for the latest release.

---

## About TAFS

**TAFS (Time Series Analysis and Forecasting Society)** is a non-profit association (“Verein”) in Vienna, Austria. It connects academics, experts, practitioners, and students focused on time-series, forecasting, and decision science. Contributions remain fully open source.  
Learn more at [taf-society.org](https://taf-society.org/).

---

## Installation

Durbyn is under active development. For the latest dev version:

```julia
using Pkg
Pkg.add(url="https://github.com/taf-society/Durbyn.jl")
```

---

## Quick peek (ETS)

```julia
using Durbyn
using Durbyn.ExponentialSmoothing

ap = air_passengers()

fit_ets = ets(ap, 12, "ZZZ")
fc_ets  = forecast(fit_ets, h = 12)
plot(fc_ets)

ses_fit = ses(ap, 12)
ses_fc  = forecast(ses_fit, h = 12)
plot(ses_fc)

holt_fit = holt(ap, 12)
holt_fc  = forecast(holt_fit, h = 12)
plot(holt_fc)

hw_fit = holt_winters(ap, 12)
hw_fc  = forecast(hw_fit, h = 12)
plot(hw_fc)
```

---

## Intermittent demand (Croston variants)

```julia
data = [6, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 
0, 0, 0, 0, 0];

# Based on Shenstone & Hyndman (2005)
m = 1
fit_crst = croston(data, m)
fc_crst  = forecast(fit_crst, 12)
plot(fc_crst)

using Durbyn.IntermittentDemand

# Classical Croston (Croston, 1972)
crst1 = croston_classic(data)
fc1   = forecast(crst1, h = 12)

residuals(crst1); residuals(fc1);
fitted(crst1);    fitted(fc1);
plot(fc1, show_fitted = true)

# Croston + SBA correction
crst2 = croston_sba(data)
fc2   = forecast(crst2, h = 12)
plot(fc2, show_fitted = true)

# Croston + SBJ correction
crst3 = croston_sbj(data)
fc3   = forecast(crst3, h = 12)
plot(fc3, show_fitted = true)
```

---

## ARIMA

```julia
using Durbyn.Arima

ap  = air_passengers()

# manual ARIMA
fit = arima(ap, 12, order = PDQ(2,1,1), seasonal = PDQ(0,1,0))
fc  = forecast(fit, h = 12)

# auto ARIMA
fit2 = auto_arima(ap, 12, d = 1, D = 1)
fc2  = forecast(fit2, h = 12)
plot(fc2)
```

---

## ARAR / ARARMA

```julia
using Durbyn
using Durbyn.Ararma

ap = air_passengers()

fit  = arar(ap, max_ar_depth = 13)
fc   = forecast(fit, h = 12)
plot(fc)

fit2 = ararma(ap, p = 0, q = 1)
fc2  = forecast(fit2, h = 12)
plot(fc2)

fit3 = auto_ararma(ap)
fc3  = forecast(fit3, h = 12)
plot(fc3)
```

---

## License

MIT License.

---

## What’s next

- Read the **Quick Start** (left sidebar).
- Explore **User Guide** pages (ETS, Intermittent Demand, ARIMA, ARAR/ARARMA).
- See the **API Reference** for full docs.
