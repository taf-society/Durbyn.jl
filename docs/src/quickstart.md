# Quick Start

Install (dev version):

```julia
using Pkg
Pkg.add(url="https://github.com/taf-society/Durbyn.jl")
```

Basic forecasting with Exponential Smoothing (ETS):

```julia
using Durbyn
using Durbyn.ExponentialSmoothing

ap = air_passengers()
fit_ets = ets(ap, 12, "ZZZ")
fc_ets  = forecast(fit_ets, h = 12)
```

Plot (example with Plots.jl):

```julia
using Plots
plot(fc_ets)
```

Next: Explore [Intermittent Demand](intermittent.md), [ARIMA](arima.md), and [ARAR/ARARMA](ararma.md).
