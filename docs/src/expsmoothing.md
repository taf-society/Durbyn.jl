# Exponential Smoothing (ETS)

Durbyn provides classical ETS variants and helpers.

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

**Arguments**
- `season` (`12` for monthly, `4` for quarterly, etc.)
- `spec` string selecting error / trend / seasonality (e.g. `"ZZZ"`)
