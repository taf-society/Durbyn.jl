# Quick Start

## Installation

Install the development version:

```julia
using Pkg
Pkg.add(url="https://github.com/taf-society/Durbyn.jl")
```

## Formula Interface (Recommended)

Durbyn provides a modern, declarative interface for model specification with full support for tables, regressors (features in ML terminology), model comparison, and panel data.

### Example 1: Single Time Series

```julia
using Durbyn

# Load data
data = (sales = [120, 135, 148, 152, 141, 158, 170, 165, 180, 195],)

# ARIMA with automatic order selection
spec = ArimaSpec(@formula(sales = p() + q() + P() + Q()))
fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 12)
plot(fc)


# Load data another data
data = (passengers = air_passengers(),)

# ARIMA with automatic order selection
spec = ArimaSpec(@formula(passengers = p() + q() + P() + Q()))
fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 12)
plot(fc)

```

### Example 2: With Regressors (Features)

```julia
using Durbyn
using Random
Random.seed!(123)

# Simulate data 
n = 120
idx = 1:n
temperature = -(15 .+ 8 .* sin.(2π .* idx ./ 12) .+ 0.5 .* randn(n))
marketing = -(0.3 .+ 0.15 .* sin.(2π .* idx ./ 6) .+ 0.05 .* randn(n))
sales = 120 .+ 1.5 .* temperature .+ 30 .* marketing .+ randn(n)
data = (sales = sales, temperature = temperature, marketing = marketing)

spec = ArimaSpec(@formula(sales = temperature + marketing + p() + d() + q() + P() + D() + Q()))
# spec = ArimaSpec(@formula(sales = auto()))
fitted = fit(spec, data, m = 12)

# Simulate future data
n_ahead = 24
future_idx = (n + 1):(n + n_ahead)
future_temp = -(15 .+ 8 .* sin.(2π .* future_idx ./ 12))
future_marketing = -(0.3 .+ 0.15 .* sin.(2π .* future_idx ./ 6))
newdata = (temperature = future_temp, marketing = future_marketing)


fc = forecast(fitted, h = n_ahead, newdata = newdata)

plot(fc)

```

### Example 3: Fitting Multiple Models Together

```julia
# Fit multiple model specifications at once
models = model(
    ArimaSpec(@formula(sales = p() + q())),
    EtsSpec(@formula(sales = e("A") + t("A") + s("A"))),
    ArimaSpec(@formula(sales = p(2) + d(1) + q(2))),
    names = ["auto_arima", "ets_aaa", "arima_212"]
)

# Fit all models
fitted = fit(models, data, m = 12)

# Forecast with all models
fc = forecast(fitted, h = 12)

# Compare results
for (name, model_result) in zip(models.names, fitted.models)
    println("$name: AIC = $(round(model_result.fit.aic, digits=2)), BIC = $(round(model_result.fit.bic, digits=2))")
end
```

### Example 4: Panel Data (Multiple Series)

!!! note "Optional Dependencies"
    This example requires `CSV` and `Downloads`:
    ```julia
    using Pkg
    Pkg.add(["CSV", "Downloads"])
    ```

```julia
using Durbyn, Durbyn.TableOps
using CSV, Downloads, Tables

# Load and reshape data
path = Downloads.download("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/refs/heads/main/Data/retail.csv")
wide = Tables.columntable(CSV.File(path))

long = pivot_longer(wide; id_cols = :date, names_to = :series, values_to = :value)
panel = PanelData(long; groupby = :series, date = :date, m = 12)
glimpse(panel)
# Fit model to all series at once
spec = ArimaSpec(@formula(value = p() + q()))
fitted = fit(spec, panel)

# Forecast all series
fc = forecast(fitted, h = 12)

# Get tidy forecast table
tbl = forecast_table(fc)
glimpse(tbl)
```

### Example 5: ETS Models with Formula

```julia

using Durbyn.Grammar

# Automatic ETS model selection
spec_ets = EtsSpec(@formula(sales = e("Z") + t("Z") + s("Z")))
fitted = fit(spec_ets, data, m = 12)
fc = forecast(fitted, h = 12)
plot(fc)
# Specialized ETS specifications
spec_ses = SesSpec(@formula(sales = ses()))
spec_holt = HoltSpec(@formula(sales = holt(damped=true)))
spec_hw = HoltWintersSpec(@formula(sales = hw(seasonal=:multiplicative)))

# Fit and forecast
fitted_ses = fit(spec_ses, data)
fc_ses = forecast(fitted_ses, h = 12)

plot(fc_ses)

```

---

## Base Models (Array Interface)

The array interface provides direct access to forecasting engines for working with numeric vectors.

### Exponential Smoothing (ETS)

```julia
using Durbyn
using Durbyn.ExponentialSmoothing

ap = air_passengers()

# Automatic ETS model selection
fit_ets = ets(ap, 12, "ZZZ")
fc_ets  = forecast(fit_ets, h = 12)
plot(fc_ets)

# Simple exponential smoothing
ses_fit = ses(ap)
ses_fc  = forecast(ses_fit, h = 12)
plot(ses_fc)

# Holt's linear trend method
holt_fit = holt(ap)
holt_fc  = forecast(holt_fit, h = 12)
plot(holt_fc)

# Holt-Winters seasonal method
hw_fit = holt_winters(ap, 12)
hw_fc  = forecast(hw_fit, h = 12)
plot(hw_fc)
```

### ARIMA

```julia
using Durbyn.Arima

ap = air_passengers()

# Manual ARIMA specification
fit = arima(ap, 12, order = PDQ(2,1,1), seasonal = PDQ(0,1,0))
fc  = forecast(fit, h = 12)
plot(fc)

# Automatic ARIMA selection
fit_auto = auto_arima(ap, 12)
fc_auto  = forecast(fit_auto, h = 12)
plot(fc_auto)
```

---

## Next Steps

!!! tip "Complete End-to-End Example"
    Want to see a comprehensive workflow with train/test split, model comparison, accuracy evaluation, and visualization?
    Check out the **[Complete End-to-End Example in Grammar Guide](grammar.md#complete-end-to-end-example)** — demonstrates fitting 6 models to panel data with full evaluation pipeline.

**Documentation:**
- **[Grammar Guide](grammar.md)** — Complete formula interface documentation for ARIMA and ETS
- **[Table Operations](tableops.md)** — Data wrangling for time series and panel data
- **[ARIMA](arima.md)** — Formula interface and base models (ARIMA, SARIMA, Auto ARIMA)
- **[Exponential Smoothing](expsmoothing.md)** — Formula interface and base models (SES, Holt, Holt-Winters, ETS)
- **[Intermittent Demand](intermittent.md)** — Croston methods for sparse/intermittent data
- **[ARAR/ARARMA](ararma.md)** — Memory-shortening algorithms
