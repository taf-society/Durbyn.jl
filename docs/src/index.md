# Durbyn.jl

![Durbyn.jl logo](assets/logo.png)

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://taf-society.github.io/Durbyn.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://taf-society.github.io/Durbyn.jl/dev/) [![Build Status](https://github.com/taf-society/Durbyn.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/taf-society/Durbyn.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/taf-society/Durbyn.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/taf-society/Durbyn.jl)

**Durbyn** is a Julia package for time-series forecasting, inspired by the R **forecast** and **fable** packages. While drawing on their methodology, Durbyn is a native Julia implementation featuring its own unique formula-based grammar for declarative model specification.

Durbyn — Kurdish for “binoculars” (Dur, far + Byn, to see), embodies foresight through science. Like Hari Seldon’s psychohistory in Asimov’s Foundation, we seek to glimpse the shape of tomorrow through the disciplined clarity of mathematics.

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

!!! tip "Performance: Multi-Threading"
    Durbyn automatically uses parallel computing when fitting models to panel data. Start Julia with multiple threads for **massive speedups** that scale with your CPU cores:
    ```bash
    julia -t auto
    ```
    See [Performance Guide](quickstart.md#Performance-Multi-Threading-for-Parallel-Computing) for all setup methods including VS Code configuration.

---

## Formula Interface (Recommended)

Durbyn provides a modern, declarative interface for model specification using `@formula`.
This is the **recommended approach** for most users, supporting single series, model comparison, and panel data forecasting.

The `PanelData` interface follows the **[tidy forecasting workflow](https://otexts.com/fpp3/a-tidy-forecasting-workflow.html)** (Hyndman & Athanasopoulos, 2021), providing a structured approach:

1. **Data Preparation** — Load, reshape, and clean data using `TableOps`
2. **Visualization** — Explore patterns with `plot()` and `glimpse()`
3. **Model Specification** — Define models using the formula interface (`@formula`)
4. **Model Training** — Fit models with `fit()`, producing fitted model objects
5. **Performance Evaluation** — Assess accuracy with `accuracy()` and diagnostics
6. **Forecasting** — Generate predictions with `forecast()`, returning tidy forecast tables

!!! note "Optional Dependencies"
    Panel data examples require `CSV` and `Downloads` packages:
    ```julia
    using Pkg
    Pkg.add(["CSV", "Downloads"])
    ```

### Complete Workflow: Model Comparison with Panel Data

```julia
using Durbyn, Durbyn.TableOps, Durbyn.Grammar
using CSV, Downloads, Tables

# 1. Load and prepare data
path = Downloads.download("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/refs/heads/main/Data/retail.csv")
wide = Tables.columntable(CSV.File(path))

# Reshape to long format
tbl = pivot_longer(wide; id_cols=:date, names_to=:series, values_to=:value)
glimpse(tbl)

# 2. Split into train and test sets
all_dates = unique(tbl.date)
split_date = all_dates[end-11]  # Hold out last 12 periods for testing

train = query(tbl, row -> row.date <= split_date)
test = query(tbl, row -> row.date > split_date)

println("Training data:")
glimpse(train)
println("\nTest data:")
glimpse(test)

# 3. Create panel data wrapper
panel = PanelData(train; groupby=:series, date=:date, m=12)
glimpse(panel)

# 4. Define multiple models for comparison
models = model(
    ArarSpec(@formula(value = arar())),                                  # ARAR
    BatsSpec(@formula(value = bats(seasonal_periods=12))),               # BATS with seasonality
    ArimaSpec(@formula(value = p() + q())),                              # Auto ARIMA
    EtsSpec(@formula(value = e("Z") + t("Z") + s("Z") + drift(:auto))),  # Auto ETS with drift
    SesSpec(@formula(value = ses())),                                    # Simple exponential smoothing
    HoltSpec(@formula(value = holt(damped=true))),                       # Damped Holt
    HoltWintersSpec(@formula(value = hw(seasonal=:multiplicative))),     # Holt-Winters multiplicative
    CrostonSpec(@formula(value = croston())),                            # Croston's method
    names=["arar", "bats", "arima", "ets_auto", "ses", "holt_damped", "hw_mul", "croston"]
)

# 5. Fit all models to all series
fitted = fit(models, panel)

# 6. Generate forecasts (h=12 to match test set)
fc = forecast(fitted, h=12)

# 7. Convert to tidy table format
fc_tbl = as_table(fc)
glimpse(fc_tbl)

# 8. Calculate accuracy metrics across all models and series
acc_results = accuracy(fc, test)
println("\nAccuracy by Series and Model:")
glimpse(acc_results)

# 9. Visualization
list_series(fc)  # Show available series

# Quick overview of all series for first model
plot(fc, series=:all, facet=true, n_cols=4)

# Detailed inspection with actual values from test set
plot(fc, series="series_10", actual=test)

# 10. Find best and worst performing series
best_series = acc_results.series[argmin(acc_results.MAPE)]
worst_series = acc_results.series[argmax(acc_results.MAPE)]

# Compare best vs worst performers
plot(fc, series=[best_series, worst_series], facet=true, actual=test)
```

**This example demonstrates:**
- **Data wrangling**: Load, reshape, and split data using TableOps
- **Model comparison**: Fit 8 forecasting methods (ARAR, BATS, ARIMA, ETS variants, Croston)
- **Panel forecasting**: Automatic iteration over multiple time series
- **Out-of-sample evaluation**: Train/test split with accuracy metrics
- **Visualization**: Faceted plots, actual vs forecast comparison
- **Tidy output**: Structured tables ready for further analysis

### Quick Examples

#### Single Series ARIMA

```julia
using Durbyn

data = (sales = [120, 135, 148, 152, 141, 158, 170, 165, 180, 195],)

spec = ArimaSpec(@formula(sales = p() + q()))
fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 12)
```

#### ARIMA with Regressors (Features)

```julia
data = (
    sales = rand(100),
    temperature = rand(100),
    promotion = rand(0:1, 100)
)

spec = ArimaSpec(@formula(sales = p(1,3) + q(1,3) + temperature + promotion))
fitted = fit(spec, data, m = 7)

# Provide future values of regressors
newdata = (temperature = rand(7), promotion = rand(0:1, 7))
fc = forecast(fitted, h = 7, newdata = newdata)
```

#### Automatic ETS Selection

```julia
spec_ets = EtsSpec(@formula(sales = e("Z") + t("Z") + s("Z")))
fitted = fit(spec_ets, data, m = 12)
fc = forecast(fitted, h = 12)
```

---

## Base Models (Array Interface)

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
arima_model = arima(ap, 12, order = PDQ(2,1,1), seasonal = PDQ(0,1,0))
fc  = forecast(arima_model, h = 12)

# auto ARIMA
auto_arima_model = auto_arima(ap, 12, d = 1, D = 1)
fc2  = forecast(auto_arima_model, h = 12)
plot(fc2)
```

---

## ARAR

```julia
using Durbyn
using Durbyn.Ararma

ap = air_passengers()

arar_model = arar(ap, max_ar_depth = 13)
fc = forecast(arar_model, h = 12)
plot(fc)
```

---

## ARARMA

```julia
using Durbyn
using Durbyn.Ararma

ap = air_passengers()

ararma_model = ararma(ap, p = 0, q = 1)
fc = forecast(ararma_model, h = 12)
plot(fc)

auto_ararma_model = auto_ararma(ap)
fc2 = forecast(auto_ararma_model, h = 12)
plot(fc2)
```

---

## BATS / TBATS

BATS (Box-Cox, ARMA errors, Trend, Seasonal) and TBATS (Trigonometric BATS) handle complex seasonal patterns with multiple seasonal periods.

### Formula Interface (Recommended)

```julia
using Durbyn
using Durbyn.ModelSpecs

# BATS with monthly seasonality
data = (sales = randn(120) .+ 10,)
spec = BatsSpec(@formula(sales = bats(seasonal_periods=12)))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)
plot(fc)
# BATS with multiple seasonal periods (hourly data with daily and weekly patterns)
spec_multi = BatsSpec(@formula(sales = bats(seasonal_periods=[24, 168])))
fitted_multi = fit(spec_multi, data)
fc_multi = forecast(fitted_multi, h = 24)

# BATS with specific components
spec_custom = BatsSpec(@formula(sales = bats(
    seasonal_periods=12,
    use_box_cox=true,
    use_trend=true,
    use_damped_trend=true,
    use_arma_errors=true
)))
fitted_custom = fit(spec_custom, data)
fc_custom = forecast(fitted_custom, h = 12)
```

### Base API

```julia
using Durbyn.Bats

y = randn(120)

# BATS with automatic component selection
bats_model = bats(y, 12)
fc = forecast(bats_model, h = 12)
plot(fc)

# BATS with multiple seasonal periods
bats_multi = bats(y, [24, 168]; use_box_cox=true, use_arma_errors=true)
fc_multi = forecast(bats_multi, h = 24)

# TBATS for non-integer seasonality
tbats_model = tbats(y, [12.5, 52.18])  # Non-integer periods
fc_tbats = forecast(tbats_model, h = 12)
```

---

## License

MIT License.

---

## What's next

- **[Grammar Guide](grammar.md)** (Recommended) — Learn the complete formula interface for ARIMA, BATS, and ETS
- **[Quick Start](quickstart.md)** — Get started quickly with formula and base models
- **User Guide** pages:
  - [Table Operations](tableops.md) — Data wrangling with Tables.jl for panel data
  - [ARIMA](arima.md) — Formula interface and base models (ARIMA, SARIMA, Auto ARIMA)
  - [Exponential Smoothing](expsmoothing.md) — Formula interface and base models (SES, Holt, Holt-Winters, ETS)
  - [BATS](bats.md) — Box-Cox, ARMA errors, Trend, Seasonal models for complex seasonality
  - [TBATS](tbats.md) — Trigonometric BATS for non-integer and very long seasonal periods
  - [Intermittent Demand](intermittent.md) — Croston methods
  - [ARAR](arar.md) — Memory-shortening AR model (Brockwell & Davis)
  - [ARARMA](ararma.md) — Memory-shortening with ARMA fitting (Parzen)
  - [Statistics](stats.md) — Box-Cox transformations, decomposition (STL, MSTL), unit root tests, ACF/PACF
  - [Optimization](optimize.md) — Nelder-Mead, BFGS, L-BFGS-B, and Brent optimization algorithms
  - [Utilities](utils.md) — Example datasets, data manipulation helpers, and other utilities
- **API Reference** — Complete API documentation
