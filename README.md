<div align="center">
<img src="docs/src/assets/logo.png"/>
</div>

# Durbyn.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://taf-society.github.io/Durbyn.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://taf-society.github.io/Durbyn.jl/dev/) [![Build Status](https://github.com/taf-society/Durbyn.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/taf-society/Durbyn.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/taf-society/Durbyn.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/taf-society/Durbyn.jl)

## About

**Durbyn** is a Julia package that implements functionality of the R **forecast** package, providing tools for time-series forecasting.

Durbyn — Kurdish for “binoculars” (Dur, far + Byn, to see), embodies foresight through science. Like Hari Seldon’s psychohistory in Asimov’s Foundation, we seek to glimpse the shape of tomorrow through the disciplined clarity of mathematics.

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

Durbyn is still in development. Once it is officially released, you will be able to install it using Julia's package manager.

For the latest development version, you can install directly from GitHub:

``` julia
Pkg.add(url="https://github.com/taf-society/Durbyn.jl")
```

**Performance Tip:** Durbyn supports automatic parallel computing when fitting models to panel data. Start Julia with multiple threads for massive speedups:
```bash
julia -t auto  # Use all available CPU cores
```
Performance scales with cores (8 cores: ~8x faster, 32 cores: ~20x faster, 96+ cores: even greater speedups).
See the [Quick Start Performance Guide](https://taf-society.github.io/Durbyn.jl/dev/quickstart/#Performance-Multi-Threading-for-Parallel-Computing) for details.

## Usage

Durbyn ships with multiple forecasting engines and a unified **formula interface**
for declarative model specification. The formula interface is the **recommended approach** for most users, providing a modern, expressive way to specify models with full support for tables, panel data, and model comparison.

For complete documentation, see the [Grammar Guide](https://taf-society.github.io/Durbyn.jl/dev/grammar/) in the docs.

### Key Components

- **Formula Interface** — Declarative model specification using `@formula` for both
  ARIMA and ETS models (`ArimaSpec`, `EtsSpec`, `SesSpec`, `HoltSpec`, etc.)
- `Durbyn.ModelSpecs` — Formula-based model specifications with `PanelData` support,
  grouped fitting, and `forecast_table` for tidy outputs
- `Durbyn.TableOps` — Lightweight, Tables.jl-friendly data wrangling helpers
  (`pivot_longer`, `arrange`, `groupby`, `summarise`, `mutate`, …) plus `glimpse` utilities, see the [Table Operations](https://taf-society.github.io/Durbyn.jl/dev/tableops/) in the docs,
- `Durbyn.ExponentialSmoothing`, `Durbyn.Arima`, `Durbyn.Ararma`,
  `Durbyn.IntermittentDemand` — Base model engines for array interface

---

## Formula Interface (Primary Usage)

The formula interface provides a declarative, flexible way to specify forecasting models
with support for single series, regressors, model comparison, and panel data.

### Example 1: Single ARIMA Model

```julia
using Durbyn

# Load data
data = (sales = [120, 135, 148, 152, 141, 158, 170, 165, 180, 195],)

# Specify model with automatic order selection
spec = ArimaSpec(@formula(sales = p() + q() + d() + P() + D() + Q()))
fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 10)
plot(fc)
```

### Example 2: ARIMA with Regressors

```julia
# Model with exogenous regressors
data = (
    sales = rand(100),
    temperature = rand(100),
    promotion = rand(0:1, 100)
)

spec = ArimaSpec(@formula(sales = p(1,3) + q(1,3) + temperature + promotion))
fitted = fit(spec, data, m = 7)

# Forecast with future regressor values
newdata = (temperature = rand(7), promotion = rand(0:1, 7))
fc = forecast(fitted, h = 7, newdata = newdata)
```

### Example 3: Fitting Multiple Models Together

```julia

# Fit multiple model specifications at once
models = model(
    ArimaSpec(@formula(value = p() + q() + P() + Q() + d() + D())),
    EtsSpec(@formula(value = e("Z") + t("Z") + s("Z") + drift(:auto))),
    SesSpec(@formula(value = ses())),
    HoltSpec(@formula(value = holt(damped=true))),
    HoltWintersSpec(@formula(value = hw(seasonal="multiplicative")); m = 12),
    CrostonSpec(@formula(value = croston())),
    names = ["arima", "ets_auto", "ses", "holt_damped", "hw_mul", "croston"]
)

fitted = fit(models, panel)       # each spec fitted to every series
fc     = forecast(fitted, h = 12) # ForecastModelCollection

fc_tbl = forecast_table(fc) # stacked tidy table with model_name column

```

### Example 4: Panel Data (Models and Multiple Time Series)

```julia
using Durbyn
using Durbyn.ModelSpecs
using Durbyn.Grammar
using Downloads
using Tables
using CSV

# Download and load data
path = Downloads.download("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/refs/heads/main/Data/retail.csv")
tbl = Tables.columntable(CSV.File(path))

# Reshape to long format
tbl = pivot_longer(tbl; id_cols=:date, names_to=:series, values_to=:value)

glimpse(tbl)

# Split into train and test sets using table operations
# Get unique dates to determine split point
all_dates = unique(tbl.date)
n_dates = length(all_dates)
split_date = all_dates[end-11]  # Hold out last 12 periods for testing

# Create train and test sets
train = query(tbl, row -> row.date <= split_date)
test = query(tbl, row -> row.date > split_date)

println("Training data:")
glimpse(train)
println("\nTest data:")
glimpse(test)

# Create panel data wrapper for training
panel = PanelData(train; groupby=:series, date=:date, m=12);

glimpse(panel)

# Define multiple models for comparison
models = model(
    ArimaSpec(@formula(value = p() + q())),
    EtsSpec(@formula(value = e("Z") + t("Z") + s("Z") + drift(:auto))),
    SesSpec(@formula(value = ses())),
    HoltSpec(@formula(value = holt(damped=true))),
    HoltWintersSpec(@formula(value = hw(seasonal="multiplicative")); m=12),
    CrostonSpec(@formula(value = croston())),
    names=["arima", "ets_auto", "ses", "holt_damped", "hw_mul", "croston"]
)

# Fit all models to all series
fitted = fit(models, panel)

# Generate forecasts (h=12 to match test set)
fc = forecast(fitted, h=12)

# Convert to tidy table format
fc_tbl = forecast_table(fc)

glimpse(fc_tbl)

# Optional: Save forecasts to CSV
# CSV.write("forecasts.csv", fc_tbl)

# Calculate accuracy metrics
# Method 1: Using ForecastModelCollection directly
acc_results = accuracy(fc, test)

println("\nAccuracy by Series and Model:")
glimpse(acc_results)

list_series(fc)  # See what's available
plot(fc)  # Quick look at first series
plot(fc, series=:all, facet=true, n_cols=4)  # Overview

# Detailed inspection
plot(fc, series="series_1", actual=test)

# Calculate accuracy
acc = accuracy(fc, test)

# Find and plot interesting cases
best = acc.series[argmin(acc.MAPE)]
worst = acc.series[argmax(acc.MAPE)]

plot(fc, series=[best, worst], facet=true, actual=test)

```

### Example 5: ETS Models with Formula

```julia
# Automatic ETS model selection
spec_ets = EtsSpec(@formula(sales = e("Z") + t("Z") + s("Z")))
fitted = fit(spec_ets, data, m = 12)
fc = forecast(fitted, h = 12)

# Specific ETS components
spec_ses = SesSpec(@formula(sales = e("A")))
spec_holt = HoltSpec(@formula(sales = e("A") + t("A")), damped = true)
spec_hw = HoltWintersSpec(@formula(sales = e("A") + t("A") + s("M")))
```

---

## Base Models (Array Interface)

The array interface provides direct access to forecasting engines for working with numeric vectors. This is useful for quick analyses or when integrating with existing code.

### Exponential Smoothing

``` julia

using Durbyn
using Durbyn.ExponentialSmoothing

ap = air_passengers();
fit_ets = ets(ap, 12, "ZZZ")
fc_ets = forecast(fit_ets, h = 12)
plot(fc_ets)


ses_fit = ses(ap)
ses_fc = forecast(ses_fit, h = 12)
plot(ses_fc)


holt_fit = holt(ap)
holt_fc = forecast(holt_fit, h = 12)
plot(holt_fc)


hw_fit = holt_winters(ap, 12)
hw_fc = forecast(hw_fit, h = 12)
plot(hw_fc)
```

### Intermittent Demand

``` julia
data = [6, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 
0, 0, 0, 0, 0];

# Based on Shenstone, L., and Hyndman, R.J. (2005)

fit_crst =croston(data)
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

### Classical ARIMA API

``` julia
using Durbyn
using Durbyn.Arima

ap = air_passengers()
# Fit an arima model
arima_model = arima(ap, 12, order = PDQ(2,1,1), seasonal=PDQ(0,1,0))

## Generate a forecast
fc = forecast(arima_model, h = 12)
# Plot the forecast
plot(fc)

# Fit an auto arima model
auto_arima_model = auto_arima(ap, 12)

## Generate a forecast
fc = forecast(auto_arima_model, h = 12)
# Plot the forecast
plot(fc)
```

### ARAR and ARARMA models

``` julia
# Ararma module

using Durbyn
using Durbyn.Ararma

ap = air_passengers();

# basing arar model
arar_model_basic = arar(ap, max_ar_depth = 13)
fc = forecast(arar_model_basic, h = 12)
plot(fc)

# arar model
ararma_model = ararma(ap, p = 0, q = 1)
fc = forecast(ararma_model, h = 12)
plot(fc)

# auto arar model
auto_ararma_model = auto_ararma(ap)
fc = forecast(auto_ararma_model, h = 12)
plot(fc)
```

You can also fit ARAR declaratively using Durbyn's forecasting grammar:

```julia
using Durbyn
using Durbyn.Ararma

series = air_passengers()
data = (value = series,)
formula = @formula(value = arar(max_ar_depth=20, max_lag=30))

arar_model = arar(formula, data)
fc = forecast(arar_model; h = 12)
plot(fc)

spec = ArarSpec(@formula(value = arar()))
panel_tbl = (
    value = vcat(series, series .* 1.05),
    region = vcat(fill("north", length(series)), fill("south", length(series)))
)
panel = PanelData(panel_tbl; groupby = :region)
group_fit = fit(spec, panel)
group_fc = forecast(group_fit; h = 6)
plot(group_fc)

```

`ArarSpec` slots into `model(...)` collections alongside ARIMA/ETS specs, so benchmarking ARAR against other models is now a one-line change.
