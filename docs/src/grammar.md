# Durbyn Grammar

Durbyn provides an expressive, composable grammar for defining forecasting models. This unified interface lets you describe ARIMA, SARIMA, and exponential smoothing models with concise, readable syntax using the `@formula` macro and specialized model specifications.

Future releases will extend this grammar to support additional statistical models (state space models, structural time series, etc.) and machine learning forecasting methods, all accessible through the same consistent interface.

---

## Overview

The Durbyn grammar system consists of:

- **Formula interface**: Use `@formula` to declaratively specify model components
- **Model specifications**: Wrap formulas in specs like `ArimaSpec`, `EtsSpec`, `SesSpec`, etc.
- **Unified fitting**: Call `fit(spec, data)` with optional grouping for panel data
- **Consistent forecasting**: Use `forecast(fitted, h)` for both single and grouped models; external variables can be passed if the model supports them

This design eliminates manual tuning loops and provides a consistent interface across all model families.

---

## ARIMA Grammar

The ARIMA grammar lets you describe ARIMA and SARIMA models with flexible order specifications and exogenous variable support.

### Formula Basics

Define the relationship between a response variable (target in ML terminology) and its ARIMA structure:

```julia
@formula(sales = p() + d() + q())
```

Every formula requires a **response variable** (left-hand side; called *target* in ML) and one or more model components (right-hand side). Components may specify ARIMA orders, seasonal orders, or **regressors** (exogenous variables; called *features* in ML).

### Non-Seasonal Orders

| Function | Meaning                         | Default or form                             |
|----------|---------------------------------|---------------------------------------------|
| `p()`    | Non-seasonal AR order           | Search range 2–5                            |
| `p(k)`   | Fix AR order                    | Uses `k` exactly                            |
| `p(min,max)` | Search AR order range       | Searches `min` through `max`                |
| `d()`    | Differencing order (auto)       | `auto_arima` chooses                        |
| `d(k)`   | Fix differencing order          | Uses `k` exactly                            |
| `q()`    | Non-seasonal MA order           | Search range 2–5                            |
| `q(k)`   | Fix MA order                    | Uses `k` exactly                            |
| `q(min,max)` | Search MA order range       | Searches `min` through `max`                |

Any range `(min,max)` triggers full `auto_arima` search. If all orders are fixed, the formula interface automatically calls the faster `arima` routine.

### Seasonal Orders

Seasonal counterparts include `P`, `D`, and `Q`:

```julia
@formula(sales = p() + d() + q() + P() + Q())
```

| Function | Meaning                             | Default or form                              |
|----------|-------------------------------------|----------------------------------------------|
| `P()`    | Seasonal AR order                   | Search range 1–2                              |
| `P(k)`   | Fix seasonal AR order               | Uses `k` exactly                              |
| `P(min,max)` | Search seasonal AR order range | Searches `min` through `max`                  |
| `D()`    | Seasonal differencing (auto)        | `auto_arima` chooses                          |
| `D(k)`   | Fix seasonal differencing order     | Uses `k` exactly                              |
| `Q()`    | Seasonal MA order                   | Search range 1–2                              |
| `Q(k)`   | Fix seasonal MA order               | Uses `k` exactly                              |
| `Q(min,max)` | Search seasonal MA order range | Searches `min` through `max`                  |

Remember to provide the seasonal period `m` when fitting: `fit(spec, data, m=12)`.

### Exogenous Regressors

#### Explicit Variables

Add regressors (features) by listing column names:

```julia
@formula(sales = p() + q() + price + promotion)
```

These become `VarTerm`s—during fitting, Durbyn pulls the matching columns from your data.

#### Automatic Selection (`auto()`)

Use `auto()` to include all numeric columns as regressors, excluding the response variable (target), group columns, and optional date column:

```julia
@formula(sales = auto())                    # pure auto ARIMA + automatic xregs
@formula(sales = p() + q() + auto())        # combine with explicit ARIMA orders
```

Automatic selection is mutually exclusive with explicit exogenous variables or `xreg_formula`.

#### Complex Designs (`xreg_formula`)

For interactions or transformations, supply a secondary formula when constructing `ArimaSpec`:

```julia
spec = ArimaSpec(
    @formula(sales = p() + q()),
    xreg_formula = Formula("~ temperature * promotion + price^2")
)
```

The `xreg_formula` is evaluated via `Utils.model_matrix`, producing the necessary design matrix before fitting.

### ARIMA Examples

**Fixed orders (fast estimation)**:
```julia
spec = ArimaSpec(@formula(sales = p(1) + d(1) + q(1)))
fitted = fit(spec, (sales = y,))
```

**Auto ARIMA with search ranges**:
```julia
spec = ArimaSpec(@formula(sales = p(0,3) + d() + q(0,3)))
fitted = fit(spec, (sales = y,))
```

**Seasonal model with exogenous variables**:
```julia
spec = ArimaSpec(@formula(sales = p() + d() + q() + P() + Q() + price + promotion), m = 12)
fitted = fit(spec, data; m = 12)
```

**Panel data with automatic xreg**:
```julia
spec = ArimaSpec(@formula(value = p() + d() + q() + P() + Q() + auto()))
panel = PanelData(tbl; groupby = :store, date = :date, m = 12)
fitted = fit(spec, panel)
fc = forecast(fitted, h = 12)
```

---

## ETS Grammar

The ETS grammar mirrors the ARIMA DSL, letting you describe exponential smoothing models with expressive, composable terms.

### Formula Basics

Use `@formula` to define the response variable (target) and its ETS components:

```julia
@formula(sales = e("A") + t("N") + s("N"))
```

Each term is created with helper functions (`e`, `t`, `s`, `drift`). The resulting formula feeds into `EtsSpec`.

### Component Functions

| Function | Meaning                  | Accepted Codes                     |
|----------|-------------------------|------------------------------------|
| `e()`    | Error component         | `"A"` additive, `"M"` multiplicative, `"Z"` auto |
| `t()`    | Trend component         | `"N"` none, `"A"` additive, `"M"` multiplicative, `"Z"` auto |
| `s()`    | Seasonal component      | `"N"` none, `"A"` additive, `"M"` multiplicative, `"Z"` auto |

Examples:

```julia
e("A")              # Additive errors
t("M")              # Multiplicative trend
s("Z")              # Auto-select seasonal type
```

Any component you omit defaults to `"Z"` (automatic selection). Combine the components as needed for your model structure.

### Damping and Drift

Use `drift()` to control trend damping:

| Call                 | Effect                                         |
|----------------------|------------------------------------------------|
| `drift()`            | Force a damped trend (`damped = true`)         |
| `drift(false)`       | Forbid damping (`damped = false`)              |
| `drift(:auto)`       | Let ETS decide (`damped = nothing`)            |
| `drift("auto")`      | Same as `drift(:auto)`                         |

You can combine `drift` with any trend choice. When omitted, the ETS search decides whether to include damping.

### Creating `EtsSpec`

Construct the specification with your formula and optional keywords (passed through to `ets`):

```julia
spec = EtsSpec(
    @formula(sales = e("Z") + t("A") + s("A") + drift()),
    m = 12,           # seasonal period
    ic = "aicc"       # information criterion for model selection
)

fitted = fit(spec, (sales = sales_vec,); m = 12)
fc = forecast(fitted, h = 12)
```

You can override spec options at fit time—keywords supplied to `fit` take precedence over those stored in the specification.

### ETS Quick Recipes

**Simple Exponential Smoothing (SES)**:
```julia
spec = EtsSpec(@formula(value = e("A") + t("N") + s("N")))
fitted = fit(spec, (value = y,))
```

**Holt's Linear Trend**:
```julia
spec = EtsSpec(@formula(value = e("A") + t("A") + s("N") + drift(false)))
fitted = fit(spec, (value = y,))
```

**Holt-Winters (Additive), monthly seasonality**:
```julia
spec = EtsSpec(@formula(value = e("A") + t("A") + s("A") + drift(:auto)), m = 12)
fitted = fit(spec, (value = y,), m = 12)
```

**Auto ETS with grouped data**:
```julia
spec = EtsSpec(@formula(value = e("Z") + t("Z") + s("Z")))
fitted = fit(spec, table; groupby = :store, m = 12)
fc = forecast(fitted, h = 8)
```

### Specialized ETS Shortcuts

You can also target specialized exponential smoothing families directly:

```julia
# Simple Exponential Smoothing
ses_spec = SesSpec(@formula(value = ses()))

# Holt's linear trend (damped trend forced on)
holt_spec = HoltSpec(@formula(value = holt(damped=true)))

# Holt-Winters with multiplicative seasonality
hw_spec = HoltWintersSpec(@formula(value = hw(seasonal="multiplicative")), m = 12)

# Croston's intermittent-demand method
croston_spec = CrostonSpec(@formula(demand = croston()))
```

These specs share the same grouped/PanelData support as `EtsSpec`, and all options passed via the specification or `fit` keywords are forwarded to the underlying implementations.

---

## Multi-Model Fitting 

Use `ModelCollection` to fit multiple specifications simultaneously:

```julia
using Durbyn
using Durbyn.ModelSpecs
using Durbyn.Grammar

# Long table with :series / :date / :value columns
panel = PanelData(tbl; groupby = :series, date = :date, m = 12)

models = model(
    ArimaSpec(@formula(value = p() + q())),
    EtsSpec(@formula(value = e("Z") + t("Z") + s("Z") + drift(:auto))),
    SesSpec(@formula(value = ses())),
    HoltSpec(@formula(value = holt(damped=true))),
    HoltWintersSpec(@formula(value = hw(seasonal="multiplicative")); m = 12),
    CrostonSpec(@formula(value = croston())),
    names = ["arima", "ets_auto", "ses", "holt_damped", "hw_mul", "croston"]
)

fitted = fit(models, panel)       # each spec fitted to every series
fc     = forecast(fitted, h = 12) # ForecastModelCollection

forecast_table(fc)                # stacked tidy table with model_name column
```

`forecast_table` stacks every model (and group) with a `model_name` column, so downstream comparisons stay tidy. You can filter to a specific model or pivot wider using `Durbyn.TableOps` functions, or use other Julia packages like `DataFrames.jl`, `DataFramesMeta.jl`, or `Query.jl`.

---

## Complete End-to-End Example

Here's a comprehensive workflow demonstrating model comparison, forecasting, and accuracy evaluation with panel data:

!!! note "Optional Dependencies"
    This example requires `CSV` and `Downloads` packages:
    ```julia
    using Pkg
    Pkg.add(["CSV", "Downloads"])
    ```

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
    ArarSpec(@formula(value = arar())),                                # ARAR via grammar
    ArimaSpec(@formula(value = p() + q())),                              # Auto ARIMA
    EtsSpec(@formula(value = e("Z") + t("Z") + s("Z") + drift(:auto))),  # Auto ETS with drift
    SesSpec(@formula(value = ses())),                                    # Simple exponential smoothing
    HoltSpec(@formula(value = holt(damped=true))),                       # Damped Holt
    HoltWintersSpec(@formula(value = hw(seasonal=:multiplicative))),     # Holt-Winters multiplicative
    CrostonSpec(@formula(value = croston())),                            # Croston's method
    names=["arar", "arima", "ets_auto", "ses", "holt_damped", "hw_mul", "croston"]
)

# 5. Fit all models to all series
fitted = fit(models, panel)

# 6. Generate forecasts (h=12 to match test set)
fc = forecast(fitted, h=12)

# 7. Convert to tidy table format
fc_tbl = forecast_table(fc)
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
# Filter accuracy results for a specific metric (e.g., MAPE)
best_series = acc_results.series[argmin(acc_results.MAPE)]
worst_series = acc_results.series[argmax(acc_results.MAPE)]

# Compare best vs worst performers
plot(fc, series=[best_series, worst_series], facet=true, actual=test)
```

**Key Features Demonstrated:**
- **Data preparation**: Download, reshape, and split data using TableOps
- **Model comparison**: Fit 7 different forecasting methods simultaneously (ARAR + classical methods)
- **Panel forecasting**: Automatic iteration over multiple time series
- **Train/test split**: Proper out-of-sample evaluation
- **Accuracy metrics**: Compare model performance across series
- **Visualization**: Multiple plotting options for analysis
- **Tidy output**: Structured forecast tables ready for downstream analysis

---

## ARAR Grammar

The ARAR grammar exposes the `arar()` term so you can configure the adaptive-reduction model with the same declarative workflow as ARIMA and ETS.

### Formula term

```julia
@formula(value = arar())                           # use defaults
@formula(value = arar(max_ar_depth=20))            # custom depth
@formula(value = arar(max_ar_depth=20, max_lag=40))
```

Both keywords are optional; if omitted, Durbyn derives appropriate values from the series length. Validation happens at macro-expansion time so mistakes are caught immediately.

### Direct formula fitting

```julia
using Durbyn
using Durbyn.Ararma

data = (value = air_passengers(),)
formula = @formula(value = arar(max_lag=30))
arar_model = arar(formula, data)          # tables.jl compatible data
fc  = forecast(arar_model; h = 12)
```

The estimator lives in the `Durbyn.Ararma` submodule, so call `arar(formula, data)` from there (either via `using Durbyn.Ararma` or `Durbyn.Ararma.arar(...)`). It works with any Tables.jl source and returns the familiar `ARAR` struct.

### Model specification (`ArarSpec`)

To leverage grouped fitting, forecasting, and model collections, wrap the formula in `ArarSpec`:

```julia
spec = ArarSpec(@formula(value = arar(max_ar_depth=15)))
fitted = fit(spec, data)
fc = forecast(fitted; h = 8)
```

For panel data:

```julia
panel = PanelData(tbl; groupby = :region)
group_fit = fit(spec, panel)
group_fc = forecast(group_fit; h = 6)
```

And to compare against other specs:

```julia
models = model(
    ArarSpec(@formula(value = arar())),
    ArimaSpec(@formula(value = p() + q())),
    EtsSpec(@formula(value = e("Z") + t("Z") + s("Z"))),
    names = ["arar", "arima", "ets"]
)

fitted = fit(models, panel)
fc = forecast(fitted; h = 12)
```

The ARAR grammar therefore integrates seamlessly with every Durbyn workflow—single series, grouped/panel data, and large-scale model comparisons.

---

## ARARMA Grammar

The ARARMA grammar extends the ARAR approach by fitting a short-memory ARMA(p,q) model after the adaptive reduction stage. Like ARIMA, it uses the `p()` and `q()` terms to specify model orders, but the distinction comes from using `ArarmaSpec` instead of `ArimaSpec`.

### Formula terms

ARARMA reuses ARIMA's order grammar:

```julia
@formula(value = p() + q())                    # auto selection with defaults
@formula(value = p(1) + q(2))                  # fixed ARARMA(1,2)
@formula(value = p(0,3) + q(0,2))              # search ranges
```

**Key differences from ARIMA:**
- ARARMA does **not** support `d()`, `D()`, `P()`, or `Q()` terms (differencing is handled by the ARAR stage)
- ARARMA does **not** support exogenous regressors (no variables, no `auto()`)
- ARARMA adds ARAR-specific parameters: `max_ar_depth` and `max_lag`

### Automatic vs Fixed Order Selection

**If ANY order is a range** → uses `auto_ararma()`:
- `p() + q()` → searches with defaults (p: 0-4, q: 0-2)
- `p(0,3) + q()` → searches p ∈ {0,1,2,3}, q with defaults
- `p(1) + q(0,2)` → searches q ∈ {0,1,2} with fixed p=1

**If ALL orders are fixed** → uses `ararma()` directly (faster):
- `p(1) + q(2)` → fits ARARMA(1,2) without search

### Direct formula fitting

```julia
using Durbyn
using Durbyn.Ararma

data = (value = air_passengers(),)

# Fixed ARARMA(1,2)
formula = @formula(value = p(1) + q(2))
ararma_model = ararma(formula, data)
fc = forecast(ararma_model; h = 12)

# Auto ARARMA with custom parameters
formula = @formula(value = p() + q())
ararma_model = ararma(formula, data, max_ar_depth=20, max_lag=30, crit=:bic)
fc = forecast(ararma_model; h = 12)
```

The estimator lives in the `Durbyn.Ararma` submodule. It works with any Tables.jl source and returns an `ArarmaModel` struct.

### Model specification (`ArarmaSpec`)

To leverage grouped fitting, forecasting, and model collections, wrap the formula in `ArarmaSpec`:

```julia
# Fixed ARARMA(2,1)
spec = ArarmaSpec(@formula(value = p(2) + q(1)))
fitted = fit(spec, data)
fc = forecast(fitted; h = 8)

# Auto ARARMA with custom ARAR parameters
spec = ArarmaSpec(
    @formula(value = p() + q()),
    max_ar_depth = 20,
    max_lag = 30,
    crit = :bic
)
fitted = fit(spec, data)
fc = forecast(fitted; h = 12)
```

For panel data:

```julia
panel = PanelData(tbl; groupby = :region, m = m)
spec = ArarmaSpec(@formula(value = p(1) + q(1)))
group_fit = fit(spec, panel)
group_fc = forecast(group_fit; h = 6)
```

And to compare against other specs:

```julia
models = model(
    ArarmaSpec(@formula(value = p() + q())),
    ArarSpec(@formula(value = arar())),
    ArimaSpec(@formula(value = p() + q() + P() + Q())),
    EtsSpec(@formula(value = e("Z") + t("Z") + s("Z"))),
    names = ["ararma", "arar", "arima", "ets"]
)

fitted = fit(models, panel)
fc = forecast(fitted; h = 12)
```

The ARARMA grammar therefore integrates seamlessly with every Durbyn workflow—single series, grouped/panel data, and large-scale model comparisons.

---

## Tips and Best Practices

### ARIMA Tips

- Any range triggers automatic model selection
- Fixed orders call fast direct estimation
- Exogenous support includes explicit columns, `auto()`, or complex formulas
- Combine with `PanelData` to store group/date metadata cleanly
- If you omit `newdata` when forecasting, Durbyn reuses each group's most recent exogenous values

### ETS Tips

- Always specify `m` (seasonal period) when you expect seasonal behavior. If you omit it, ETS defaults to `m = 1`
- Keywords like `lambda`, `alpha`, or `ic` are forwarded directly to the underlying `ets` implementation
- Grouped fits reuse the same grammar—`fit(spec, data; groupby = [:region])` returns `GroupedFittedModels`
- Forecast works the same way for both single and grouped models

### General Tips

- Use `PanelData` to encapsulate grouping, date, and seasonal period information
- Specifications are reusable—define once, fit to multiple datasets
- Keywords in `fit()` override those stored in the spec
- `forecast_table()` provides tidy output for downstream analysis and visualization
- Combine multiple specs in a `ModelCollection` for easy model comparison
