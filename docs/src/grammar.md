# Durbyn Grammar

Durbyn provides an expressive, composable grammar for defining forecasting models. This unified interface lets you describe ARIMA, SARIMA, and exponential smoothing models with concise, readable syntax using the `@formula` macro and specialized model specifications.

---

## Overview

The Durbyn grammar system consists of:

- **Formula interface**: Use `@formula` to declaratively specify model components
- **Model specifications**: Wrap formulas in specs like `ArimaSpec`, `EtsSpec`, `SesSpec`, etc.
- **Unified fitting**: Call `fit(spec, data)` with optional grouping for panel data
- **Consistent forecasting**: Use `forecast(fitted, h)` for both single and grouped models

This design eliminates manual tuning loops and provides a consistent interface across all model families.

---

## ARIMA Grammar

The ARIMA grammar lets you describe ARIMA and SARIMA models with flexible order specifications and exogenous variable support.

### Formula Basics

Define the relationship between a target series and its ARIMA structure:

```julia
@formula(sales = p() + d() + q())
```

Every formula requires a target (left-hand side) and one or more terms (right-hand side). Terms may specify ARIMA orders, seasonal orders, or exogenous variables.

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

Add predictors by listing column names:

```julia
@formula(sales = p() + q() + price + promotion)
```

These become `VarTerm`s—during fitting, Durbyn pulls the matching columns from your data.

#### Automatic Selection (`auto()`)

Use `auto()` to include all numeric columns except the target, group columns, and optional date column:

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

Use `@formula` to define the target series and its ETS components:

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

## Multi-Model Comparison

Use `ModelCollection` to fit and compare multiple specifications simultaneously:

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

`forecast_table` stacks every model (and group) with a `model_name` column, so downstream comparisons stay tidy. You can always filter to a specific model or pivot wider if needed.

---

## Complete End-to-End Example

Here's a complete workflow using the ARIMA grammar with panel data:

```julia
using CSV
using Downloads
using Tables
using Durbyn
using Durbyn.TableOps

# Download and reshape data
local_path = Downloads.download(
    "https://raw.githubusercontent.com/Akai01/example-time-series-datasets/refs/heads/main/Data/retail.csv"
)
retail = CSV.File(local_path)
tbl = Tables.columntable(retail)

glimpse(tbl)  # Column overview

# Convert to long format
tbl_long = pivot_longer(tbl; id_cols = :date, names_to = :series, values_to = :value)
glimpse(tbl_long)

# Wrap in PanelData with group/date metadata
panel_tbl = PanelData(tbl_long; groupby = :series, date = :date, m = 12)

# Auto ARIMA with automatic xreg selection
spec = ArimaSpec(@formula(value = auto()))
# Alternative specifications:
# spec = ArimaSpec(@formula(value = p() + d() + q() + P() + D() + Q()))  # Auto ARIMA
# spec = ArimaSpec(@formula(value = p(1)))                               # ARIMA(1,0,0)

fitted = fit(spec, panel_tbl)

# Inspect grouped result
glimpse(fitted)

# Forecast 12 steps ahead for every series
fc = forecast(fitted, h = 12)

# Convert to tidy forecast table
fc_tbl = forecast_table(fc)
glimpse(fc_tbl)
```

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
