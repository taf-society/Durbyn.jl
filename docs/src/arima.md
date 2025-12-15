
# Forecasting Using ARIMA, SARIMA, ARIMAX, SARIMAX, and Auto ARIMA

!!! tip "Formula Interface is the Recommended Approach"
    This page starts with the **formula interface** (recommended for most users),
    which provides declarative model specification with support for regressors, panel data,
    and model comparison. The array interface (base models) is covered later.
    See the **[Grammar Guide](grammar.md)** for complete documentation.

## 1. ARIMA (AutoRegressive Integrated Moving Average)

### Definition
An ARIMA model is denoted as **ARIMA(p, d, q)**, where:
- **p**: order of the autoregressive (AR) part
- **d**: degree of differencing needed to achieve stationarity
- **q**: order of the moving average (MA) part

Formally, the model is written as:

```math
\Phi(B) \Delta^d X_t = \Theta(B) \varepsilon_t,
```

where:

- ``B`` is the backshift operator (``BX_t = X_{t-1}``),
- ``\Phi(B) = 1 - \phi_1B - \cdots - \phi_pB^p``,
- ``\Theta(B) = 1 + \theta_1B + \cdots + \theta_qB^q``,
- ``\Delta^d = (1 - B)^d`` is the differencing operator,
- ``\varepsilon_t`` is white noise.

If ``d = 0``, the model reduces to ARMA(p, q).

### Key Features
- Handles **non-stationary time series** via differencing.
- Shocks (innovations) have **permanent effects** for ``d > 0``.
- Commonly used for macroeconomic and financial data.

---

## 2. SARIMA (Seasonal ARIMA)

### Definition
Seasonal ARIMA extends ARIMA to account for **seasonality**. It is denoted as:

```math
ARIMA(p, d, q)(P, D, Q)_m,
```

where:
- ``P, D, Q`` are the seasonal AR, differencing, and MA orders,
- ``m`` is the seasonal period (e.g., 12 for monthly data with yearly seasonality).

### Model Form
```math
\Phi(B)\Phi_s(B^m) \Delta^d \Delta_m^D X_t = \Theta(B)\Theta_s(B^m)\varepsilon_t,
```

where:
- ``\Phi_s(B^m)`` and ``\Theta_s(B^m)`` capture seasonal AR and MA terms,
- ``\Delta_m^D = (1 - B^m)^D`` applies seasonal differencing.

### Key Features
- Captures both **short-term dynamics** (p, d, q) and **seasonal effects** (P, D, Q).
- Widely applied to monthly or quarterly economic indicators, sales, or climate data.

---

## 3. ARIMAX (ARIMA with Exogenous Variables)

### Definition
An ARIMAX model incorporates external regressors (covariates) into the ARIMA framework:

```math
\Phi(B) \Delta^d X_t = \beta Z_t + \Theta(B) \varepsilon_t,
```

where:

- ``Z_t`` is a vector of exogenous predictors,
- ``\beta`` are their coefficients.

### Key Features
- Useful when external factors (e.g., interest rates, marketing spend, policy variables) explain additional variance beyond past values of the series.
- Requires careful checking of exogeneity assumptions.

---

## 4. SARIMAX (Seasonal ARIMAX)

### Definition
SARIMAX generalizes SARIMA by including **exogenous regressors**:

```math
\Phi(B)\Phi_s(B^m) \Delta^d \Delta_m^D X_t = \beta Z_t + \Theta(B)\Theta_s(B^m)\varepsilon_t.
```

### Key Features
- Combines **seasonality** and **exogenous influences**.
- Powerful for real-world applications such as:
  - Forecasting retail sales with promotions (exogenous variable) and seasonal cycles.
  - Modeling energy demand with weather as an exogenous driver.

---

## 5. Auto ARIMA

### Definition
**Auto ARIMA** automates the process of identifying the best ARIMA/SARIMA model by searching across possible values of (p, d, q) and seasonal (P, D, Q), selecting the model that minimizes an information criterion such as AIC, AICc, or BIC.

### Algorithm (Hyndman & Khandakar, 2008)
1. **Unit root tests** (ADF, KPSS, or combinations) to determine differencing orders \( d \) and \( D \).
2. **Initial model selection** based on heuristics.  
3. **Stepwise search** over (p, q, P, Q) with bounds (e.g., up to 5 for non-seasonal and 2 for seasonal).  
4. Evaluate models by likelihood and information criteria.  
5. Refit the best model with full maximum likelihood.  

### Advantages
- Removes the manual effort of model identification.  
- Scales well to large numbers of series.  
- Ensures differencing is tested systematically (avoids over-differencing).

### Limitations
- Stepwise search may not find the global optimum.  
- Computationally expensive for very large seasonal periods.  
- Still requires diagnostic checking of residuals.  

---

## 6. Model Selection & Diagnostics

### Identification
- Use **ACF/PACF plots** and **unit root tests** (ADF, PP, KPSS) to choose orders manually (or confirm Auto ARIMA results).
- Differencing ensures stationarity (``d, D``).

### Estimation
- Maximum Likelihood Estimation (MLE) or Conditional Sum of Squares.

### Diagnostics
- Residual analysis: check for white noise.
- Information criteria: AIC, BIC, AICc.  
- Out-of-sample forecast validation.

---

# Formula Interface (Primary Usage)

The formula interface provides a modern, declarative way to specify ARIMA models with full support for single series, regressors, model comparison, and panel data.

## Example 1: Single ARIMA Model

```julia
using Durbyn

# Load data
data = (sales = [120, 135, 148, 152, 141, 158, 170, 165, 180, 195],)

# Specify model with automatic order selection
spec = ArimaSpec(@formula(sales = p() + q() + P() + Q() + d() + D()))
fitted_model = fit(spec, data, m = 12)
fc = forecast(fitted_model, h = 12)

# Check model summary
println(fitted_model)

# Access fitted values and residuals
fitted_values = fitted(fitted_model)
resids = residuals(fitted_model)
```

**Key features:**
- `p()`, `q()`, `P()`, `Q()`, `d()` and `D()` with no arguments triggers automatic order selection
- `m = 12` specifies monthly seasonality
- Formula syntax clearly shows response variable (`sales`)

## Example 2: ARIMA with Regressors

When you have external variables that influence the response, include them as regressors:

```julia
# Model with exogenous regressors
data = (
    sales = rand(100),
    temperature = rand(100),
    promotion = rand(0:1, 100)
)

# Specify model with regressors
spec = ArimaSpec(@formula(sales = p(1,3) + q(1,3) + temperature + promotion))
fitted = fit(spec, data, m = 7)

# Forecast requires future regressor values
newdata = (temperature = rand(7), promotion = rand(0:1, 7))
fc = forecast(fitted, h = 7, newdata = newdata)
```

**Terminology:**
- **Response variable**: The variable being forecasted (`sales`)
- **Regressors**: External predictors (`temperature`, `promotion`)

**Key features:**
- `p(1,3)` starts searching for best AR order between 1 and 3
- Regressors are simply added to the formula
- Future regressor values must be provided via `newdata`

## Example 3: Manual ARIMA Specification

For full control over model orders:

```julia
# Specify exact orders for SARIMA model
spec = ArimaSpec(@formula(sales = p(2) + d(1) + q(1) + P(1) + D(1) + Q(1)))
fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 12)

# Or use specific values with regressors
spec = ArimaSpec(@formula(sales = p(1) + d(1) + q(1) + temperature + promotion))
fitted = fit(spec, data, m = 12)
```

**ARIMA order specification:**
- `p(k)`: AR order = k
- `d(k)`: Differencing order = k
- `q(k)`: MA order = k
- `P(k)`: Seasonal AR order = k
- `D(k)`: Seasonal differencing = k
- `Q(k)`: Seasonal MA order = k

## Example 4: Fitting Multiple Models Together

Fit different model specifications and manually compare results:

```julia
# Define multiple candidate models
models = model(
    ArimaSpec(@formula(sales = p() + q())),                    # Auto ARIMA
    ArimaSpec(@formula(sales = p(2) + d(1) + q(2))),          # ARIMA(2,1,2)
    ArimaSpec(@formula(sales = p(1) + d(1) + q(1) + P(1) + D(1) + Q(1))),  # SARIMA
    names = ["auto_arima", "arima_212", "sarima_111_111"]
)

# Fit all models
fitted = fit(models, data, m = 12)

# Forecast with all models
fc = forecast(fitted, h = 12)

# Check forecast accuracy
accuracy(fc, test)
```

**Key features:**
- Fit multiple specifications at once
- Mix different model types (ARIMA, ETS, etc.)
- Check model accuracy
- Forecasts generated for all models

## Example 5: Panel Data (Multiple Time Series)

Fit the same model specification to many series efficiently:

```julia
using Durbyn.TableOps
using CSV, Downloads

# Load panel data
path = Downloads.download("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/refs/heads/main/Data/retail.csv")
wide = Tables.columntable(CSV.File(path))

# Reshape to long format
long = pivot_longer(wide; id_cols = :date, names_to = :series, values_to = :value)

# Create panel data wrapper
panel = PanelData(long; groupby = :series, date = :date, m = 12)

# Fit model to all series at once
spec = ArimaSpec(@formula(value = p() + q()))
fitted = fit(spec, panel)

# Forecast all series
fc = forecast(fitted, h = 12)

# Get tidy forecast table
tbl = as_table(fc)

# Optional: Save forecasts to CSV
# CSV.write("forecasts.csv", tbl)

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

**Panel data features:**
- Fits model separately to each group
- Returns structured output for all series
- `as_table` creates tidy format for analysis
- Efficient for hundreds or thousands of series

## Example 6: Panel Data with Grouping Variables

For complex panel structures:

```julia

# Use PanelData interface
panel = PanelData(train; groupby=[:product, :location, :product_line], date=:date, m=7);

spec = ArimaSpec(@formula(sales = p() + q()))
fitted = fit(spec, panel)
fc = forecast(fitted, h = 14)

# Data with multiple grouping variables
spec = ArimaSpec(@formula(sales = p() + q()))
fitted = fit(spec, data,
             groupby = [:product, :location, :product_line],
             m = 7)
fc = forecast(fitted, h = 7)

# Filter forecasts for specific groups
tbl = as_table(fc)

```

---

# Array Interface (Base Models)

The array interface provides direct access to ARIMA estimation for numeric vectors.
This is useful for quick analyses or integration with existing code for example using Durbyn base models as backend for Python or R packages.

## Forecasting Using Seasonal ARIMA Model

```julia
using Durbyn
using Durbyn.Arima

ap  = air_passengers()
arima_model = arima(ap, 12, order = PDQ(2,1,1), seasonal = PDQ(0,1,0))
fc  = forecast(arima_model, h = 12)
plot(fc)

```

## Forecasting Using Auto-ARIMA Model
```julia
auto_arima_model = auto_arima(ap, 12)
fc2  = forecast(auto_arima_model, h = 12)
plot(fc2)
```


## References
- Kunst, R. (2011). *Applied Time Series Analysis — Part II*. University of Vienna.  
- Hyndman, R.J., & Khandakar, Y. (2008). *Automatic Time Series Forecasting: The forecast Package for R*. Journal of Statistical Software, 27(3).  
- Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (1994). *Time Series Analysis, Forecasting and Control*.  
- Hamilton, J.D. (1994). *Time Series Analysis*.  
