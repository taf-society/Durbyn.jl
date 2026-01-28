# Naive Forecasting Methods

!!! tip "Formula Interface is the Recommended Approach"
    Use `NaiveSpec`, `SnaiveSpec`, `RwSpec`, or `MeanfSpec` with `@formula` for a declarative interface
    that works with panel data, grouped fitting, and model comparison. The base (array) API is shown near the end.

Naive forecasting methods serve as simple **benchmark models** for more complex forecasting approaches.
Despite their simplicity, they often perform surprisingly well, especially for highly volatile or
unpredictable data. Durbyn implements four naive methods:

| Method | Function | Description | Best for |
|--------|----------|-------------|----------|
| **Naive** | `naive()` | Uses last observation as forecast | Random walk data, benchmarks |
| **Seasonal Naive** | `snaive()` | Uses observation from m periods ago | Strong seasonal patterns |
| **Random Walk with Drift** | `rw(drift=true)` | Naive + linear trend | Trending data |
| **Mean** | `meanf()` | Uses sample mean as forecast | Stationary data |

---

## Formula Interface (primary usage)

### Example 1: Naive forecast

```julia
using Durbyn, Durbyn.Grammar

data = (sales = [120, 135, 148, 152, 141, 158, 170, 165, 180, 195],)

# Naive: forecast = last observation
spec = NaiveSpec(@formula(sales = naive_term()))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)
```

### Example 2: Seasonal naive forecast

```julia
using Durbyn, Durbyn.Grammar

# Monthly sales data with yearly seasonality
data = (sales = [100, 110, 125, 140, 155, 170,
                 160, 150, 135, 120, 105, 95,
                 105, 115, 130, 145, 160, 175,
                 165, 155, 140, 125, 110, 100],)

# Seasonal naive: forecast = value from same month last year
spec = SnaiveSpec(@formula(sales = snaive_term()), m = 12)
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)
```

### Example 3: Random walk with drift

```julia
using Durbyn, Durbyn.Grammar

# Trending data
data = (value = cumsum(randn(50) .+ 0.5),)

# RW with drift: forecast = last value + h * average change
spec = RwSpec(@formula(value = rw_term(drift = true)))
fitted = fit(spec, data)
fc = forecast(fitted, h = 10)

# Access drift coefficient
fitted.fit.drift      # Drift value
fitted.fit.drift_se   # Drift standard error
```

### Example 4: Mean forecast

```julia
using Durbyn, Durbyn.Grammar

# Stationary data around a mean
data = (temp = 20.0 .+ randn(100),)

# Mean: forecast = sample mean for all horizons
spec = MeanfSpec(@formula(temp = meanf_term()))
fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 12)

# Access mean
fitted.fit.mu_original  # Mean on original scale
```

### Example 5: Model comparison

```julia
using Durbyn, Durbyn.Grammar

data = (y = [100, 105, 102, 108, 115, 120, 118, 125, 130, 128],)

# Compare naive methods
naive_spec = NaiveSpec(@formula(y = naive_term()))
rw_spec = RwSpec(@formula(y = rw_term(drift = true)))
mean_spec = MeanfSpec(@formula(y = meanf_term()))

naive_fit = fit(naive_spec, data)
rw_fit = fit(rw_spec, data)
mean_fit = fit(mean_spec, data, m = 1)

# Compare residual variance
naive_fit.fit.sigma2  # Naive residual variance
rw_fit.fit.sigma2     # RW with drift residual variance
```

### Example 6: Panel data / grouped fitting

```julia
using Durbyn, Durbyn.TableOps, Durbyn.ModelSpecs, Durbyn.Grammar

# Stacked table with :product column
panel = PanelData(tbl; groupby = :product, date = :date, m = 12)

# Fit seasonal naive to each group
spec = SnaiveSpec(@formula(sales = snaive_term()))
fitted = fit(spec, panel)
fc = forecast(fitted, h = 12)
```

### Example 7: Box-Cox transformation

```julia
using Durbyn, Durbyn.Grammar

# Positive data with increasing variance
data = (sales = exp.(cumsum(randn(50) .* 0.1 .+ 0.05)),)

# Apply log transformation (lambda = 0) with bias adjustment
spec = NaiveSpec(@formula(sales = naive_term()), lambda = 0.0, biasadj = true)
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)
```

---

## Base API (array interface)

```julia
using Durbyn

y = randn(100)

# Naive forecast
fit_naive = naive(y)
fc = forecast(fit_naive; h = 12)

# Seasonal naive (monthly data)
y_seasonal = repeat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3)
fit_snaive = snaive(y_seasonal, 12)
fc = forecast(fit_snaive; h = 24)

# Random walk with drift
y_trend = cumsum(randn(100))
fit_rw = rw(y_trend; drift = true)
fc = forecast(fit_rw; h = 10)

# Mean forecast
fit_mean = meanf(y, 1)
fc = forecast(fit_mean, 12)
```

`NaiveFit` exposes `fitted`, `residuals`, `sigma2`, `lag`, `drift`, `drift_se`, and optional
Box-Cox parameters (`lambda`, `biasadj`).

---

## Methodology

### Naive Method

The naive forecast uses the last observed value as the forecast for all future horizons:

```math
\hat{y}_{T+h|T} = y_T \quad \text{for all } h = 1, 2, \ldots
```

This is equivalent to a random walk model without drift.

### Seasonal Naive Method

The seasonal naive method uses the observation from the same season in the previous cycle:

```math
\hat{y}_{T+h|T} = y_{T+h-m \cdot k}
```

where ``m`` is the seasonal period and ``k = \lceil h/m \rceil`` is the number of complete
seasonal cycles back.

For example, with monthly data (``m = 12``), the forecast for January next year uses
the value from January this year.

### Random Walk with Drift

The random walk with drift includes a linear trend based on the average historical change:

```math
\hat{y}_{T+h|T} = y_T + h \cdot b
```

where the drift term ``b`` is estimated as:

```math
b = \frac{y_T - y_1}{T - 1}
```

This is the average change per period over the entire series.

### Mean Method

The mean method uses the historical average as the forecast:

```math
\hat{y}_{T+h|T} = \bar{y} = \frac{1}{T}\sum_{t=1}^{T} y_t
```

This assumes the data fluctuates around a constant mean with no trend or seasonality.

---

## Prediction Intervals

Naive methods produce prediction intervals that widen with the forecast horizon,
reflecting increasing uncertainty over time.

### Naive / Random Walk (without drift)

Forecast variance grows linearly with horizon:

```math
\text{Var}(\hat{y}_{T+h|T}) = h \cdot \sigma^2
```

The standard error is:

```math
\text{SE}(h) = \sqrt{h} \cdot \sigma
```

### Seasonal Naive

Variance increases with the number of complete seasonal cycles:

```math
\text{Var}(\hat{y}_{T+h|T}) = \lceil h/m \rceil \cdot \sigma^2
```

where ``m`` is the seasonal period. The variance increases in steps each time a new
seasonal cycle is entered.

### Random Walk with Drift

Includes additional uncertainty from the drift estimate:

```math
\text{Var}(\hat{y}_{T+h|T}) = h \cdot \sigma^2 + h^2 \cdot \text{SE}(b)^2
```

where ``\text{SE}(b) = \sigma / \sqrt{T-1}`` is the standard error of the drift coefficient.

### Mean Method

Uses the t-distribution for intervals:

```math
\hat{y}_{T+h|T} \pm t_{1-\alpha/2, T-1} \cdot s \cdot \sqrt{1 + 1/T}
```

where ``s`` is the sample standard deviation and ``T`` is the number of observations.

---

## When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| No clear pattern, random fluctuations | `naive()` |
| Strong, stable seasonal pattern | `snaive()` |
| Clear upward or downward trend | `rw(drift=true)` |
| Stationary data around a level | `meanf()` |
| Benchmark for complex models | Any naive method |

!!! info "Benchmarking Best Practice"
    Always compare your forecast model against naive benchmarks. If a sophisticated model
    cannot beat the seasonal naive method, consider whether the added complexity is justified.

---

## Missing Value Handling

All naive methods handle missing values gracefully:

- **Leading missings**: Skipped when finding the starting point
- **Trailing missings**: The last valid observation is used
- **Scattered missings**: Converted to NaN internally; forecasts use the most recent valid value

```julia
y = [1.0, missing, 3.0, 4.0, missing, 6.0]
fit = naive(y)  # Uses 6.0 (last valid) for forecasts
```

For seasonal naive, if a particular seasonal position has missing values, the method
searches backwards through prior seasonal cycles to find the most recent valid observation
at that position.

---

## Box-Cox Transformation

All naive methods support Box-Cox transformation for variance stabilization:

```julia
# Log transformation (lambda = 0)
fit = naive(y; lambda = 0.0)

# Square root transformation (lambda = 0.5)
fit = naive(y; lambda = 0.5)

# Automatic lambda selection
fit = naive(y; lambda = "auto")

# With bias adjustment for back-transformation
fit = naive(y; lambda = 0.0, biasadj = true)
```

The `biasadj` option applies a bias correction when transforming forecasts back to the
original scale, which can improve accuracy for skewed distributions.

---

## References

- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.).
  OTexts. [https://otexts.com/fpp3/](https://otexts.com/fpp3/)
- Makridakis, S., Wheelwright, S. C., & Hyndman, R. J. (1998). *Forecasting: Methods and
  Applications* (3rd ed.). John Wiley & Sons.
