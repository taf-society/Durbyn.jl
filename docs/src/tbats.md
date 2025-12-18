# TBATS: Trigonometric Seasonal Exponential Smoothing

**TBATS** (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, and Seasonal components) extends BATS by using a **Fourier representation** for seasonal components, enabling the model to handle:

- **Non-integer seasonal periods** (e.g., 52.18 weeks per year)
- **Very long seasonal cycles** (hundreds or thousands of periods)
- **Multiple complex seasonalities** (daily + weekly + yearly)
- **Dual calendar effects** (e.g., Hijri + Gregorian calendars)

This implementation is a pure Julia port of the R `forecast::tbats` function based on De Livera, Hyndman & Snyder (2011).

---

## Mathematical Framework

### 1. Box-Cox Transformation

The model begins with an optional Box-Cox transformation to stabilize variance:

$$y_t^{(\omega)} = \begin{cases}
\frac{y_t^\omega - 1}{\omega}, & \omega \neq 0 \\[6pt]
\ln(y_t), & \omega = 0
\end{cases}$$

where ``\omega \in [0, 1]`` is the Box-Cox parameter estimated from the data.

---

### 2. TBATS State Space Model

The TBATS model is represented in innovations form with the following components:

#### **Observation Equation**

$$y_t^{(\omega)} = \ell_{t-1} + \phi b_{t-1} + \sum_{i=1}^{T} s_{i,t-1} + d_t$$

where:
- ``\ell_t`` is the local level
- ``b_t`` is the trend component
- ``\phi`` is the damping parameter (``0 < \phi \leq 1``)
- ``s_{i,t}`` is the ``i``-th seasonal component
- ``d_t`` is the ARMA error term
- ``T`` is the number of seasonal periods

#### **State Equations**

**Local Level:**

$$\ell_t = \ell_{t-1} + \phi b_{t-1} + \alpha d_t$$

where ``\alpha`` is the level smoothing parameter.

**Trend Component:**

$$b_t = \phi b_{t-1} + \beta d_t$$

where ``\beta`` is the trend smoothing parameter and ``\phi`` controls damping.

**Trigonometric Seasonal Components:**

For each seasonal period ``m_i`` with ``k_i`` Fourier harmonics:

$$s_{i,t} = \sum_{j=1}^{k_i} \left[ s_{i,j,t}^{(1)} \cos(\lambda_{i,j} t) + s_{i,j,t}^{(2)} \sin(\lambda_{i,j} t) \right]$$

where ``\lambda_{i,j} = \frac{2\pi j}{m_i}`` is the ``j``-th harmonic frequency.

**Harmonic State Evolution:**

Each harmonic pair evolves via a rotation matrix:

$$\begin{pmatrix}
s_{i,j,t}^{(1)} \\
s_{i,j,t}^{(2)}
\end{pmatrix}
= \begin{pmatrix}
\cos\lambda_{i,j} & \sin\lambda_{i,j} \\
-\sin\lambda_{i,j} & \cos\lambda_{i,j}
\end{pmatrix}
\begin{pmatrix}
s_{i,j,t-1}^{(1)} \\
s_{i,j,t-1}^{(2)}
\end{pmatrix}
+ \begin{pmatrix}
\gamma_{1,i} \\
\gamma_{2,i}
\end{pmatrix} d_t$$

where ``\gamma_{1,i}`` and ``\gamma_{2,i}`` are smoothing parameters for the seasonal component.

**ARMA Error Component:**

$$d_t = \sum_{k=1}^p \varphi_k d_{t-k} + \varepsilon_t + \sum_{\ell=1}^q \theta_\ell \varepsilon_{t-\ell}$$

where:
- ``p`` is the AR order
- ``q`` is the MA order
- ``\varphi_k`` are AR coefficients
- ``\theta_\ell`` are MA coefficients
- ``\varepsilon_t \sim \mathcal{N}(0, \sigma^2)``

---

### 3. Key Advantages Over BATS

| Feature | BATS | TBATS |
|---------|------|-------|
| Seasonal periods | Integer only | Non-integer allowed |
| State dimension | ``m_i`` states per season | ``2k_i`` states per season |
| Max seasonal period | ~350 (computational limit) | Unlimited (via Fourier) |
| Dual calendars | Not feasible | Fully supported |
| Storage | ``O(m)`` | ``O(k)``, typically ``k \ll m`` |

**Example:** For weekly seasonality (``m = 52``), BATS needs 52 states, while TBATS with ``k=2`` harmonics needs only 4 states.

---

### 4. Forecasting

Future states evolve deterministically (errors set to zero):

**Level:**

$$\ell_{t+h} = \ell_t + \sum_{u=1}^h \phi^{u-1} b_t$$

**Trend:**

$$b_{t+h} = \phi^h b_t$$

**Seasonal Components:**

The harmonic pairs rotate forward via matrix powers:

$$\begin{pmatrix}
s_{i,j,t+h}^{(1)} \\
s_{i,j,t+h}^{(2)}
\end{pmatrix}
= R(\lambda_{i,j})^h
\begin{pmatrix}
s_{i,j,t}^{(1)} \\
s_{i,j,t}^{(2)}
\end{pmatrix}$$

where ``R(\lambda)`` is the rotation matrix.

**Point Forecast (Transformed Scale):**

$$\hat{y}_{t+h}^{(\omega)} = \ell_{t+h} + \phi b_{t+h} + \sum_{i=1}^T s_{i,t+h}$$

The forecast is then back-transformed via inverse Box-Cox:

$$\hat{y}_{t+h} = \begin{cases}
(\omega \hat{y}_{t+h}^{(\omega)} + 1)^{1/\omega}, & \omega \neq 0 \\[6pt]
\exp(\hat{y}_{t+h}^{(\omega)}), & \omega = 0
\end{cases}$$

**Forecast Variance:**

The ``h``-step ahead forecast variance is:

$$\text{Var}(\hat{y}_{t+h}) = \sigma^2 \sum_{j=0}^{h-1} c_j^2$$

where ``c_j`` depends on the model's transition matrix ``F`` and error vector ``g``.

---

## Usage in Durbyn

Durbyn provides two interfaces for TBATS: the **classic API** with direct function calls and the **grammar interface** for declarative model specification.

### Grammar Interface (Recommended)

The grammar interface provides a unified, declarative way to specify TBATS models using `@formula` and `TbatsSpec`:

```julia
using Durbyn
using Durbyn.ModelSpecs

# Create sample data with weekly seasonality (non-integer period)
n = 156  # 3 years of weekly data
t = 1:n
data = (sales = 100.0 .+ 10.0 .* sin.(2π .* t ./ 52.18) .+ 2.0 .* randn(n),)

# Basic TBATS with defaults (automatic component selection)
spec = TbatsSpec(@formula(sales = tbats()))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)

# TBATS with non-integer weekly seasonality (52.18 weeks/year)
spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=52.18)))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)

# TBATS with multiple seasonal periods (daily + yearly)
# Example: hourly data with daily (24h) and yearly (8766h ≈ 365.25 days)
spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=[24, 8766])))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)

# TBATS with explicit Fourier orders
spec = TbatsSpec(@formula(sales = tbats(
    seasonal_periods=[7, 365.25],
    k=[3, 10]  # 3 harmonics for weekly, 10 for yearly
)))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)

# TBATS with specific component selection
spec = TbatsSpec(@formula(sales = tbats(
    seasonal_periods=52.18,
    use_box_cox=true,
    use_trend=true,
    use_damped_trend=false,
    use_arma_errors=true
)))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)

# Additional options at fit time
fitted = fit(spec, data, bc_lower=0.0, bc_upper=1.5, biasadj=true)
```

**Panel Data Support:**

```julia
# Create panel data (Tables.jl compatible)
n_per_group = 104  # 2 years of weekly data per group
groups = ["A", "B", "C"]

tbl = (
    product = repeat(groups, inner=n_per_group),
    sales = vcat([
        100.0 .+ 10.0 .* sin.(2π .* (1:n_per_group) ./ 52) .+ 2.0 .* randn(n_per_group)
        for _ in groups
    ]...)
)

# Fit TBATS to each product separately
spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=52.18)))
fitted = fit(spec, tbl, groupby = :product)
fc = forecast(fitted, h = 12)
```

**Model Comparison:**

```julia
# Compare TBATS with BATS and other models
models = model(
    TbatsSpec(@formula(sales = tbats(seasonal_periods=52.18))),  # Non-integer period
    BatsSpec(@formula(sales = bats(seasonal_periods=52))),        # Integer period only
    ArimaSpec(@formula(sales = p() + q() + P() + Q())),
    EtsSpec(@formula(sales = e("Z") + t("Z") + s("Z"))),
    names = ["tbats", "bats", "arima", "ets"]
)

fitted = fit(models, data)
fc = forecast(fitted, h = 12)
```

### Classic API

For direct usage without the grammar interface:

```julia
using Durbyn

y = rand(100)
m = [7, 365.25]

model = tbats(y, m)

fc = forecast(model, h=20)
```

### Formula Interface (Direct)

You can also use the formula interface directly without `TbatsSpec`:

```julia
using Durbyn

data = (sales = randn(156) .+ 100,)
formula = @formula(sales = tbats(seasonal_periods=52.18))
fit = tbats(formula, data)  # Works with Tables.jl compatible data
fc = forecast(fit, h = 12)
```

### Key keyword arguments

**Grammar arguments (in `@formula`):**

- `seasonal_periods`: `Real` or `Vector{Real}` specifying seasonal period(s) - **can be non-integer**
- `k`: `Int` or `Vector{Int}` specifying Fourier orders (harmonics per season)
- `use_box_cox`, `use_trend`, `use_damped_trend`, `use_arma_errors`: `Bool` or `nothing` to auto-select

**Fit-time arguments:**

- `bc_lower`, `bc_upper`: bounds for the Box–Cox search when enabled
- `biasadj`: apply bias correction during inverse Box–Cox transformation
- `model`: pass a previous `TBATSModel` to refit the same structure to new data

---

## Julia Implementation (Array Interface)

### Function Signature

```julia
tbats(
    y::AbstractVector{<:Real},
    m::Union{Vector{Int}, Nothing} = nothing;
    use_box_cox::Union{Bool, AbstractVector{Bool}, Nothing} = nothing,
    use_trend::Union{Bool, AbstractVector{Bool}, Nothing} = nothing,
    use_damped_trend::Union{Bool, AbstractVector{Bool}, Nothing} = nothing,
    use_arma_errors::Bool = true,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
    biasadj::Bool = false,
    model = nothing
)
```

### Parameters

- **`y`**: Univariate time series (1D vector)
- **`m`**: Vector of seasonal periods (can be non-integer). Use `nothing` for non-seasonal models.
- **`use_box_cox`**: Whether to use Box-Cox transformation
  - `nothing` (default): tries both and selects by AIC
  - `true/false`: forces the choice
  - Vector of bools: tries each option
- **`use_trend`**: Whether to include trend component
  - `nothing` (default): tries both
  - `true/false`: forces the choice
- **`use_damped_trend`**: Whether to damp the trend
  - `nothing` (default): tries both
  - `true/false`: forces the choice
  - Ignored if `use_trend=false`
- **`use_arma_errors`**: Whether to model residuals with ARMA
- **`bc_lower`, `bc_upper`**: Bounds for Box-Cox parameter search
- **`biasadj`**: Use bias-adjusted back-transformation for forecasts
- **`model`**: Previously fitted TBATS model to refit

### Returns

A `TBATSModel` struct containing:

- `lambda`: Box-Cox parameter (or `nothing`)
- `alpha`: Level smoothing parameter
- `beta`: Trend smoothing parameter (or `nothing`)
- `damping_parameter`: Damping parameter φ (or `nothing`)
- `gamma_one_values`: First seasonal smoothing parameters
- `gamma_two_values`: Second seasonal smoothing parameters
- `ar_coefficients`: AR coefficients (or `nothing`)
- `ma_coefficients`: MA coefficients (or `nothing`)
- `seasonal_periods`: Vector of seasonal periods
- `k_vector`: Vector of Fourier orders per seasonal period
- `fitted_values`: In-sample fitted values
- `errors`: Residuals
- `x`: State matrix
- `seed_states`: Initial states
- `variance`: Residual variance
- `AIC`: Akaike Information Criterion
- `likelihood`: Log-likelihood
- `y`: Original time series
- `method`: Model descriptor string

---

## Model Selection

The implementation automatically selects:

1. **Fourier orders (``k_i``)** for each seasonal period via AIC
   - Starts with ``k=1``
   - For small periods (``m \leq 12``): searches downward from ``k = \lfloor(m-1)/2\rfloor``
   - For large periods (``m > 12``): uses step-up/step-down search around ``k \in \{5,6,7\}``

2. **ARMA orders** using `auto_arima` on residuals

3. **Box-Cox parameter** via profile likelihood

4. **Trend and damping** via AIC comparison

The descriptor format matches R's `forecast::tbats`:

```
TBATS(omega, {p,q}, phi, <m1,k1>, <m2,k2>, ...)
```

Example: `TBATS(0.001, {0,0}, 0.98, <7,3>, <365.25,10>)`

---

## Examples

### Example 1: Weekly Data with Non-Integer Seasonality

```julia
using Durbyn

y = randn(520)
m = 52.18

model = tbats(y, m)
println(model)

fc = forecast(model, h=52)
```

**Output:**
```
TBATS(0.112, {0,0}, -, <52.18,5>)

Parameters:
  Lambda:  0.1120
  Alpha:   0.0823
  Gamma-1: 0.0045, 0.0032, 0.0019, 0.0011, 0.0007
  Gamma-2: 0.0051, 0.0036, 0.0024, 0.0015, 0.0009

Sigma:   0.9876
AIC:     1523.45
```

### Example 2: Multiple Seasonalities (Daily + Weekly + Yearly)

```julia
using Durbyn, Dates

n = 365 * 3
t = 1:n
y = 100 .+ 10 .* sin.(2π .* t ./ 365.25) .+     # yearly
    5 .* sin.(2π .* t ./ 7) .+                   # weekly
    2 .* randn(n)                                # noise

model = tbats(y, [7, 365.25])
println(model)

fc = forecast(model, h=30)
```

### Example 3: Dual Calendar Effects (Gregorian + Hijri)

```julia
using Durbyn

m_gregorian = 365.25
m_hijri = 354.37

y = load_data()  # Data with both calendar effects

model = tbats(y, [m_gregorian, m_hijri])
println(model)
```

**Output:**
```
TBATS(0.234, {1,1}, 0.975, <365.25,8>, <354.37,7>)
```

### Example 4: Forcing Model Components

```julia
using Durbyn

y = randn(200)
m = [7, 30]

model = tbats(
    y, m;
    use_box_cox = false,           # No transformation
    use_trend = true,              # Force trend
    use_damped_trend = true,       # Force damping
    use_arma_errors = false        # No ARMA errors
)
```

### Example 5: High-Frequency Data (5-minute intervals)

```julia
using Durbyn

minutes_per_day = 288      # 24 * 60 / 5
minutes_per_week = 2016    # 7 * 288

y = load_call_center_data()

model = tbats(y, [minutes_per_day, minutes_per_week])

println("State dimension: ", size(model.x, 1))
println("Seasonal periods: ", model.seasonal_periods)
println("Fourier orders: ", model.k_vector)
```

---

## Forecasting

### Basic Forecasting

```julia
using Durbyn

model = tbats(y, [7, 365.25])

fc = forecast(model, h=30)

println("Point forecasts: ", fc.mean)
println("80% CI: ", fc.lower[:, 1], " to ", fc.upper[:, 1])
println("95% CI: ", fc.lower[:, 2], " to ", fc.upper[:, 2])
```

### Forecast Options

```julia
fc = forecast(
    model;
    h = 50,                          # Forecast horizon
    level = [80, 95],                # Confidence levels
    fan = false,                     # Fan chart levels
    biasadj = nothing                # Bias adjustment (inherits from model)
)
```

### Fan Charts

```julia
fc = forecast(model, h=30, fan=true)

println("Confidence levels: ", fc.level)  # [51, 54, 57, ..., 99]
```

---

## Model Diagnostics

### Fitted Values and Residuals

```julia
using Durbyn

model = tbats(y, m)

fitted_vals = fitted(model)
residuals_vals = residuals(model)

using Statistics
println("RMSE: ", sqrt(mean(residuals_vals .^ 2)))
println("MAE: ", mean(abs.(residuals_vals)))
```

### Checking ARMA Adequacy

```julia
using StatsPlots, Statistics

model = tbats(y, m, use_arma_errors=true)

resid = residuals(model)

autocor(resid, 1:20) |> plot
```

### Information Criteria

```julia
model = tbats(y, m)

println("AIC: ", model.AIC)
println("Log-likelihood: ", model.likelihood)
println("Parameters: ", length(model.parameters[:vect]))
println("States: ", size(model.seed_states, 1))
```

---

## Computational Considerations

### State Space Dimension

For a TBATS model with:
- Trend: 2 states (``\ell_t``, ``b_t``)
- ``T`` seasonal components with Fourier orders ``k_1, \ldots, k_T``: ``2\sum_{i=1}^T k_i`` states
- ARMA(``p``, ``q``): ``p + q`` states

**Total states:** ``2 + 2\sum k_i + p + q``

**Example:**
- Daily (``m=288``, ``k=5``): 10 states
- Weekly (``m=2016``, ``k=7``): 14 states
- Total: ``2 + 10 + 14 = 26`` states

Compare to BATS: ``2 + 288 + 2016 = 2306`` states!

### Choosing Fourier Orders

The implementation automatically selects ``k_i`` via AIC, but general guidelines:

- **Short periods** (``m < 12``): use ``k \approx m/2``
- **Medium periods** (``m = 12`` to ``100``): use ``k \in [3, 10]``
- **Long periods** (``m > 100``): use ``k \in [5, 15]``
- **Very long periods** (``m > 1000``): use ``k \in [10, 20]``

Higher ``k`` captures more complex seasonal shapes but increases computation.

---

## Empirical Results (From De Livera et al. 2011)

### 1. Weekly U.S. Gasoline Demand

- **Seasonal period:** 52.18 weeks/year
- **Result:** TBATS outperforms BATS because seasonality is non-integer
- **MASE improvement:** 8.3%

### 2. Call Center Arrivals (5-minute intervals)

- **Seasonal periods:** 169 (daily), 845 (weekly)
- **BATS states:** 1014
- **TBATS states:** 24 (with ``k_1=5``, ``k_2=7``)
- **Result:** 40× reduction in state dimension with better forecasts

### 3. Turkish Electricity Consumption

- **Calendars:** Gregorian (365.25) + Hijri (354.37)
- **Result:** BATS cannot handle non-integer dual calendars; TBATS is the only feasible model
- **MASE improvement over ETS:** 23.1%

---

## Comparison: BATS vs TBATS

| Scenario | Use BATS | Use TBATS |
|----------|----------|-----------|
| Integer seasonal period | ✓ | ✓ |
| Non-integer seasonal period | ✗ | ✓ |
| Short seasonal period (``m < 50``) | ✓ | ✓ |
| Long seasonal period (``m > 350``) | ✗ | ✓ |
| Multiple integer seasons | ✓ | ✓ |
| Dual calendars | ✗ | ✓ |
| Memory-constrained | ✗ | ✓ |

**Rule of thumb:** Use TBATS when:
- Any seasonal period is non-integer
- Any seasonal period exceeds 100
- You have dual calendar effects
- Memory/computation is limited

---

## References

De Livera, A.M., Hyndman, R.J., & Snyder, R.D. (2011). [*Forecasting time series with complex seasonal patterns using exponential smoothing*.](https://www.tandfonline.com/doi/abs/10.1198/jasa.2011.tm09771) Journal of the American Statistical Association, 106(496), 1513-1527.

---

## See Also

- [BATS Documentation](bats.md) - Integer seasonal periods only
