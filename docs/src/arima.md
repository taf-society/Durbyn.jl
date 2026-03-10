
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
Reference: [BJL2015, Ch. 3], [Hamilton1994, Ch. 3].

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
Reference: [BJL2015, Ch. 9], [HK2008].

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
Reference: [HK2008], [BJL2015, Ch. 9].

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
Reference: [HK2008], [BJL2015, Ch. 9].

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

### Algorithm (similar in spirit to Hyndman & Khandakar, 2008)
1. **Stationarity / seasonality diagnostics** determine differencing orders ``d`` and ``D``.
2. **Initial model selection** uses clipped starting orders and simple baseline models.
3. **Stepwise search** (or exhaustive search when `stepwise=false`) explores ``(p, q, P, Q)`` within configurable bounds.
4. Models are ranked by the chosen information criterion (`AICc` by default; `AIC` and `BIC` also supported).
5. If approximation mode is used, the best candidate is refit without approximation before returning.

### Advantages
- Removes the manual effort of model identification.  
- Scales well to large numbers of series.  
- Ensures differencing is tested systematically (avoids over-differencing).

### Limitations
- Stepwise search may not find the global optimum.
- Computationally expensive for very large seasonal periods.
- Still requires diagnostic checking of residuals.

### Detailed Algorithm

Reference: Hyndman & Khandakar (2008); the implementation below describes Durbyn's current `auto_arima`.

#### Phase 1: Differencing Order Selection

**Stationary shortcut:** If `stationary=true`, the implementation sets ``d = 0`` and ``D = 0`` immediately.

**Seasonal differencing (``D``):**

- If ``m = 1``, then ``D = 0`` and seasonal AR/MA orders are disabled.
- If ``D`` is not supplied and the trimmed series has length ``\le 2m``, then ``D = 0``.
- Otherwise `nsdiffs` is used with `max_D=1` by default.
- Durbyn defaults to `seasonal_test = :seas`, which uses Hyndman's seasonal-strength heuristic.
- `seasonal_test = :ocsb` is also supported.
- Canova-Hansen is referenced in the literature, but it is **not** currently implemented in Durbyn's `nsdiffs`.
- If seasonal differencing would collapse the differenced series or make differenced regressors constant, ``D`` is reduced by one.

**Non-seasonal differencing (``d``):**

- If ``d`` is not supplied, `ndiffs` is applied to the seasonally differenced series.
- Durbyn defaults to `test = :kpss`; `:adf` and `:pp` are also supported.
- The default cap is ``d \le 2``.
- If additional differencing would collapse the differenced series or make differenced regressors constant, ``d`` is reduced by one.

#### Phase 2: Initial Stepwise Fits

With ``d`` and ``D`` fixed, the default stepwise search starts from clipped versions of
`start_p=2`, `start_q=2`, `start_P=1`, and `start_Q=1`. For very short series
(`length(x) < 10`), starts are reduced to ``p,q \le 1`` and ``P = Q = 0``.

The initial candidates are:

| # | Candidate |
|---|---|
| 1 | ARIMA(start_p,d,start_q)(start_P,D,start_Q)``_m`` |
| 2 | ARIMA(0,d,0)(0,D,0)``_m`` |
| 3 | ARIMA(pp,d,0)(PP,D,0)``_m`` where ``pp \in \{0,1\}``, ``PP \in \{0,1\}`` subject to bounds |
| 4 | ARIMA(0,d,qq)(0,D,QQ)``_m`` where ``qq \in \{0,1\}``, ``QQ \in \{0,1\}`` subject to bounds |
| 5 | If a mean/drift term is allowed, ARIMA(0,d,0)(0,D,0)``_m`` without it |

**Constant handling:**

- If ``d + D = 0``, a non-zero mean may be included when `allowmean=true`.
- If ``d + D = 1``, a drift term may be included when `allowdrift=true`.
- If ``d + D \ge 2``, no constant term is searched.

The candidate with the smallest selected information criterion becomes the **current model**.

#### Phase 3: Stepwise Neighborhood Search

From the current model ARIMA(p,d,q)(P,D,Q)``_m``, the stepwise loop evaluates up to **17 local moves**
(subject to bounds and duplicate filtering):

1. ``P - 1``
2. ``Q - 1``
3. ``P + 1``
4. ``Q + 1``
5. ``(P - 1, Q - 1)``
6. ``(P - 1, Q + 1)``
7. ``(P + 1, Q - 1)``
8. ``(P + 1, Q + 1)``
9. ``p - 1``
10. ``q - 1``
11. ``p + 1``
12. ``q + 1``
13. ``(p - 1, q - 1)``
14. ``(p - 1, q + 1)``
15. ``(p + 1, q - 1)``
16. ``(p + 1, q + 1)``
17. Toggle the mean/drift term (if allowed)

If any move improves the selected criterion, it becomes the new current model and the loop restarts.
If a full pass finds no improvement, the search stops.

#### Search Constraints and Scoring

- Defaults: ``max_p = 5``, ``max_q = 5``, ``max_P = 2``, ``max_Q = 2``
- When ``m > 1`` and seasonal terms are allowed, ``p`` and ``q`` are additionally clipped to ``m - 1``
- If `stepwise=false`, Durbyn performs an exhaustive search over models satisfying ``p + q + P + Q \le max_order``
- Models that fail during estimation are discarded
- Models are also discarded if any AR or MA root has modulus ``< 1.01`` (near-unit-root guard)
- Default ranking criterion is `AICc`; `AIC` and `BIC` are also supported
- When approximation mode is enabled, the search uses CSS-style approximations and then refits the best admissible model without approximation

#### Information Criteria

If ``k`` is the number of free mean/ARIMA/xreg parameters and ``\ell`` is the maximized log-likelihood,
define ``m = k + 1`` to include ``\sigma^2``:

```math
\text{AIC} = -2\ell + 2m,
\qquad
\text{BIC} = -2\ell + m \log n,
\qquad
\text{AICc} = \text{AIC} + \frac{2m(m+1)}{n-m-1}.
```

Information-criterion values are only comparable across models with the same ``d`` and ``D``.

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

## 7. Implementation Mathematics (Core Algorithm)

This section documents the equations used by Durbyn's ARIMA core implementation.
Citation keys used in this section: `[BJL2015]`, `[HK2008]`, `[Hamilton1994]`, `[Jones1980]`, `[Monahan1984]`, `[Harvey1989]`, `[DK2012]`, `[Akaike1974]`, `[Schwarz1978]`, `[HurvichTsai1989]`, `[BD2016]`.

### 7.1 Multiplicative Seasonal ARIMA
References: [BJL2015, Ch. 9], [HK2008].

```math
\phi(B)\,\Phi(B^s)\,(1-B)^d(1-B^s)^D\,y_t
=
\theta(B)\,\Theta(B^s)\,\varepsilon_t
```

with:

```math
\phi(B)=1-\phi_1B-\cdots-\phi_pB^p,\quad
\Phi(B^s)=1-\Phi_1B^s-\cdots-\Phi_PB^{Ps}
```
```math
\theta(B)=1+\theta_1B+\cdots+\theta_qB^q,\quad
\Theta(B^s)=1+\Theta_1B^s+\cdots+\Theta_QB^{Qs}.
```

### 7.2 Polynomial Convolution
References: [BJL2015, Ch. 9], [Hamilton1994, Ch. 3].

For \(a(z)=\sum_i a_i z^i\), \(b(z)=\sum_j b_j z^j\):

```math
c(z)=a(z)b(z),\qquad
c_k=\sum_i a_i\,b_{k-i}.
```

### 7.3 Differencing Polynomial
References: [BJL2015, Ch. 9], [HK2008].

Define:

```math
\Delta(B)=(1-B)^d(1-B^s)^D.
```

If \(\Delta(B)=1+\sum_{j=1}^{m}\delta_j B^j\), the implementation stores:

```math
\texttt{Delta} = -[\delta_1,\dots,\delta_m].
```

### 7.4 Full AR/MA Expansion
References: [BJL2015, Ch. 9], [Hamilton1994, Ch. 3].

Non-seasonal + seasonal AR expansion:

```math
\phi(B)\Phi(B^s)=1-\sum_{j=1}^{p^*}\varphi_j B^j.
```

Non-seasonal + seasonal MA expansion:

```math
\theta(B)\Theta(B^s)=1+\sum_{j=1}^{q^*}\vartheta_j B^j.
```

These are obtained by polynomial convolution of:

```math
[1,-\phi_1,\dots,-\phi_p]\otimes[1,0,\dots,-\Phi_1,\dots]
```
```math
[1,\theta_1,\dots,\theta_q]\otimes[1,0,\dots,\Theta_1,\dots].
```

### 7.5 Stationarity and Invertibility
References: [Hamilton1994, Sec. 3.2], [BJL2015, Ch. 3].

AR stationarity is checked via roots of:

```math
1-\varphi_1 z-\cdots-\varphi_{p^*} z^{p^*}=0,
```

requiring:

```math
|z_k|>1\ \forall k.
```

MA invertibility is enforced by reflecting roots of:

```math
1+\vartheta_1 z+\cdots+\vartheta_{q^*} z^{q^*}=0
```

inside the unit circle:

```math
|r_k|<1\ \Rightarrow\ r_k \leftarrow \frac{1}{r_k},
```

then reconstructing the polynomial coefficients.

### 7.6 Jones / Monahan AR Transform
References: [Jones1980], [Monahan1984].

For unconstrained optimizer parameters \(u_j\), partial autocorrelations are:

```math
w_j=\tanh(u_j).
```

Durbin-Levinson recursion builds stationary AR coefficients:

```math
\phi_j^{(j)}=w_j,\quad
\phi_i^{(j)}=\phi_i^{(j-1)}-w_j\phi_{j-i}^{(j-1)},\ i=1,\dots,j-1.
```

The inverse map uses reverse recursion and:

```math
u_j=\operatorname{atanh}(w_j).
```

### 7.7 Initial Stationary Covariance (Lyapunov)
References: [Harvey1989, Ch. 3], [DK2012, Ch. 5].

For ARMA state block:

```math
P = TPT^\top + RR^\top.
```

Durbyn solves this discrete Lyapunov equation with:
1. Smith doubling for small state dimension.
2. Kronecker system for larger dimension:

```math
\mathrm{vec}(P) =
\left(I - T\otimes T\right)^{-1}\mathrm{vec}(RR^\top).
```

### 7.8 State-Space Form
References: [Harvey1989, Ch. 3], [DK2012, Ch. 2].

State dimension:

```math
r=\max(p^*,q^*+1),\qquad rd=r+n_{\Delta}.
```

Observation vector:

```math
Z = [1, 0,\dots,0,\Delta_1,\dots,\Delta_{n_\Delta}]^\top.
```

Transition (companion + differencing block):

```math
\alpha_{t+1}=T\alpha_t+R\varepsilon_t,\qquad
y_t=Z^\top\alpha_t.
```

Process covariance:

```math
V=RR^\top,\quad
R=[1,\vartheta_1,\dots,\vartheta_{r-1},0,\dots,0]^\top.
```

Diffuse initialization for integrated states uses large prior variance \(\kappa\) on differencing-state diagonals.

### 7.9 Kalman Filter Recursions
References: [DK2012, Ch. 4], [Harvey1989, Ch. 3].

Prediction:

```math
a_{t|t-1}=Ta_{t-1|t-1},\qquad
P_{t|t-1}=TP_{t-1|t-1}T^\top+V.
```

Innovation:

```math
v_t = y_t - Z^\top a_{t|t-1},\qquad
F_t = Z^\top P_{t|t-1} Z.
```

Update:

```math
a_{t|t}=a_{t|t-1}+P_{t|t-1}Z\,\frac{v_t}{F_t},
```
```math
P_{t|t}=P_{t|t-1}-\frac{(P_{t|t-1}Z)(P_{t|t-1}Z)^\top}{F_t}.
```

Standardized residual:

```math
\tilde e_t = \frac{v_t}{\sqrt{F_t}}.
```

### 7.10 Concentrated Gaussian Log-Likelihood
References: [DK2012, Ch. 7], [Harvey1989, Ch. 3].

For non-diffuse observations:

```math
\sigma^2(\theta)=\frac{1}{n}\sum_t \frac{v_t^2}{F_t}.
```

Objective minimized:

```math
f(\theta)=\frac12\left[\log \sigma^2(\theta) + \frac1n\sum_t\log F_t\right].
```

Recovered log-likelihood:

```math
-2\ell = 2n f(\hat\theta) + n + n\log(2\pi),\qquad
\ell = -\frac12\left(2n f(\hat\theta)+n+n\log(2\pi)\right).
```

### 7.11 CSS Objective
References: [BJL2015, Ch. 7], [HK2008].

On differenced series \(w_t\), conditional residual recursion:

```math
e_t = w_t - \sum_j \varphi_j w_{t-j} - \sum_j \vartheta_j e_{t-j}.
```

CSS variance and objective:

```math
\sigma^2_{\text{CSS}} = \frac{1}{n_\text{eff}}\sum_t e_t^2,\qquad
f_{\text{CSS}}(\theta)=\frac12\log \sigma^2_{\text{CSS}}.
```

### 7.12 Information Criteria
References: [Akaike1974], [Schwarz1978], [HurvichTsai1989], [BD2016, Sec. 5.5.2].

If \(k\) is number of free mean/ARIMA/xreg parameters and \(\ell\) the maximized log-likelihood, define \(m=k+1\) (including \(\sigma^2\)):

```math
\text{AIC}=-2\ell+2m,
```
```math
\text{BIC}=-2\ell+m\log n,
```
```math
\text{AICc}=\text{AIC}+\frac{2m(m+1)}{n-m-1}.
```

### 7.13 Forecasting
References: [DK2012, Ch. 4], [HK2008].

h-step point forecast:

```math
\hat y_{T+h|T}=Z^\top a_{T+h|T}+x_{T+h}^\top\hat\beta.
```

Forecast covariance recursion:

```math
a_{t+1|T}=Ta_{t|T},\qquad
P_{t+1|T}=TP_{t|T}T^\top+V.
```

Forecast variance:

```math
\operatorname{Var}(\hat y_{T+h|T}) = \hat\sigma^2\,(Z^\top P_{T+h|T} Z).
```

Prediction interval at level \(1-\alpha\):

```math
\hat y_{T+h|T}\pm z_{1-\alpha/2}\sqrt{\operatorname{Var}(\hat y_{T+h|T})}.
```

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
This is useful for quick analyses or integration with existing code.

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
- [Akaike1974] Akaike, H. (1974). *A new look at the statistical model identification*. IEEE Transactions on Automatic Control, 19(6), 716-723.
- [BD2016] Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.). Springer.
- [BJL2015] Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- [DK2012] Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.
- [Hamilton1994] Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- [Harvey1989] Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- [HK2008] Hyndman, R. J., & Khandakar, Y. (2008). *Automatic time series forecasting: the forecast package for R*. Journal of Statistical Software, 27(3), 1-22.
- [HurvichTsai1989] Hurvich, C. M., & Tsai, C.-L. (1989). *Regression and time series model selection in small samples*. Biometrika, 76(2), 297-307.
- [Jones1980] Jones, R. H. (1980). *Maximum likelihood fitting of ARMA models to time series with missing observations*. Technometrics, 22(3), 389-395.
- [Kunst2011] Kunst, R. (2011). *Applied Time Series Analysis — Part II*. University of Vienna.
- [Monahan1984] Monahan, J. F. (1984). *A note on enforcing stationarity in autoregressive-moving average models*. Biometrika, 71(2), 403-404.
- [Schwarz1978] Schwarz, G. (1978). *Estimating the dimension of a model*. Annals of Statistics, 6(2), 461-464.
