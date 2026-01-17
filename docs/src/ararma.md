# ARARMA Model

!!! tip "Formula Interface is the Recommended Approach"
    This page starts with the **formula interface** (recommended for most users),
    which provides declarative model specification with support for panel data
    and model comparison. The array interface (base models) is covered later.
    See the **[Grammar Guide](grammar.md)** for complete documentation.

---

## Overview

**ARARMA** extends the ARAR approach by first applying an adaptive AR prefilter to shorten memory (the *ARAR* stage), and then fitting a short-memory **ARMA(p, q)** model on the prefiltered residuals. The goal is to capture long/persistent structure via a composed AR filter and the remaining short-term dynamics via an ARMA kernel.

**When to use ARARMA:**
- When residuals from AR-only models show MA structure
- For capturing both long-memory and short-term dynamics
- When you need automatic model selection via `auto_ararma`
- For series with complex autocorrelation patterns

**Reference:** Parzen, E. (1982). *ARARMA Models for Time Series Analysis and Forecasting*. Journal of Forecasting, 1(1), 67-82.

---

## Formula Interface

### Basic Example

```julia
using Durbyn

series = air_passengers()
data = (sales = series,)

# Using ArarmaSpec for fit/forecast workflow
spec = ArarmaSpec(@formula(sales = p(1) + q(2)))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)
plot(fc)
```

### Auto ARARMA (Model Selection)

```julia
# Auto ARARMA with default search ranges
spec = ArarmaSpec(@formula(sales = p() + q()))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)

# Auto ARARMA with custom search ranges
spec = ArarmaSpec(@formula(sales = p(0,3) + q(0,2)))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)

# With custom ARAR parameters
spec = ArarmaSpec(
    @formula(sales = p() + q()),
    max_ar_depth = 20,
    max_lag = 30,
    crit = :bic
)
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)
```

### Panel Data (Multiple Series)

```julia
# Create panel data with multiple regions
panel_tbl = (
    sales = vcat(series, series .* 1.05),
    region = vcat(fill("north", length(series)), fill("south", length(series)))
)

# Wrap in PanelData for grouped fitting
panel = PanelData(panel_tbl; groupby = :region, m = 12)

# Fit to all groups
spec = ArarmaSpec(@formula(sales = p(1) + q(1)))
group_fit = fit(spec, panel)

# Forecast all groups
group_fc = forecast(group_fit, h = 6)
plot(group_fc)
```

### Model Collections (Benchmarking)

`ArarmaSpec` slots into model collections for easy benchmarking against other forecasting methods:

```julia
# Compare ARARMA against ARAR, ARIMA, and ETS
models = model(
    ArarmaSpec(@formula(sales = p() + q())),
    ArarSpec(@formula(sales = arar())),
    ArimaSpec(@formula(sales = p() + q() + P() + Q())),
    EtsSpec(@formula(sales = e("Z") + t("Z") + s("Z"))),
    names = ["ararma", "arar", "arima", "ets"]
)

# Fit all models
fitted_models = fit(models, panel)

# Forecast with all models
fc_models = forecast(fitted_models, h = 12)

# Compare forecasts
plot(fc_models)
```

### Automatic vs Fixed Order Selection

**Automatic selection** (uses `auto_ararma`):
- Any order term with a range triggers automatic model selection
- `p()` or `q()` with no arguments use default search ranges
- Examples: `p() + q()`, `p(0,3) + q()`, `p(1) + q(0,2)`

**Fixed orders** (uses `ararma` directly - faster):
- When all orders are fixed, the formula interface calls `ararma()` directly
- Much faster as it skips the search process
- Example: `p(1) + q(2)` fits ARARMA(1,2) directly

---

## Model Theory

Given a univariate series ``\{Y_t,\ t=1,2,\ldots,n\}``, ARARMA produces a fitted model and forecasting mechanism that combine both stages.

### Stage 1 - Memory Shortening (ARAR)

As in ARAR, we iteratively test for long memory and, if detected, apply a memory-shortening AR filter. At iteration r, the procedure evaluates delays ``\tau=1,\ldots,15`` by ordinary least squares and scores each delay by a relative error measure:

```math
\mathrm{ERR}(\phi,\tau)
\;=\;
\frac{\displaystyle\sum_{t=\tau+1}^{n}\!\big(Y_t-\phi\,Y_{t-\tau}\big)^2}
     {\displaystyle\sum_{t=\tau+1}^{n}\!Y_t^{\,2}},
\qquad
\hat\phi(\tau)\in\arg\min_{\phi}\ \mathrm{ERR}(\phi,\tau).
```

Let ``\mathrm{Err}(\tau) = \mathrm{ERR}\!\big(\hat\phi(\tau),\tau\big)`` and ``\hat\tau=\arg\min_\tau \mathrm{Err}(\tau)``. Then:

- If ``\mathrm{Err}(\hat\tau)\le 8/n`` or if ``\hat\phi(\hat\tau)\ge 0.9`` with ``\hat\tau>2``, declare long memory and filter:
  ```math
  \tilde Y_t \;=\; Y_t - \hat\phi\,Y_{t-\hat\tau}.
  ```
- If ``\hat\phi(\hat\tau)\ge 0.9`` with ``\hat\tau\in\{1,2\}``, fit an AR(2) at lags 1 and 2 and filter:
  ```math
  \tilde Y_t \;=\; Y_t - \hat\phi_1 Y_{t-1} - \hat\phi_2 Y_{t-2}.
  ```
- Otherwise, stop.

Each successful reduction composes the prefilter polynomial ``\Psi(B)`` (with ``\Psi_0=1``):
```math
S_t \;=\; \Psi(B)Y_t \;=\; Y_t + \Psi_1 Y_{t-1} + \cdots + \Psi_k Y_{t-k},
\qquad
\Psi(B) \;=\; 1 + \Psi_1 B + \cdots + \Psi_k B^k.
```
The reduction loop terminates when short memory is reached or after three passes (rarely more than two are needed in practice).

### Stage 2 - Best-Lag Subset AR

Let ``X_t = S_t - \bar S`` with ``\bar S`` the sample mean of the final prefiltered series. Over 4-term lag tuples ``(1,i,j,k)`` satisfying ``1<i<j<k\le m`` (with ``m`` typically 13 or 26), we fit the subset AR:
```math
X_t \;=\; \phi_1 X_{t-1} \;+\; \phi_{i} X_{t-i} \;+\; \phi_{j} X_{t-j} \;+\; \phi_{k} X_{t-k} \;+\; Z_t,
\qquad Z_t\sim \mathrm{WN}(0,\sigma^2).
```

Yule-Walker equations (using sample autocorrelations ``\hat\rho(\cdot)`` of ``X_t``) yield the coefficients:
```math
\begin{bmatrix}
1 & \hat\rho(i-1) & \hat\rho(j-1) & \hat\rho(k-1)\\
\hat\rho(i-1) & 1 & \hat\rho(j-i) & \hat\rho(k-i)\\
\hat\rho(j-1) & \hat\rho(j-i) & 1 & \hat\rho(k-j)\\
\hat\rho(k-1) & \hat\rho(k-i) & \hat\rho(k-j) & 1
\end{bmatrix}
\!\!
\begin{bmatrix}
\phi_1\\[2pt]\phi_i\\[2pt]\phi_j\\[2pt]\phi_k
\end{bmatrix}
=
\begin{bmatrix}
\hat\rho(1)\\[2pt]\hat\rho(i)\\[2pt]\hat\rho(j)\\[2pt]\hat\rho(k)
\end{bmatrix}.
```

The implied variance is
```math
\sigma^2 \;=\; \hat\gamma(0)\,\Big(1 - \phi_1 \hat\rho(1) - \phi_i \hat\rho(i) - \phi_j \hat\rho(j) - \phi_k \hat\rho(k)\Big),
```
where ``\hat\gamma(\cdot)`` are sample autocovariances of ``X_t``. The algorithm selects ``(1,i,j,k)`` minimizing this ``\sigma^2``.

Define the **composite AR kernel** by convolving the prefilter with the selected subset AR:
```math
\phi(B) \;=\; 1 - \phi_1 B - \phi_i B^{i} - \phi_j B^{j} - \phi_k B^{k},
\qquad
\xi(B) \;=\; \Psi(B)\,\phi(B).
```
Let ``c = \big(1-\phi_1-\phi_i-\phi_j-\phi_k\big)\,\bar S`` be the AR intercept.

### Stage 3 - Short-Memory ARMA(p, q) on AR Residuals

Using the AR-only fit implied by ``\xi(B)`` and ``c``, compute residuals and fit an **ARMA(p, q)** by maximizing the conditional Gaussian likelihood. Denote the ARMA polynomials
```math
\Phi(B) \;=\; 1 - \varphi_1 B - \cdots - \varphi_p B^{p},
\qquad
\Theta(B) \;=\; 1 + \theta_1 B + \cdots + \theta_q B^{q}.
```
The ARMA stage estimates ``(\varphi_1,\ldots,\varphi_p,\,\theta_1,\ldots,\theta_q,\,\sigma^2)`` via Nelder-Mead. The code log-parameterizes the variance for numerical stability.

The ARMA optimizer uses log-barrier penalties to enforce stability (``\sum|\varphi| < 0.95``, ``\sum|\theta| < 0.95``).

**Information criteria.** With effective residual length ``n_{\text{eff}}`` and ``k=p+q+1`` parameters (including variance), the log-likelihood ``\ell`` yields
```math
\mathrm{AIC}=2k-2\ell, \qquad \mathrm{BIC}=(\log n_{\text{eff}})\,k-2\ell.
```

### Forecasting

With the composite kernel ``\xi(B)`` and intercept ``c`` from Stage 2, and the ARMA(p,q) layer from Stage 3, h-step-ahead forecasts are formed recursively. Let ``P_n Y_{n+h}`` denote the minimum-MSE predictor of ``Y_{n+h}``. Writing ``\xi(B)=1+\xi_1 B+\cdots+\xi_{K} B^{K}``,
```math
P_n Y_{n+h}
\;=\;
- \sum_{j=1}^{K} \xi_j \, P_n Y_{n+h-j}
\;+\; c \;+\; \text{(MA terms from } \Theta(B)\text{)},
\qquad h\ge 1,
```
with initialization ``P_n Y_{n+h}=Y_{n+h}`` for ``h\le 0`` and future shocks set to zero for the MA recursion. Forecast standard errors follow from the MA representation and the estimated innovation variance ``\sigma^2``.

---

## Array Interface (Base Model)

```julia
using Durbyn
using Durbyn.Ararma

ap = air_passengers()

# ARARMA with specified orders
fit1 = ararma(ap, p = 4, q = 1)
fc1 = forecast(fit1, h = 12)
plot(fc1)

# Automatic ARARMA order selection
fit2 = auto_ararma(ap)
fc2 = forecast(fit2, h = 12)
plot(fc2)

# Access model components
println(fit1)               # Model summary
fitted_vals = fitted(fit1)
resid = residuals(fit1)
```

### Function Signatures

```julia
ararma(y::Vector{<:Real};
       max_ar_depth::Int=26,
       max_lag::Int=40,
       p::Int=4,
       q::Int=1,
       options::NelderMeadOptions=NelderMeadOptions()) -> ArarmaModel
```

**Arguments:**
- `y`: A numeric vector containing the observed time series
- `max_ar_depth`: Maximum lag depth for subset AR selection (default: 26)
- `max_lag`: Maximum lag for autocovariance computation (default: 40)
- `p`: AR order for ARMA stage (default: 4)
- `q`: MA order for ARMA stage (default: 1)
- `options`: Nelder-Mead optimization options

**Returns:** An `ArarmaModel` struct containing fitted model components

```julia
auto_ararma(y::Vector{<:Real};
            max_p::Int=4,
            max_q::Int=2,
            crit::Symbol=:aic,
            max_ar_depth::Int=26,
            max_lag::Int=40,
            options::NelderMeadOptions=NelderMeadOptions()) -> ArarmaModel
```

**Arguments:**
- `y`: A numeric vector containing the observed time series
- `max_p`: Maximum AR order to search (default: 4)
- `max_q`: Maximum MA order to search (default: 2)
- `crit`: Information criterion for selection, `:aic` or `:bic` (default: `:aic`)

**Returns:** The best `ArarmaModel` according to the selected criterion

---

## Comparison with ARAR

| Feature | ARAR | ARARMA |
|---------|------|--------|
| Reference | Brockwell & Davis (2016) | Parzen (1982) |
| Memory shortening | Yes (threshold 0.93) | Yes (threshold 0.9) |
| AR component | Subset AR(4) via Yule-Walker | Subset AR(4) via Yule-Walker |
| MA component | No | Yes, ARMA(p,q) on residuals |
| Use case | Simple, robust forecasting | Captures MA structure in residuals |

See **[ARAR](arar.md)** for the simpler AR-only model.

---

## Reference

- Parzen, E. (1982). *ARARMA Models for Time Series Analysis and Forecasting*. Journal of Forecasting, 1(1), 67-82.
