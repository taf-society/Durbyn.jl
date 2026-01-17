# ARAR Model

!!! tip "Formula Interface is the Recommended Approach"
    This page starts with the **formula interface** (recommended for most users),
    which provides declarative model specification with support for panel data
    and model comparison. The array interface (base models) is covered later.
    See the **[Grammar Guide](grammar.md)** for complete documentation.

---

## Overview

The **ARAR** (AutoRegressive with Adaptive Reduction) model combines memory-shortening transformations with autoregressive modeling. It first reduces long memory in the input series via iterative filtering, then fits an AR model with adaptively selected lags to the shortened series.

**When to use ARAR:**
- Short or nonstationary time series
- When conventional ARIMA methods are unstable or overfit
- When you want a pure AR model without MA components
- Fast, robust forecasting with automatic lag selection

**Reference:** Brockwell, P.J., & Davis, R.A. (2016). *Introduction to Time Series and Forecasting*. Springer.

---

## Formula Interface

### Basic Example

```julia
using Durbyn

series = air_passengers()
data = (sales = series,)

# Using ArarSpec for fit/forecast workflow
spec = ArarSpec(@formula(sales = arar()))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)
plot(fc)
```

### Custom Parameters

```julia
# Specify max_ar_depth and max_lag
spec = ArarSpec(@formula(sales = arar(max_ar_depth=20, max_lag=30)))
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
spec = ArarSpec(@formula(sales = arar()))
group_fit = fit(spec, panel)

# Forecast all groups
group_fc = forecast(group_fit, h = 6)
plot(group_fc)
```

### Model Collections (Benchmarking)

`ArarSpec` slots into model collections for easy benchmarking against other forecasting methods:

```julia
# Compare ARAR against ARIMA and ETS
models = model(
    ArarSpec(@formula(sales = arar())),
    ArimaSpec(@formula(sales = p() + q() + P() + Q())),
    EtsSpec(@formula(sales = e("Z") + t("Z") + s("Z"))),
    names = ["arar", "arima", "ets"]
)

# Fit all models
fitted_models = fit(models, panel)

# Forecast with all models
fc_models = forecast(fitted_models, h = 12)

# Compare forecasts
plot(fc_models)
```

---

## Model Theory

The ARAR model applies a memory-shortening transformation; if the underlying process of a time series ``\{Y_t,\ t=1,2,\ldots,n\}`` is "long-memory", it then fits an autoregressive model.

### Memory Shortening

The model follows five steps to classify ``Y_t`` and take one of three actions:

- **L:** declare ``Y_t`` as long memory and form ``\tilde Y_t = Y_t - \hat\phi\, Y_{t-\hat\tau}``
- **M:** declare ``Y_t`` as moderately long memory and form ``\tilde Y_t = Y_t - \hat\phi_1 Y_{t-1} - \hat\phi_2 Y_{t-2}``
- **S:** declare ``Y_t`` as short memory.

If ``Y_t`` is declared **L** or **M**, the series is transformed again until the transformed series is classified as short memory. (At most three transformations are applied; in practice, more than two is rare.)

### Algorithm Steps

1. For each ``\tau=1,2,\ldots,15``, find ``\hat\phi(\tau)`` that minimizes
   ```math
   \mathrm{ERR}(\phi,\tau) \;=\;
   \frac{\displaystyle\sum_{t=\tau+1}^{n}\!\big(Y_t-\phi\,Y_{t-\tau}\big)^2}
        {\displaystyle\sum_{t=\tau+1}^{n}\!Y_t^{\,2}},
   ```
   then set ``\mathrm{Err}(\tau)=\mathrm{ERR}\big(\hat\phi(\tau),\tau\big)`` and choose
   ``\hat\tau=\arg\min_{\tau}\mathrm{Err}(\tau)``.
2. If ``\mathrm{Err}(\hat\tau)\le 8/n``, then ``Y_t`` is a long-memory series.
3. If ``\hat\phi(\hat\tau)\ge 0.93`` and ``\hat\tau>2``, then ``Y_t`` is a long-memory series.
4. If ``\hat\phi(\hat\tau)\ge 0.93`` and ``\hat\tau\in\{1,2\}``, then ``Y_t`` is a long-memory series.
5. If ``\hat\phi(\hat\tau)<0.93``, then ``Y_t`` is a short-memory series.

### Subset Autoregressive Model

We now describe how ARAR fits an autoregression to the mean-corrected series
``X_t=S_t-\bar S``, ``t=k+1,\ldots,n``, where ``\{S_t\}`` is the memory-shortened version of ``\{Y_t\}`` obtained above and ``\bar S`` is the sample mean of ``S_{k+1},\ldots,S_n``.

The fitted model has the form
```math
X_t \;=\; \phi_1 X_{t-1} \;+\; \phi_{l_1} X_{t-l_1} \;+\; \phi_{l_2} X_{t-l_2} \;+\; \phi_{l_3} X_{t-l_3} \;+\; Z_t,
\qquad Z_t \sim \mathrm{WN}(0,\sigma^2).
```

### Yule-Walker Equations

The coefficients ``\phi_j`` and the noise variance ``\sigma^2`` follow from the Yule-Walker equations for given lags ``l_1,l_2,l_3``:

```math
\begin{bmatrix}
1 & \hat\rho(l_1-1) & \hat\rho(l_2-1) & \hat\rho(l_3-1)\\
\hat\rho(l_1-1) & 1 & \hat\rho(l_2-l_1) & \hat\rho(l_3-l_1)\\
\hat\rho(l_2-1) & \hat\rho(l_2-l_1) & 1 & \hat\rho(l_3-l_2)\\
\hat\rho(l_3-1) & \hat\rho(l_3-l_1) & \hat\rho(l_3-l_2) & 1
\end{bmatrix}
\begin{bmatrix}
\phi_1\\[2pt]
\phi_{l_1}\\[2pt]
\phi_{l_2}\\[2pt]
\phi_{l_3}
\end{bmatrix}
=
\begin{bmatrix}
\hat\rho(1)\\[2pt]
\hat\rho(l_1)\\[2pt]
\hat\rho(l_2)\\[2pt]
\hat\rho(l_3)
\end{bmatrix}.
```

```math
\sigma^2 \;=\; \hat\gamma(0)\,\Big( 1 - \phi_1\hat\rho(1) - \phi_{l_1}\hat\rho(l_1) - \phi_{l_2}\hat\rho(l_2) - \phi_{l_3}\hat\rho(l_3) \Big),
```
where ``\hat\gamma(j)`` and ``\hat\rho(j)``, ``j=0,1,2,\ldots``, are the sample autocovariances and autocorrelations of ``X_t``.
The algorithm computes ``\phi(\cdot)`` for each set of lags with ``1<l_1<l_2<l_3\le m`` (``m`` typically 13 or 26) and selects the model with minimal Yule-Walker estimate of ``\sigma^2``.

### Forecasting

If the short-memory filter found in the first step has coefficients ``\Psi_0,\Psi_1,\ldots,\Psi_k`` (``k\ge0``, ``\Psi_0=1``), then
```math
S_t \;=\; \Psi(B)Y_t \;=\; Y_t + \Psi_1 Y_{t-1} + \cdots + \Psi_k Y_{t-k},
\qquad
\Psi(B) \;=\; 1 + \Psi_1 B + \cdots + \Psi_k B^k .
```

If the subset AR coefficients are ``\phi_1,\phi_{l_1},\phi_{l_2},\phi_{l_3}`` then, for ``X_t=S_t-\bar S``,
```math
\phi(B)X_t \;=\; Z_t, \qquad
\phi(B) \;=\; 1 - \phi_1 B - \phi_{l_1} B^{l_1} - \phi_{l_2} B^{l_2} - \phi_{l_3} B^{l_3}.
```

From the two displays above,
```math
\xi(B)Y_t \;=\; \phi(1)\,\bar S \;+\; Z_t,
\qquad \xi(B) \;=\; \Psi(B)\,\phi(B).
```

Assuming this model is appropriate and ``Z_t`` is uncorrelated with ``Y_j`` for ``j<t``, the minimum-MSE linear predictors ``P_n Y_{n+h}`` of ``Y_{n+h}`` (for ``n>k+l_3``) satisfy the recursion
```math
P_n Y_{n+h} \;=\; - \sum_{j=1}^{k+l_3} \xi_j \, P_n Y_{n+h-j} \;+\; \phi(1)\,\bar S, \qquad h\ge 1,
```
with initial conditions ``P_n Y_{n+h}=Y_{n+h}`` for ``h\le 0``.

---

## Array Interface (Base Model)

The array interface provides direct access to the ARAR fitting engine for working with numeric vectors.

```julia
using Durbyn
using Durbyn.Ararma

ap = air_passengers()

# Basic ARAR with default parameters
arar_fit = arar(ap)
fc = forecast(arar_fit, h = 12)
plot(fc)

# ARAR with custom parameters
arar_fit = arar(ap, max_ar_depth = 20, max_lag = 30)
fc = forecast(arar_fit, h = 12)
plot(fc)

# Access model components
println(arar_fit)           # Model summary
fitted_vals = fitted(arar_fit)
resid = residuals(arar_fit)
```

### Function Signature

```julia
arar(y::AbstractArray;
     max_ar_depth::Union{Int, Nothing}=nothing,
     max_lag::Union{Int, Nothing}=nothing) -> ARAR
```

**Arguments:**
- `y`: A one-dimensional array containing the observed time series data
- `max_ar_depth`: Maximum lag to consider when selecting the best 4-lag AR model (default: auto-selected based on series length)
- `max_lag`: Maximum lag for computing autocovariances (default: auto-selected based on series length)

**Returns:** An `ARAR` struct containing fitted model components

---

## Comparison with ARARMA

| Feature | ARAR | ARARMA |
|---------|------|--------|
| Reference | Brockwell & Davis (2016) | Parzen (1982) |
| Memory shortening | Yes (threshold 0.93) | Yes (threshold 0.9) |
| AR component | Subset AR(4) via Yule-Walker | Subset AR(4) via Yule-Walker |
| MA component | No | Yes, ARMA(p,q) on residuals |
| Use case | Simple, robust forecasting | Captures MA structure in residuals |

See **[ARARMA](ararma.md)** for the extended model with ARMA fitting.

---

## Reference

- Brockwell, Peter J., and Richard A. Davis. *Introduction to Time Series and Forecasting*. [Springer](https://link.springer.com/book/10.1007/978-3-319-29854-2) (2016)
