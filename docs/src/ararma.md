# ARAR and ARARMA Models

!!! tip "Formula Interface is the Recommended Approach"
    This page starts with the **formula interface** (recommended for most users),
    which provides declarative model specification with support for panel data
    and model comparison. The array interface (base models) is covered later.
    See the **[Grammar Guide](grammar.md)** for complete documentation.

---

## Formula Interface

The ARAR model participates in Durbyn's forecasting grammar, allowing you to build models declaratively with `@formula` and integrate them into `model(...)` collections or grouped workflows.

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

This shared syntax keeps ARAR on equal footing with ARIMA, ETS, and other forecasting families.

---

## ARAR Model Theory

The ARAR model applies a memory-shortening transformation; if the underlying process of a time series ``\{Y_t,\ t=1,2,\ldots,n\}`` is "long-memory", it then fits an autoregressive model.

### Memory Shortening

The model follows five steps to classify ``Y_t`` and take one of three actions:

- **L:** declare ``Y_t`` as long memory and form ``\tilde Y_t = Y_t - \hat\phi\, Y_{t-\hat\tau}``
- **M:** declare ``Y_t`` as moderately long memory and form ``\tilde Y_t = Y_t - \hat\phi_1 Y_{t-1} - \hat\phi_2 Y_{t-2}``
- **S:** declare ``Y_t`` as short memory.

If ``Y_t`` is declared **L** or **M**, the series is transformed again until the transformed series is classified as short memory. (At most three transformations are applied; in practice, more than two is rare.)

### Steps

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

### Yule–Walker Equations

The coefficients ``\phi_j`` and the noise variance ``\sigma^2`` follow from the Yule–Walker equations for given lags ``l_1,l_2,l_3``:

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
The algorithm computes ``\phi(\cdot)`` for each set of lags with ``1<l_1<l_2<l_3\le m`` (``m`` typically 13 or 26) and selects the model with minimal Yule–Walker estimate of ``\sigma^2``.

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

### Reference
- Brockwell, Peter J., and Richard A. Davis. *Introduction to Time Series and Forecasting*. [Springer](https://link.springer.com/book/10.1007/978-3-319-29854-2) (2016)

---

## Array Interface (Base Models)

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
```

---

## ARARMA Model Theory

**ARARMA** extends the ARAR approach by first applying an adaptive **AR** prefilter to shorten memory (the *ARAR* stage), and then fitting a short-memory **ARMA(p, q)** model on the prefiltered residuals. The goal is to capture long/persistent structure via a composed AR filter ``\Psi(B)`` and the remaining short-term dynamics via an ARMA kernel.

Given a univariate series ``\{Y_t,\ t=1,2,\ldots,n\}``, ARARMA produces a fitted model and forecasting mechanism that combine both stages.

### Stage 1 — Memory Shortening (ARAR)

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

- If ``\mathrm{Err}(\hat\tau)\le 8/n`` or if ``\hat\phi(\hat\tau)\ge 0.93`` with ``\hat\tau>2``, declare long memory and filter:
  ```math
  \tilde Y_t \;=\; Y_t - \hat\phi\,Y_{t-\hat\tau}.
  ```
- If ``\hat\phi(\hat\tau)\ge 0.93`` with ``\hat\tau\in\{1,2\}``, fit an AR(2) at lags 1 and 2 by normal equations and filter:
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

### Stage 2 — Best-Lag Subset AR (on the prefiltered series)

Let ``X_t = S_t - \bar S`` with ``\bar S`` the sample mean of the final prefiltered series. Over 4-term lag tuples ``(1,i,j,k)`` satisfying ``1<i<j<k\le m`` (with ``m`` typically 13 or 26), we fit the subset AR:
```math
X_t \;=\; \phi_1 X_{t-1} \;+\; \phi_{i} X_{t-i} \;+\; \phi_{j} X_{t-j} \;+\; \phi_{k} X_{t-k} \;+\; Z_t,
\qquad Z_t\sim \mathrm{WN}(0,\sigma^2).
```

Yule–Walker equations (using sample autocorrelations ``\hat\rho(\cdot)`` of ``X_t``) yield the coefficients:
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

### Stage 3 — Short-Memory ARMA(p, q) on AR Residuals

Using the AR-only fit implied by ``\xi(B)`` and ``c``, compute residuals and fit an **ARMA(p, q)** by maximizing the conditional Gaussian likelihood. Denote the ARMA polynomials
```math
\Phi(B) \;=\; 1 - \varphi_1 B - \cdots - \varphi_p B^{p},
\qquad
\Theta(B) \;=\; 1 + \theta_1 B + \cdots + \theta_q B^{q}.
```
The ARMA stage estimates ``(\varphi_1,\ldots,\varphi_p,\,\theta_1,\ldots,\theta_q,\,\sigma^2)`` via Nelder–Mead. The code log-parameterizes the variance for numerical stability.

**Information criteria.** With effective residual length ``n_{\text{eff}}`` and ``k=p+q`` parameters (variance excluded), the log-likelihood ``\ell`` yields
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

### References
- Parzen, E. (1985). *ARARMA Models for Time Series Analysis and Forecasting*. **Journal of Forecasting**, 1(1), 67–82.

---

## ARARMA Array Interface

```julia
using Durbyn
using Durbyn.Ararma

ap = air_passengers()

# ARARMA with specified orders
fit1 = ararma(ap, p = 0, q = 1)
fc1 = forecast(fit1, h = 12)
plot(fc1)

# Automatic ARARMA order selection
fit2 = auto_ararma(ap)
fc2 = forecast(fit2, h = 12)
plot(fc2)
```
