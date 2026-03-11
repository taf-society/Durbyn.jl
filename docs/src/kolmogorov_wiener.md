# Kolmogorov-Wiener Optimal Filters

## Overview

The Kolmogorov-Wiener (KW) module implements **optimal finite-sample filters** for time series, based on:

> Schleicher, C. (2004). *Kolmogorov-Wiener Filters for Finite Time-Series.*
> SSRN Working Paper. DOI: 10.2139/ssrn.769584.

Standard symmetric filters — Hodrick-Prescott, Baxter-King bandpass, Butterworth — are
designed for *infinite* series. Applied to finite samples, they either lose observations at the
endpoints or introduce ad-hoc adjustments (mirroring, padding) with no optimality guarantee.
The KW approach derives **minimum mean squared error** filter weights at every observation,
including endpoints, by exploiting the autocovariance structure of the data-generating process.

This is particularly relevant for:

- **Output gap estimation** — endpoint bias in the HP filter directly affects real-time policy conclusions
- **Business cycle dating** — bandpass-filtered series lose the most recent observations, exactly where timeliness matters
- **Trend-cycle decomposition** — the KW filter provides an exact additive decomposition with optimal endpoint handling
- **Real-time signal extraction** — filter weights adapt automatically at the boundaries

---

## Mathematical Framework

### 1. The Filtering Problem

Let ``\{y_t\}`` be a time series of length ``T``. An **ideal symmetric filter** applies weights
``B_j`` for ``j = -Q, \ldots, Q`` to extract a signal:

```math
\hat{y}_t^* = \sum_{j=-Q}^{Q} B_j \, y_{t+j}
```

This requires observations ``y_{t-Q}`` through ``y_{t+Q}``, which are unavailable near the
endpoints. The KW approach replaces the ideal filter with an **optimal finite-sample filter**
``\hat{B}`` of length ``N = n_1 + 1 + n_2`` (where ``n_1`` observations precede ``t`` and
``n_2`` follow it):

```math
\hat{y}_t = \sum_{i=1}^{N} \hat{B}_i \, y_i
```

The weights ``\hat{B}`` minimize the **mean squared error**:

```math
\text{MSE} = E\!\left[\left(\hat{y}_t - \hat{y}_t^*\right)^2\right]
```

### 2. Frequency Domain: Ideal Filter Coefficients

The ideal filter is defined by its **transfer function** ``H(\omega)`` on ``[0, \pi]``.
The impulse response coefficients are recovered via the inverse Fourier cosine transform:

```math
B_0 = \frac{1}{\pi} \int_0^{\pi} H(\omega) \, d\omega, \qquad
B_j = \frac{1}{\pi} \int_0^{\pi} H(\omega) \cos(j\omega) \, d\omega, \quad j \neq 0
```

The filter is symmetric: ``B_{-j} = B_j``. Three standard filters are supported:

#### Bandpass Filter (Eq. 30)

Passes frequencies in the band ``[\omega_a, \omega_b]``:

```math
H(\omega) = \begin{cases} 1 & \omega_a \le \omega \le \omega_b \\ 0 & \text{otherwise} \end{cases}
```

Closed-form coefficients:

```math
B_0 = \frac{\omega_b - \omega_a}{\pi}, \qquad
B_j = \frac{\sin(\omega_b j) - \sin(\omega_a j)}{\pi j}
```

For business cycle extraction, the band ``[2\pi/32, \; 2\pi/6]`` isolates periods of 6 to 32
quarters (Burns & Mitchell, 1946; Baxter & King, 1999).

#### Hodrick-Prescott Filter

The HP filter (Hodrick & Prescott, 1997) is a highpass filter with transfer function:

```math
H(\omega) = \frac{4\lambda(1 - \cos\omega)^2}{1 + 4\lambda(1 - \cos\omega)^2}
```

where ``\lambda`` controls the smoothness of the trend (``\lambda = 1600`` for quarterly data,
``\lambda = 129\,600`` for monthly data). Coefficients are computed via numerical integration
(adaptive Simpson's rule).

#### Butterworth Filter

The Butterworth highpass of order ``n`` with cutoff frequency ``\omega_c``:

```math
|H(\omega)|^2 = \frac{\lambda \left(2 - 2\cos\omega\right)^{2n}}{1 + \lambda \left(2 - 2\cos\omega\right)^{2n}}, \qquad \lambda = \left(\frac{1}{\tan(\omega_c/2)}\right)^{2n}
```

This provides a sharper frequency cutoff than the HP filter, with roll-off controlled by the
order parameter.

### 3. Autocovariance from ARIMA Models

The KW filter exploits the autocovariance structure ``\gamma_k = \text{Cov}(y_t, y_{t+k})``
of the stationary component. Given an ARIMA``(p,d,q)(P,D,Q)_s`` model fitted to the data:

1. **Extract** the expanded AR polynomial ``\Phi(B)`` and MA polynomial ``\Theta(B)``
   (including seasonal factors)
2. **Compute Wold coefficients** ``\psi_k`` via the recursion:
   ```math
   \psi_k = \theta_k + \sum_{j=1}^{\min(k,p)} \phi_j \, \psi_{k-j}
   ```
   with ``\psi_0 = 1`` and ``\theta_k = 0`` for ``k > q``
3. **Compute autocovariance** of the stationary ARMA component:
   ```math
   \gamma_k = \sigma^2 \sum_{j=0}^{L} \psi_j \, \psi_{j+k}
   ```
   where ``\sigma^2`` is the innovation variance

The integration order is ``d = d_{\text{ns}} + D_s``, combining nonseasonal and seasonal
differencing. This is an approximation: the paper assumes ``(1-L)^d`` differencing, while
seasonal differencing ``(1-L^s)^D`` has a different structure. The autocovariance ``\gamma``
is exact (computed from the expanded ARMA representation); only the proposition dispatch
uses this simplified order.

---

## Optimal Filter Propositions

The paper derives four propositions for computing optimal weights, depending on the
integration order ``d`` and the structure of the stationary component.

### Proposition 1: Stationary Process (``d = 0``)

When the process is stationary, the optimal filter satisfies:

```math
\hat{\Gamma} \, \hat{B} = \Gamma \, B
```

where:

- ``\hat{\Gamma}`` is the ``N \times N`` **Toeplitz autocovariance matrix** with entries
  ``\hat{\Gamma}_{ij} = \gamma_{|i-j|}``
- ``\Gamma`` is the ``N \times (2Q+1)`` **cross-covariance matrix** between observations and
  ideal filter positions, with entries ``\Gamma_{m,c} = \gamma_{|m + Q - n_1 - c|}``
- ``B`` is the ``(2Q+1)``-vector of ideal filter coefficients

The solution ``\hat{B} = \hat{\Gamma}^{-1} (\Gamma B)`` gives the MSE-optimal weights.

### Proposition 2: Random Walk with White Noise (``d \ge 1``, non-symmetric ideal filter)

When the stationary component ``\varepsilon_t`` is white noise (no AR or MA structure),
the solution simplifies to **tail redistribution**:

- **Interior observations** (where all ideal coefficients fall within the sample): use ideal
  weights directly
- **Endpoint observations** (where some ideal coefficients extend beyond the sample):
  redistribute the truncated tails to the nearest boundary observation

```math
\hat{B}_1 = B_{-n_1} + \sum_{j=-Q}^{-n_1-1} B_j, \qquad
\hat{B}_N = B_{n_2} + \sum_{j=n_2+1}^{Q} B_j
```

This preserves the sum of weights: ``\sum_i \hat{B}_i = \sum_j B_j = \beta``.

### Proposition 3: Random Walk, Symmetric Case (``d \ge 1``, symmetric ideal filter)

For the general case with ``\beta \ne 0`` and white noise innovations, the optimal filter
solves the augmented system (Eq. 21):

```math
\begin{bmatrix} D \\ \iota' \end{bmatrix} \hat{B}
= \begin{bmatrix} M B + \frac{\beta}{2} \tau \\ \beta \end{bmatrix}
```

where:

- ``D`` is the ``(N-1) \times N`` **cumulation matrix** (lower-triangular ones: partial sums)
- ``\iota'`` is the ``1 \times N`` row of ones (sum constraint ``\sum \hat{B}_i = \beta``)
- ``M`` is the ``(N-1) \times N`` **block matrix** with structure:
  - Top block ``M_1`` (``n_1 \times (n_1+1)``): upper triangular, entries ``-1`` except last
    column ``-\tfrac{1}{2}``
  - Bottom block ``M_2`` (``n_2 \times (n_2+1)``): lower triangular, entries ``1`` except first
    column ``\tfrac{1}{2}``
- ``\tau = \mathbf{1}_{N-1}`` is a vector of ones

For symmetric highpass filters (``\beta = 0``), this simplifies to the endpoint-adjustment
form shown in Schleicher. In implementation, this is numerically equivalent to the
tail-redistribution solution for finite truncation.

### Proposition 4: ARIMA with ARMA Structure (``d \ge 1``, general)

The most general case handles non-trivial autocovariance. The optimal filter solves (Eq. 25):

```math
\begin{bmatrix} \hat{\Gamma} D \\ \iota' \end{bmatrix} \hat{B}
= \begin{bmatrix} \tilde{\Gamma} C \\ \beta \end{bmatrix}
```

where:

- ``\hat{\Gamma}`` is the ``(N-1) \times (N-1)`` Toeplitz autocovariance matrix of the
  differenced (stationary) component
- ``D`` is the ``(N-1) \times N`` cumulation matrix
- ``\tilde{\Gamma}`` is the ``(N-1) \times (N-1+2Q)`` **integrated cross-covariance matrix**
  with entries ``\tilde{\Gamma}_{m,c} = \gamma_{|m+Q-c|}``
- ``C`` is the ``(N-1+2Q)``-vector of **cumulated ideal coefficients**:
  ```math
  C_j = \begin{cases}
  0 & j < -Q \\
  \sum_{k=-Q}^{j} B_k & -Q \le j \le Q \\
  \beta & j > Q
  \end{cases}
  ```
- ``\iota'`` enforces the sum constraint ``\sum \hat{B}_i = \beta``

For symmetric filters, the symmetric Proposition 4 form is used:

```math
\begin{bmatrix} \hat{\Gamma} D \\ \iota' \end{bmatrix} \hat{B}
= \begin{bmatrix} \tilde{\Gamma}\left(MB + \frac{\beta}{2}\tau\right) \\ \beta \end{bmatrix}
```

**Key property:** As ``\gamma`` approaches white noise (AR/MA coefficients ``\to 0``),
Proposition 4 converges to Proposition 2/3. This is verified numerically in the test suite.

### Proposition Selection

The implementation automatically selects the appropriate proposition:

| Condition | Proposition | Method |
|-----------|-------------|--------|
| ``d = 0`` | Prop 1 | Full autocovariance solve |
| ``d \ge 1``, white noise, non-symmetric ideal filter | Prop 2 | Tail redistribution |
| ``d \ge 1``, white noise, symmetric ideal filter | Prop 3 | Augmented random-walk symmetric system |
| ``d \ge 1``, ARMA structure, non-symmetric ideal filter | Prop 4 | Integrated cross-covariance system |
| ``d \ge 1``, ARMA structure, symmetric ideal filter | Prop 4 (symmetric) | Integrated cross-covariance + ``MB + (\beta/2)\tau`` |

White noise detection uses a threshold: ``|\gamma_k| < 10^{-10} \cdot |\gamma_0|`` for all
``k \ge 1``.

Symmetry detection checks ``B_{-j} \approx B_j`` with a relative tolerance of ``10^{-10}``.

---

## Forecasting

The module extends the KW framework to **Wiener-optimal multi-step-ahead prediction**.
For each forecast horizon ``j = 1, \ldots, h``, the optimal weight vector ``b_j`` solves:

```math
\Gamma \, b_j = g_j
```

where ``\Gamma`` is the ``p \times p`` Toeplitz autocovariance matrix and
``g_j = [\gamma(j), \gamma(j+1), \ldots, \gamma(j+p-1)]'`` is the cross-covariance between
the regressor block and the future value at lag ``j``.

The point forecast and MSE are:

```math
\hat{y}_{T+j} = b_j' z, \qquad \text{MSE}_j = \gamma_0 - g_j' b_j
```

where ``z = [y_T, y_{T-1}, \ldots, y_{T-p+1}]'`` is the regressor vector.

### Integrated Series

For ``d \ge 1``, the forecast operates on the **fully differenced** series:

1. Apply ``(1 - L^s)^{D_s} (1 - L)^{d_{\text{ns}}}`` to stationarize
2. Forecast the stationary component using Wiener prediction
3. **Cumulate back** to levels by undoing the differencing operations (nonseasonal first,
   then seasonal)
4. **Cumulate the forecast error covariance** through the inverse differencing operator to
   obtain prediction intervals at the level scale

---

## Usage

### Optimal Filtering

```julia
using Durbyn

y = air_passengers()

# HP filter — optimal endpoint handling
r = kolmogorov_wiener(y, :hp; m=12, lambda=1600.0)
r.filtered          # cycle component
r.y .- r.filtered   # trend (by subtraction)

# Trend output directly
r_trend = kolmogorov_wiener(y, :hp; m=12, lambda=1600.0, output=:trend)

# Bandpass filter for business cycle (6–32 quarters)
r_bp = kolmogorov_wiener(y, :bandpass; low=6, high=32, m=4)

# Butterworth highpass
r_bw = kolmogorov_wiener(y, :butterworth; order=2, omega_c=pi/16, m=12)

# Custom transfer function
r_c = kolmogorov_wiener(y, :custom; transfer_fn=w -> w < pi/6 ? 1.0 : 0.0, m=12)
```

### Trend-Cycle Decomposition

```julia
d = kw_decomposition(y; m=12)
d.trend                        # smooth trend
d.remainder                    # business cycle component
d.trend .+ d.remainder ≈ d.data  # exact identity

plot(d)  # multi-panel: Data, Trend, Remainder
```

### Forecasting

```julia
# From filter result
fc = forecast(r; h=24)
fc.mean            # point forecasts
fc.upper[:, 2]     # 95% upper bounds
fc.lower[:, 2]     # 95% lower bounds

# From decomposition
fc = forecast(d; h=24)
```

### ARIMA Control

```julia
# Pre-fitted ARIMA model
fit = auto_arima(y, 12)
r = kolmogorov_wiener(y, :hp; arima_model=fit)

# Explicit ARIMA constraints
r = kolmogorov_wiener(y, :hp; m=12, d=1, D=1, max_p=3, stepwise=false)

# Box-Cox transformation (separate from HP lambda)
r = kolmogorov_wiener(y, :hp; m=12, boxcox_lambda=0.0, biasadj=true)
```

### Grammar Interface

The KW filter can be used with Durbyn's forecasting grammar for declarative
model specification, including panel data support.

```julia
using Durbyn

# Specify a KW filter model via @formula
spec = KwFilterSpec(@formula(gdp = kw_filter()))

# HP filter (default)
spec = KwFilterSpec(@formula(gdp = kw_filter(filter=:hp, lambda=1600)))

# Bandpass filter for business cycles
spec = KwFilterSpec(@formula(gdp = kw_filter(filter=:bandpass, low=6, high=32)))

# Butterworth filter
spec = KwFilterSpec(@formula(gdp = kw_filter(filter=:butterworth, order=2, omega_c=0.2)))

# Fit to tabular data (NamedTuple, DataFrame, etc.)
data = (gdp = y,)
fitted_model = fit(spec, data, m=4)

# Forecast
fc = forecast(fitted_model, h=8)

# Trend extraction
spec_trend = KwFilterSpec(@formula(gdp = kw_filter(output=:trend)))
fitted_trend = fit(spec_trend, data, m=4)
```

**Panel data** — fit to multiple groups simultaneously:

```julia
data = (gdp = vcat(y1, y2), country = vcat(fill(:US, n), fill(:UK, n)))

# Single groupby column
spec = KwFilterSpec(@formula(gdp = kw_filter()))
grouped = fit(spec, data, m=4, groupby=:country)

# Multiple groupby columns
grouped = fit(spec, data, m=4, groupby=[:country, :sector])

# PanelData wrapper
panel = PanelData(data, groupby=[:country], m=4)
grouped = fit(spec, panel)
```

---

## API Reference

```@docs
Durbyn.kolmogorov_wiener
Durbyn.kw_decomposition
Durbyn.KWFilterResult
Durbyn.KwFilterSpec
Durbyn.kw_filter
```

---

## References

- Schleicher, C. (2004). *Kolmogorov-Wiener Filters for Finite Time-Series.* SSRN. DOI: 10.2139/ssrn.769584.
- Hodrick, R. J. & Prescott, E. C. (1997). *Postwar U.S. Business Cycles: An Empirical Investigation.* Journal of Money, Credit and Banking, 29(1), 1–16.
- Baxter, M. & King, R. G. (1999). *Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series.* Review of Economics and Statistics, 81(4), 575–593.
- Christiano, L. J. & Fitzgerald, T. J. (2003). *The Band Pass Filter.* International Economic Review, 44(2), 435–465.
- Butterworth, S. (1930). *On the Theory of Filter Amplifiers.* Experimental Wireless and the Wireless Engineer, 7, 536–541.
