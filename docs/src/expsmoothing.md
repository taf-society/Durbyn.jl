# Exponential Smoothing (ETS): State-Space Form, Additive & Multiplicative Models

This page summarizes the **ETS state-space framework wich is implemented in Durbyn.jl as ``ets()``** for automatic forecasting, and the **admissible parameter regions** for stability/forecastability.  
It includes both **additive** and **multiplicative** error models, following Hyndman et al. (2002, 2008).

---

## Model taxonomy and notation

ETS models are categorized by (Error, Trend, Seasonality):

- **ANN / MNN** — simple exponential smoothing (additive vs multiplicative error)  
- **AAN / MAN** — additive trend (Holt, with additive vs multiplicative error)  
- **ADN** — damped additive trend (only additive error common in practice)  
- **AAA / MAM** — additive trend + additive/multiplicative seasonality  
- **ANA / AAA / ADA** — seasonal additive-error forms (with no / additive / damped trend)  
- Other hybrids (e.g. multiplicative seasonality with additive error, damped multiplicative trend) can be defined analogously.

We use smoothing parameters ``\alpha,\beta,\gamma`` and damping ``\phi`` (if present).  
**Additive vs multiplicative error models give the same point forecasts but different likelihoods and intervals.**

---

## Additive error state-space form

```math
\begin{aligned}
\textbf{Observation:}\quad
& Y_t = Hx_{t-1} + \varepsilon_t, \\
\textbf{State:}\quad
& x_t = Fx_{t-1} + G\varepsilon_t, \qquad \varepsilon_t \sim WN(0,\sigma^2).
\end{aligned}
```

Forecast mean and variance at horizon ``h``:

```math
\mu_n(h) = H F^{h-1} x_n, \qquad
v_n(h) = \sigma^2\left(1 + \sum_{j=1}^{h-1} (HF^{j-1}G)^2\right).
```

---

## Multiplicative error form

For multiplicative error models:

- **Observation:**  
  ```math
  Y_t = \hat{Y}_t (1+\varepsilon_t),
  ```  
  where ``\varepsilon_t \sim WN(0,\sigma^2)``.

- **Key property:** Point forecasts are the same as additive-error models, but prediction intervals scale with the level.

### Examples

- **MNN** (no trend, no seasonality):
  ```math
  Y_t = \ell_{t-1}(1+\varepsilon_t), \qquad
  \ell_t = \ell_{t-1}(1+\alpha\varepsilon_t).
  ```

- **MAN** (additive trend):
  ```math
  Y_t = (\ell_{t-1}+b_{t-1})(1+\varepsilon_t), \\
  \ell_t = (\ell_{t-1}+b_{t-1})(1+\alpha\varepsilon_t), \\
  b_t = b_{t-1} + \beta(\ell_{t-1}+b_{t-1})\varepsilon_t.
  ```

- **MAM** (additive trend + multiplicative seasonality):
  ```math
  Y_t = (\ell_{t-1}+b_{t-1})s_{t-m}(1+\varepsilon_t), \\
  \ell_t = (\ell_{t-1}+b_{t-1})(1+\alpha\varepsilon_t), \\
  b_t = b_{t-1}+\beta(\ell_{t-1}+b_{t-1})\varepsilon_t, \\
  s_t = s_{t-m}(1+\gamma\varepsilon_t).
  ```

Other multiplicative combinations (e.g. damped trend, hybrid seasonality) follow analogously.

---

## Model properties

Let ``M = F-GH``.

- **Observability**: ``\operatorname{rank}([H^\top,(F^\top)H^\top,\dots,(F^\top)^{p-1}H^\top])=p``  
- **Reachability**: ``\operatorname{rank}([G,FG,\dots,F^{p-1}G])=p``  
- **Stability**: eigenvalues of ``M`` lie inside the unit circle  
- **Forecastability**: weaker notion, unstable modes do not affect forecasts if orthogonal to forecast functional

Non-seasonal additive/multiplicative ETS are minimal (reachable & observable).  
Standard seasonal ETS are not (contain redundant seasonal states).

---

## Admissible regions (non-seasonal, additive & multiplicative)

For ANN/AAN/ADN (and their multiplicative analogues), the admissible stability regions are identical:

- **ANN / MNN**
  ```math
  0 < \alpha < 2.
  ```

- **AAN / MAN**
  ```math
  0 < \alpha < 2, \qquad 0 < \beta < 4-2\alpha.
  ```

- **ADN** (damped additive trend)
  ```math
  0 < \phi \le 1, \qquad
  1-\tfrac{1}{\phi} < \alpha < 1+\tfrac{1}{\phi}, \qquad
  \alpha(\phi-1) < \beta < (1+\phi)(2-\alpha).
  ```

Thus, admissible regions do not depend on whether errors are additive or multiplicative.

---

## Seasonal ETS

### Standard Holt–Winters seasonal form

In ANA/AAA/ADA with recursion ``s_t=s_{t-m}+\gamma\varepsilon_t``,  
``M`` has a **unit eigenvalue** → unstable, non-minimal.  
Forecasts can remain valid (forecastable) but states are corrupted.

Characteristic polynomial factorization (ADA case):
```math
f(\lambda) = (1-\lambda)P(\lambda),
```
with forecastability polynomial ``P(\lambda)`` whose roots must lie inside the unit circle.  
AAA is the special case ``\phi=1``.

### Normalized seasonal ETS

Fix instability by imposing a sum-to-zero seasonal constraint each period:
```math
S(B)s_t = \theta(B)\gamma\varepsilon_t,
```
where ``S(B)=1+B+\cdots+B^{m-1}``,  
``\theta(B)=\tfrac{1}{m}[(m-1)+(m-2)B+\cdots+B^{m-2}]``.

Operationally: after updating seasonals, subtract the average of last ``m`` shocks.  
This normalization restores stability.

---

## References

- Hyndman, Koehler, Snyder & Grose (2002). *A state space framework for automatic forecasting using exponential smoothing methods.*
- Hyndman, Akram & Archibald (2006). *The admissible parameter space for exponential smoothing models.*
- Hyndman, R.J., Koehler, A.B., Ord, J.K., Snyder, R.D. (2008) *Forecasting with exponential smoothing: the state space approach*, Springer-Verlag: New York. http://www.exponentialsmoothing.net
- Hyndman and Athanasopoulos (2018) *Forecasting: principles and practice*, 2nd edition, OTexts: Melbourne, Australia. https://otexts.com/fpp2/


---

## Simple Exponential Smoothing (SES)

**Simple exponential smoothing** is the simplest form of exponential smoothing (equivalent to ETS(A,N,N) or ETS(M,N,N)), with no trend or seasonality components. It is suitable for forecasting data with no clear trend or seasonal pattern.

### Mathematical Formulation

#### Additive Error Form (ANN)
```math
\begin{aligned}
Y_t &= \ell_{t-1} + \varepsilon_t, \\
\ell_t &= \ell_{t-1} + \alpha\varepsilon_t,
\end{aligned}
```
where ``\ell_t`` is the level at time ``t``, ``\alpha \in (0,1)`` is the smoothing parameter, and ``\varepsilon_t \sim WN(0,\sigma^2)``.

**Component form:**
```math
\ell_t = \alpha Y_t + (1-\alpha)\ell_{t-1}
```

**Forecast function:** The ``h``-step ahead forecast is simply the last estimated level:
```math
\hat{Y}_{n+h|n} = \ell_n \quad \text{for all } h \ge 1
```

**Prediction variance:**
```math
\text{Var}[\hat{Y}_{n+h|n}] = \sigma^2 h
```

#### Multiplicative Error Form (MNN)
```math
\begin{aligned}
Y_t &= \ell_{t-1}(1 + \varepsilon_t), \\
\ell_t &= \ell_{t-1}(1 + \alpha\varepsilon_t),
\end{aligned}
```

Point forecasts are identical to the additive form, but prediction intervals scale with the level.

### Admissible Parameter Space

For stability and forecastability:
```math
0 < \alpha < 2
```

In practice, ``\alpha`` is typically constrained to ``(0,1)`` for conventional exponential smoothing behavior.

### Usage

The `ses()` function provides two initialization methods:

- **`initial = "optimal"`** (default): Uses state-space optimization via ETS framework
- **`initial = "simple"`**: Uses conventional Holt-Winters initialization

```julia
using Durbyn
using Durbyn.ExponentialSmoothing

# Load example data
y = [10.5, 12.3, 11.8, 13.1, 12.9, 14.2, 13.8, 15.1, 14.7, 16.0]

# Fit SES with optimal initialization
fit = ses(y)

# Fit SES with specified alpha
fit_fixed = ses(y, alpha = 0.3)

# Fit SES with Box-Cox transformation
fit_bc = ses(y, lambda = 0.5)

# Generate forecasts
fc = forecast(fit, h = 6)

# For seasonal data (frequency m)
monthly_data = randn(60) .+ 100
fit_seasonal = ses(monthly_data, 12)  # m = 12 for monthly data
fc_seasonal = forecast(fit_seasonal, h = 12)
```

### Model Output

The `SES` struct contains:
- **`fitted`**: Fitted values (one-step ahead predictions)
- **`residuals`**: Residuals (observed - fitted)
- **`components`**: Model components (level)
- **`x`**: Original time series data
- **`par`**: Model parameters (alpha)
- **`initstate`**: Initial level estimate
- **`states`**: Level estimates over time
- **`sigma2`**: Residual variance
- **`aic`, `bic`, `aicc`**: Information criteria (when `initial = "optimal"`)
- **`mse`, `amse`**: Mean squared error measures
- **`lambda`**: Box-Cox transformation parameter (if used)
- **`biasadj`**: Boolean flag for bias adjustment

### When to Use SES

Use simple exponential smoothing when:
- Data exhibits no clear trend or seasonal pattern
- You need quick, computationally efficient forecasts
- Recent observations should be weighted more heavily than older ones
- You have limited data and want a parsimonious model

**Limitations:**
- Cannot capture trend or seasonality
- Forecasts are constant (flat line)
- May underperform for data with systematic patterns

For data with trend or seasonality, consider:
- **Holt's method** (`holt()`) for trended data
- **Holt-Winters** (`hw()`) for seasonal data
- **ETS** (`ets()`) for automatic model selection

---

## Holt's Linear Trend Method

**Holt's method** (also known as double exponential smoothing) extends SES to capture linear trends in time series data. It uses two smoothing parameters: α for the level and β for the trend component.

### Mathematical Formulation

#### Standard Holt's Method (Additive Trend)
```math
\begin{aligned}
Y_t &= \ell_{t-1} + b_{t-1} + \varepsilon_t, \\
\ell_t &= \alpha Y_t + (1-\alpha)(\ell_{t-1} + b_{t-1}), \\
b_t &= \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1},
\end{aligned}
```
where ``\ell_t`` is the level, ``b_t`` is the trend, ``\alpha, \beta \in (0,1)`` are smoothing parameters, and ``\varepsilon_t \sim WN(0,\sigma^2)``.

**Component form:**
- Level: ``\ell_t = \alpha Y_t + (1-\alpha)(\ell_{t-1} + b_{t-1})``
- Trend: ``b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}``

**Forecast function:** The ``h``-step ahead forecast incorporates the trend:
```math
\hat{Y}_{n+h|n} = \ell_n + h \cdot b_n
```

#### Damped Trend
```math
\begin{aligned}
Y_t &= \ell_{t-1} + \phi b_{t-1} + \varepsilon_t, \\
\ell_t &= \alpha Y_t + (1-\alpha)(\ell_{t-1} + \phi b_{t-1}), \\
b_t &= \beta(\ell_t - \ell_{t-1}) + (1-\beta)\phi b_{t-1},
\end{aligned}
```
where ``\phi \in (0,1]`` is the damping parameter.

**Forecast function:**
```math
\hat{Y}_{n+h|n} = \ell_n + (\phi + \phi^2 + \cdots + \phi^h) b_n = \ell_n + \phi\frac{1-\phi^h}{1-\phi}b_n
```

The damping parameter controls how quickly the trend dampens:
- ``\phi = 1``: Standard Holt (no damping)
- ``\phi < 1``: Damped trend (trend flattens out in forecasts)

**Advantages of damped trend:**
- More realistic long-term forecasts
- Prevents unbounded linear extrapolation
- Often improves forecast accuracy for horizons h > 10

#### Exponential (Multiplicative) Trend
```math
\begin{aligned}
Y_t &= \ell_{t-1} \cdot b_{t-1}^{\phi} + \varepsilon_t, \\
\ell_t &= \alpha Y_t + (1-\alpha) \ell_{t-1} \cdot b_{t-1}^{\phi}, \\
b_t &= \beta \frac{\ell_t}{\ell_{t-1}} + (1-\beta) b_{t-1}^{\phi},
\end{aligned}
```

Used when the trend grows/declines exponentially rather than linearly.

### Admissible Parameter Space

For standard Holt (no damping):
```math
\begin{aligned}
0 &< \alpha < 2, \\
0 &< \beta < 4 - 2\alpha
\end{aligned}
```

For damped Holt (``\phi < 1``):
```math
\begin{aligned}
0 &< \phi \le 1, \\
1 - \frac{1}{\phi} &< \alpha < 1 + \frac{1}{\phi}, \\
\alpha(\phi - 1) &< \beta < (1+\phi)(2-\alpha)
\end{aligned}
```

### Usage

```julia
using Durbyn
using Durbyn.ExponentialSmoothing

# Simulate data with linear trend
t = 1:50
y = 100 .+ 2 .* t .+ randn(50) .* 5

# Standard Holt's method (m parameter optional since no seasonality)
fit = holt(y)
println(fit)

# Generate forecasts
fc = forecast(fit, h=10)
plot(fc)

# Damped trend (recommended for long horizons)
fit_damped = holt(y, damped=true)
fc_damped = forecast(fit_damped, h=24)

# Holt with fixed parameters
fit_fixed = holt(y, alpha=0.8, beta=0.2)

# Exponential trend
fit_exp = holt(y, exponential=true)

# With Box-Cox transformation
fit_bc = holt(y, lambda="auto", biasadj=true)

# Simple initialization
fit_simple = holt(y, initial="simple")

# Can also specify m explicitly (though typically not needed)
fit_explicit = holt(y, 1, damped=true)
```

### Model Output

The `Holt` struct contains:
- **`fitted`**: Fitted values (one-step ahead predictions)
- **`residuals`**: Residuals (observed - fitted)
- **`components`**: Model components (level and trend)
- **`x`**: Original time series data
- **`par`**: Model parameters (alpha, beta, and phi if damped)
- **`initstate`**: Initial level and trend estimates
- **`states`**: Level and trend estimates over time
- **`sigma2`**: Residual variance
- **`aic`, `bic`, `aicc`**: Information criteria (when `initial = "optimal"`)
- **`mse`, `amse`**: Mean squared error measures
- **`lambda`**: Box-Cox transformation parameter (if used)
- **`biasadj`**: Boolean flag for bias adjustment
- **`method`**: Method description (e.g., "Holt's method", "Damped Holt's method")

### When to Use Holt's Method

Use Holt's linear trend method when:
- Data exhibits a clear linear trend (increasing or decreasing)
- No seasonal pattern is present
- You need to extrapolate the trend into the future
- Recent trend behavior should influence forecasts

**Use damped trends when:**
- Long-horizon forecasts are needed (h > 10)
- The trend may not continue indefinitely at the same rate
- You want more conservative, realistic forecasts
- Historical data shows trends that eventually flatten

**Limitations:**
- Cannot capture seasonality (use Holt-Winters `hw()` instead)
- Assumes trend is approximately linear
- Without damping, forecasts can be unrealistic for long horizons
- May overreact to recent trend changes

**Comparison with SES:**
- SES: No trend, forecasts are flat (constant)
- Holt: Linear trend, forecasts increase/decrease linearly
- Damped Holt: Trend that dampens, forecasts flatten over time

---

# Forecast with Automatic ETS model

```julia
using Durbyn
using Durbyn.ExponentialSmoothing
# Fit automatically selected ETS model to a monthly series (m = 12)
ap = air_passengers()
fit = ets(ap(), 12, "ZZZ")

# Specify a particular structure (multiplicative seasonality, additive trend, additive errors)
fit2 = ets(ap, 12, "AAM")
fc2 = forecast(fit2, h=12)
plot(fc2)

# Use a damped trend search and automatic Box–Cox selection
fit3 = ets(ap, 12, "ZZZ"; damped=nothing, lambda="auto", biasadj=true)
fc3 = forecast(fit3, h=12)
plot(fc3)
```