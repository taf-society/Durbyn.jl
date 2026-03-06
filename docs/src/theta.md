# Theta Method (STM, OTM, DSTM, DOTM)

!!! tip "Formula Interface is the Recommended Approach"
    Use `ThetaSpec` with `@formula` for a declarative interface that works with panel data,
    grouped fitting, and model comparison. The base (array) API is shown near the end.

The Theta method decomposes a series into **theta lines** (one capturing long-run trend,
one capturing short-run curvature) and recombines their forecasts. Durbyn implements the
four variants discussed by Fiorucci et al. (2016):

- `:STM` - Simple Theta (theta = 2 fixed, alpha optimized)
- `:OTM` - Optimized Theta (theta and alpha optimized)
- `:DSTM` - Dynamic Simple Theta (theta = 2, regression coefficients updated each step)
- `:DOTM` - Dynamic Optimized Theta (dynamic regression + optimized theta, alpha)

Seasonality is handled with additive or multiplicative decomposition prior to fitting
(auto-detected unless specified).

### Variant Selection Guide

| Variant | Best for | Trade-off |
|---------|----------|-----------|
| **STM** | Quick baseline forecasts | Fast but less flexible (fixed θ=2) |
| **OTM** | Series where θ≠2 improves fit | Better accuracy, slightly slower |
| **DSTM** | Non-stationary trend patterns | Adapts to changing trends |
| **DOTM** | Complex series with evolving dynamics | Most flexible, best for longer series |

Use `theta()` with no model argument to let Durbyn auto-select the best variant via in-sample MSE.

---

## Formula Interface (primary usage)

```julia
using Durbyn, Durbyn.Grammar

data = (sales = [120, 135, 148, 152, 141, 158, 170, 165, 180, 195],)

# Auto-select among STM/OTM/DSTM/DOTM via in-sample MSE
spec = ThetaSpec(@formula(sales = theta()))
fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 12)
```

### Dynamic Optimised Theta with explicit options

```julia
using Durbyn, Durbyn.Grammar

data = (demand = collect(1.0:100.0),)

spec = ThetaSpec(@formula(demand = theta(
    model = :DOTM,
    decomposition = "multiplicative",
    nmse = 5,              # optimise on 1-5 step SSE
    theta_param = nothing, # optimise theta
    alpha = nothing        # optimise alpha
)))

fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 12)
```

### Panel data / grouped fitting

```julia
using Durbyn, Durbyn.TableOps, Durbyn.ModelSpecs, Durbyn.Grammar

# stacked table with :series column
panel = PanelData(tbl; groupby = :series, date = :date, m = 12)
spec = ThetaSpec(@formula(value = theta(model = :OTM, decomposition = "additive")))
fitted = fit(spec, panel)
fc = forecast(fitted, h = 12)
```

Key options (all optional): `model`, `alpha`, `theta_param`, `decomposition` ("multiplicative" or "additive"), `nmse` (1–30 step MSE objective).

---

## Base API (array interface)

```julia
using Durbyn

y = collect(1.0:50.0) .+ randn(50)

# Fit a specific variant
fit_otm = theta(y, 12; model_type = OTM, nmse = 3)   # Optimised Theta

# Let Durbyn choose the best variant by MSE
fit_auto = auto_theta(y, 12)

fc = forecast(fit_auto; h = 8)  # returns Forecast with mean & intervals
```

`ThetaFit` exposes `model_type`, `alpha`, `theta`, `initial_level`, `mse`, fitted values,
residuals, and decomposition metadata.

---

## Methodology (Equation Mapping)

This implementation follows the equations in the Theta model papers and the math specification.

Seasonality test (used before decomposition, with lag `m`):

```math
t_m = \frac{r_m}{\sqrt{\left(1 + 2\sum_{k=1}^{m-1} r_k^2\right)/n}},
\qquad
\text{seasonal if } |t_m| > 1.645.
```

Theta-line decomposition:

```math
Z_t(\theta) = \theta Y_t + (1-\theta)(A_n + B_n t),
```
```math
B_n = \frac{6}{n^2-1}\left[\frac{2}{n}\sum_{t=1}^{n} tY_t - \frac{1+n}{n}\sum_{t=1}^{n} Y_t\right],
\qquad
A_n = \frac{1}{n}\sum_{t=1}^{n}Y_t - \frac{n+1}{2}B_n.
```

Static Theta state equations (STM/OTM):

```math
\mu_t = \ell_{t-1} + \left(1-\frac{1}{\theta}\right)\left[(1-\alpha)^{t-1}A_n + \frac{1-(1-\alpha)^t}{\alpha}B_n\right],
\qquad
\ell_t = \alpha Y_t + (1-\alpha)\ell_{t-1}.
```

Dynamic Theta state equations (DSTM/DOTM):

```math
\mu_t = \ell_{t-1} + \left(1-\frac{1}{\theta}\right)\left[(1-\alpha)^{t-1}A_{t-1} + \frac{1-(1-\alpha)^t}{\alpha}B_{t-1}\right],
```
```math
\bar{Y}_t = \frac{(t-1)\bar{Y}_{t-1} + Y_t}{t},
\qquad
B_t = \frac{(t-2)B_{t-1} + 6(Y_t-\bar{Y}_{t-1})/t}{t+1},
\qquad
A_t = \bar{Y}_t - \frac{t+1}{2}B_t.
```

Optimization target (`nmse = H`) is horizon-averaged multi-step SSE:

```math
\mathcal{L} = \frac{1}{H}\sum_{h=1}^{H}\text{MSE}_h,\qquad
\text{MSE}_h = \frac{1}{N_h}\sum_t (Y_{t+h}-\widehat{Y}_{t+h\mid t})^2.
```

For DSTM/DOTM, error accumulation starts at index `t=3` as in the dynamic model definition.

Out-of-sample recursion uses the same state equations, with forecast substitution
(`Y_{t+1}` replaced by `\widehat{Y}_{t+1|t}`) for horizons `h>1`.

### Prediction intervals

The conditional variance for ``h``-step-ahead forecasts:

```math
\mathrm{Var}\!\left[Y_{n+h} \mid Y_1,\ldots,Y_n\right] = \left[1 + (h-1)\alpha^2\right]\sigma^2,
```

yielding prediction intervals:

```math
\widehat{Y}_{n+h\mid n} \pm z_{1-a/2}\sqrt{\left[1 + (h-1)\alpha^2\right]\sigma^2}.
```

### Connection to SES with drift

The mapping to SES with drift (Theorem 2 in Fiorucci et al.) is ``b = (1-\tfrac{1}{\theta})B_n`` and
``\ell^{**}_0 = \ell_0 + (1-\tfrac{1}{\theta})A_n``.

---

## References

- Fiorucci, J. A., Pellegrini, T. R., Louzada, F., Petropoulos, F., & Koehler, A. B. (2016).
  *Models for optimising the theta method and their relationship to state space models.*
  International Journal of Forecasting, 32(4), 1151–1161.
- Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a decomposition 
  approach to forecasting. International Journal of Forecasting, 16(4), 521-530.
- Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method.
  International Journal of Forecasting, 19(2), 287-290.
