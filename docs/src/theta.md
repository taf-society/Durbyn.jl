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

## Methodology (Fiorucci et al., 2016)

The classic Theta line with second-difference scaling ``\theta``:

```math
Z_t(\theta) = \theta Y_t + (1-\theta)(A_n + B_n t), \qquad
A_n = \overline{Y} - \tfrac{n+1}{2}B_n,\quad
B_n = \frac{6}{n^2-1}\!\left(\frac{2}{n}\sum_{t=1}^{n} tY_t - \frac{1+n}{n}\sum_{t=1}^{n} Y_t\right).
```

Recomposition for ``\theta_1 = 0, \theta_2 = \theta`` gives
```math
Y_t = \left(1-\tfrac{1}{\theta}\right)(A_n + B_n t) + \tfrac{1}{\theta} Z_t(\theta).
```

Forecasts combine regression and the SES extrapolation of the theta line:

```math
\widehat{Y}_{n+h\mid n} =
\left(1-\tfrac{1}{\theta}\right)\!\bigl[A_n + B_n(n+h)\bigr]
 + \tfrac{1}{\theta}\,\widetilde{Z}_{n+h\mid n}(\theta),
```

with SES recursion and closed form for ``h=1``:

```math
\ell_t = \alpha Y_t + (1-\alpha)\ell_{t-1}, \qquad \ell_0 = \ell_0^*/\theta
```
```math
\widetilde{Z}_{n+1\mid n}(\theta) =
\theta \ell_n + (1-\theta)\!\left[
  A_n\bigl(1-(1-\alpha)^n\bigr) +
  B_n\Bigl(n+\bigl(1-\tfrac{1}{\alpha}\bigr)\bigl(1-(1-\alpha)^n\bigr)\Bigr)
\right].
```

Optimised Theta (state-space form):

```math
Y_t = \mu_t + \varepsilon_t, \qquad
\mu_t = \ell_{t-1} + \left(1-\tfrac{1}{\theta}\right)\!\left[(1-\alpha)^{t-1}A_n + \tfrac{1-(1-\alpha)^t}{\alpha}B_n\right], \qquad
\ell_t = \alpha Y_t + (1-\alpha)\ell_{t-1}.
```

The ``h``-step-ahead forecast from origin ``n``:

```math
\widehat{Y}_{n+h\mid n} = \ell_n + \left(1-\tfrac{1}{\theta}\right)\!\left[(1-\alpha)^n A_n + \left[(h-1) + \tfrac{1-(1-\alpha)^{n+1}}{\alpha}\right]B_n\right].
```

Dynamic variants update regression coefficients each period:

```math
\widehat{Y}_{t+1\mid t} = \ell_t + \left(1-\tfrac{1}{\theta}\right)\!\left[(1-\alpha)^t A_t + \tfrac{1-(1-\alpha)^{t+1}}{\alpha}B_t\right],
```

```math
A_t = \overline{Y}_t - \tfrac{t+1}{2} B_t, \qquad
B_t = \tfrac{1}{t+1}\!\left[(t-2)B_{t-1} + \tfrac{6}{t}(Y_t - \overline{Y}_{t-1})\right], \qquad
\overline{Y}_t = \tfrac{1}{t}\!\left[(t-1)\overline{Y}_{t-1} + Y_t\right].
```

Parameters are estimated by SSE / Gaussian likelihood:

```math
(\widehat{\ell}_0, \widehat{\alpha}, \widehat{\theta}) = \arg\min_{\ell_0,\alpha,\theta} \sum_{t=1}^{n} (Y_t - \mu_t)^2,
\qquad
l = -\tfrac{n}{2}\log(\mathrm{SSE}/n) - \tfrac{n}{2}(1+\log 2\pi).
```

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
