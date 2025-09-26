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