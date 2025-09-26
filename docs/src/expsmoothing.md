# Exponential Smoothing (ETS): State-Space Form & Admissible Parameter Space

This page summarizes (i) the **ETS state-space framework** for automatic forecasting and (ii) the **admissible parameter regions** for stability/forecastability. Durbyn.jl ``ets()`` function implements the standard ETS references used in R’s `forecast::ets` Hyndman et al. (2002, 2008).

## Model taxonomy and notation

We use the ETS (Error, Trend, Seasonality) codes with **additive errors**:

- **ANN** — simple exponential smoothing (no trend, no seasonality)  
- **AAN** — additive trend (Holt)  
- **ADN** — **damped** additive trend (Holt–damped, with ``0<\phi\le 1``)  
- **ANA / AAA / ADA** — additive seasonality of period ``m`` with none / additive / damped trend

Hyndman et al (2002) show each method has a **single-source-of-error state-space** representation that yields the same point forecasts as classical smoothing while enabling likelihood-based inference and AIC selection.

We use smoothing parameters ``\,\alpha, \beta, \gamma\,\in\mathbb{R}`` and damping ``\,\phi\in (0,1]`` (if present). Here, **``\beta`` is the trend smoothing parameter** (not the product ``\alpha\beta``) so the admissible regions are bounded in the natural way.

## State-space formulation

We write the additive-error ETS in innovations state-space form:

```math
\begin{aligned}
\textbf{Observation:}\quad
& Y_t \;=\; H\,x_{t-1} \;+\; \varepsilon_t, \\
\textbf{State:}\quad
& x_t \;=\; F\,x_{t-1} \;+\; G\,\varepsilon_t, \qquad \varepsilon_t \sim \text{WN}(0,\sigma^2).
\end{aligned}
```

**Forecast mean and variance** at horizon ``h`` (conditional on ``x_n``):

```math
\mu_n(h)=\mathbb{E}[Y_{n+h}\mid x_n]=H\,F^{\,h-1}x_n,
\qquad
v_n(h)=\operatorname{Var}(Y_{n+h}\mid x_n)=\sigma^2\!\left(1+\sum_{j=1}^{h-1} (H F^{j-1} G)^2\right).
```

Matrices ``(F,G,H)`` depend on the ETS variant (ANN/AAN/…/ADA). The usual smoothing recursions fall out by writing the state update in **error-correction** form.

## Model properties

Let ``M := F - G H``.

- **Observability**: ``\operatorname{rank}\!\big([H^\top,\,F^\top H^\top,\dots,(F^\top)^{p-1}H^\top]\big)=p``  
- **Reachability**: ``\operatorname{rank}\!\big([G,FG,\dots,F^{p-1}G]\big)=p``  
- **Stability**: all eigenvalues of ``M`` lie **strictly** inside the unit circle (equivalently, invertible reduced ARIMA).  
- **Forecastability** (weaker): for each eigenpair ``(\lambda_i,v_i)``, either ``|\lambda_i|<1`` or ``H F^j v_i = 0`` for all ``j\ge 0`` (unstable modes do not affect forecasts).

## Admissible regions (non-seasonal, additive errors)

For ANN/AAN/ADN the **stability** regions admit clean inequalities:

- **ANN**  
  ```math
  0 \;<\; \alpha \;<\; 2.
  ```

- **AAN**  
  ```math
  0 \;<\; \alpha \;<\; 2, 
  \qquad
  0 \;<\; \beta \;<\; 4-2\alpha.
  ```

- **ADN** (damped trend)  
  ```math
  0 \;<\; \phi \;\le\; 1,\qquad
  1-\tfrac{1}{\phi} \;<\; \alpha \;<\; 1+\tfrac{1}{\phi},\qquad
  \alpha(\phi-1) \;<\; \beta \;<\; (1+\phi)(2-\alpha).
  ```

Under the “usual” constraints ``0<\beta<\alpha<1`` you lie **inside** these regions, so non-seasonal ETS is stable.

## Seasonal ETS: standard vs normalized

### The pitfall (standard Holt–Winters seasonality)
In **ANA/AAA/ADA** with the usual seasonal recursion ``s_t = s_{t-m} + \gamma \varepsilon_t``, the state matrix ``M`` always has a **unit eigenvalue**; the model is **unstable** and **neither** reachable **nor** observable. Forecasts can still be **forecastable** (the unstable direction is orthogonal to the forecast functional), but the **state estimates are corrupted**.

For ADA, the characteristic equation factors as

```math
f(\lambda)=(1-\lambda)\,P(\lambda),
```

with the **forecastability** polynomial

```math
\begin{aligned}
P(\lambda)
&= \lambda^{m+1}
+ (\alpha+\beta-\phi)\lambda^{m}
+ (\alpha+\beta-\alpha\phi)\lambda^{m-1}
+ \cdots \\
&\quad + (\alpha+\beta-\alpha\phi+\gamma-1)\lambda
+ \phi(1-\alpha-\gamma).
\end{aligned}
```

Forecastability requires **all roots of ``P(\lambda)``** to lie strictly inside the unit circle.  
For **AAA**, set ``\phi=1``.

### The fix (normalized seasonality)
Impose a **sum-to-zero** seasonal constraint every period (Roberts’ normalization). In backshift notation:

```math
S(B)\,s_t = \theta(B)\,\gamma\,\varepsilon_t,
\quad
S(B)=1+B+\cdots+B^{m-1},
\quad
\theta(B)=\frac1m\!\left[(m-1)+(m-2)B+\cdots+B^{m-2}\right].
```

Operationally: update seasonals as usual, **then subtract the average of the last ``m`` shocks** so the ``m`` seasonal states sum to zero.  
With this normalization, the seasonal ETS becomes **stable**. The stability polynomial of **normalized ADA** equals the forecastability polynomial of **standard ADA** after the re-parameterization ``\alpha\mapsto \alpha-\gamma/m``.

## Seasonal quick screens & definitive checks

### Easy necessary bounds (fast rejects)

- **ANA** (no trend):
  ```math
  \max(-m\alpha,\,0) \;<\; \gamma \;<\; 2-\alpha,
  \qquad
  -\frac{2}{m-1} \;<\; \alpha \;<\; 2-\gamma.
  ```

- **ADA** (damped trend):
  ```math
  0 \;<\; \phi \;\le\; 1,
  \qquad
  \max\!\Big(1-\tfrac{1}{\phi}-\alpha,\,0\Big) \;<\; \gamma \;<\; 1+\tfrac{1}{\phi}-\alpha,
  ```
  plus additional (more involved) bounds on ``\alpha`` and ``\beta`` and, ultimately, the **root test** on ``P(\lambda)`` above.

In practice: (1) check these simple inequalities; (2) if passed, **numerically** check the roots of ``P(\lambda)`` are inside the unit disk.  
For **normalized** seasonal ETS, this becomes a **stability** check via ``\alpha\mapsto \alpha-\gamma/m``.

---

## References

- **Hyndman et al (2002)** — Hyndman, Koehler, Snyder & Grose. *A state space framework for automatic forecasting using exponential smoothing methods.* International Journal of Forecasting 18(3):439–454.  
- **Hyndman et al (2006)** — Hyndman, Akram & Archibald. *The admissible parameter space for exponential smoothing models.*

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