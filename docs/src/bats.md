# BATS: Box-Cox, ARMA errors, Trend, Seasonal Models

The **BATS** framework extends exponential smoothing to accommodate multiple,
possibly long seasonal cycles together with Box–Cox variance stabilization
and ARMA error correction. It was introduced by De Livera, Hyndman &
Snyder (2011) as part of the innovation-state-space family and is the
method implemented by Durbyn’s [`bats`] function.

This page summarizes the core equations, highlights limitations (and why
TBATS was proposed in the paper), and shows how to use the Julia interface.

---

## 1. Box–Cox transformation

Each BATS model may apply a Box–Cox transformation to the observed series,
which stabilizes variance prior to modeling:

```math
y_t^{(\omega)} =
\begin{cases}
\dfrac{y_t^\omega - 1}{\omega}, & \omega \neq 0, \\
\ln y_t, & \omega = 0 .
\end{cases}
```

The parameter ``\omega`` (often denoted ``\lambda`` in code) is estimated
within the automated model search when the user permits Box–Cox transforms.

---

## 2. BATS state-space formulation

After optional transformation, BATS is written in innovations form with an
ARMA error process.

### Observation equation

```math
y_t^{(\omega)} = \ell_{t-1} + \phi b_{t-1} + \sum_i s_{i,t-1} + d_t,
```

where ``\ell_t`` is the level, ``b_t`` the trend, ``\phi`` the damping
parameter, ``s_{i,t}`` the seasonal state for seasonal period ``m_i``, and
``d_t`` the ARMA error term.

### State equations

Level and trend:

```math
\ell_t = \ell_{t-1} + \phi b_{t-1} + \alpha d_t, \qquad
b_t = \phi b_{t-1} + \beta d_t.
```

Additive seasonality for each seasonal block ``i`` (normalized form):

```math
s_{i,t} = -\sum_{j=1}^{m_i-1} s_{i,t-j} + \gamma_i d_t.
```

### ARMA error component

```math
d_t = \varepsilon_t + \sum_{k=1}^p \varphi_k \varepsilon_{t-k}
      + \sum_{\ell=1}^q \theta_\ell d_{t-\ell},
\qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2).
```

Combining these pieces yields the descriptor `BATS(ω, {p,q}, φ, {m₁,…,m_J})`
that Durbyn prints for each fitted model.

---

## 3. Limitations and relation to TBATS

In the original paper, TBATS was introduced to address several BATS
limitations:

- Seasonal periods must be integers, and each requires storing ``m_i`` state
  components, which becomes expensive for very long cycles.
- Non-integer or dual-calendar seasonalities (e.g., Hijri and Gregorian)
  cannot be represented exactly.

TBATS replaces the seasonal states with Fourier (trigonometric) terms to
overcome those issues. **Durbyn provides both BATS and TBATS implementations.**
The `bats` function corresponds strictly to the BATS formulation above for
integer seasonal periods, while `tbats` supports non-integer periods and
more efficient handling of long seasonal cycles via Fourier representation.

!!! note "TBATS documentation"
    For detailed information about TBATS, including non-integer seasonal periods,
    dual calendar effects, and computational efficiency, see the
    [TBATS documentation](tbats.md).

---

## 4. Usage in Durbyn

### Basic example

```julia
using Durbyn

# Hourly demand with weekly (24*7) and yearly (24*365) seasonality
m = [168, 8760]
fit = bats(load, m; use_box_cox = true, use_arma_errors = true)

println(string(fit))
fc = forecast(fit; h = 168)
```

### Key keyword arguments

- `use_box_cox`, `use_trend`, `use_damped_trend`: `Bool`, `Vector{Bool}`, or
  `nothing` to try both options; the best combination is chosen using AIC.
- `use_arma_errors`: toggles fitting an ARMA(p, q) model to the residuals via
  [`auto_arima`]; if the selected ARMA orders are zero, the pure
  exponential-smoothing state-space model is retained.
- `bc_lower`, `bc_upper`: bounds for the Box–Cox search when enabled.
- `biasadj`: apply bias correction during inverse Box–Cox transformation.
- `model`: pass a previous `BATSModel` to refit the same structure to new
  data without re-running the full model selection process.

The convenience method `bats(y, m::Int; kwargs...)` simply wraps the vector
interface (`[m]`), making single-season calls ergonomic.

---

## 5. Reference

- De Livera, A.M., Hyndman, R.J., & Snyder, R.D. (2011). [*Forecasting time
  series with complex seasonal patterns using exponential smoothing.*](https://www.tandfonline.com/doi/abs/10.1198/jasa.2011.tm09771)
  Journal of the American Statistical Association, 106(496), 1513–1527.
