<!-- HERO -->
<div class="hero">
  <div class="hero-content">
    <img class="logo" src="assets/logo.svg" alt="Durbyn.jl logo"/>
    <h1>Durbyn.jl</h1>
    <p>Modern, pragmatic time‑series forecasting in Julia — inspired by the classics, engineered for today.</p>
    <div class="cta-row">
      <a class="btn primary" href="quickstart.html">Quick Start</a>
      <a class="btn" href="api.html">API Reference</a>
      <a class="btn ghost" href="https://github.com/taf-society/Durbyn.jl">GitHub ↗</a>
    </div>
    <div class="badges">
      <img alt="docs-dev" src="https://img.shields.io/badge/docs-dev-blue"/>
      <img alt="license" src="https://img.shields.io/github/license/taf-society/Durbyn.jl"/>
      <img alt="stars" src="https://img.shields.io/github/stars/taf-society/Durbyn.jl?style=social"/>
    </div>
  </div>
</div>

---

## Why Durbyn?

<div class="feature-grid">
  <div class="card">
    <h3>Classic Methods, Clean API</h3>
    <p>From ETS to ARIMA/ARARMA and intermittent‑demand models — approachable names, sensible defaults.</p>
  </div>
  <div class="card">
    <h3>Composable</h3>
    <p>Models and forecasts are simple Julia structs; plug into your data/ML pipeline with ease.</p>
  </div>
  <div class="card">
    <h3>Fast</h3>
    <p>Julia performance with type stability and vectorized internals where it counts.</p>
  </div>
</div>

> **Heads up.** This site documents the development version. See the <a href="../stable/">stable docs</a> for the latest release.

---

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/taf-society/Durbyn.jl")
# after registering & tagging:
# Pkg.add("Durbyn")
```

---

## Quick peek

```julia
using Durbyn
# toy monthly airline passengers series
y = air_passengers()

# Exponential Smoothing (auto-config)
fit = Durbyn.ExponentialSmoothing.ets(y, 12, "ZZZ")
fc  = forecast(fit, h = 12)

println(fc.mean)
```

!!! tip "Plotting"
    Pair with your favorite plotting backend. For example, `Makie.jl` or `Plots.jl`:

    ```julia
    using Plots
    plot([y; fill(NaN, 12)], label = "observed")
    plot!(length(y) .+ (1:12), fc.mean, ribbon = (fc.lower, fc.upper), label = "forecast")
    ```

---

## Feature highlights

- **ETS family**: additive/multiplicative trend and seasonality
- **ARIMA / ARAR / ARARMA** baselines
- **Intermittent demand** helpers
- Strong **type docs** and **API reference** generated automatically

---

## Learn more

- Start with the <a href="quickstart.html">Quick Start</a>
- Browse the <a href="api.html">API Reference</a>
- Explore model pages: <a href="expsmoothing.html">ETS</a>, <a href="arima.html">ARIMA</a>, <a href="ararma.html">ARAR/ARARMA</a>

---

## Acknowledgements

Durbyn builds on decades of time‑series literature and Julia community practices. ♥
