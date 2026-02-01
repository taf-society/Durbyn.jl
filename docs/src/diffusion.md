# Diffusion Models: Technology Adoption and Market Penetration Forecasting

Diffusion models are specialized forecasting tools designed to predict **technology adoption**, **market penetration**, and **product life cycle dynamics**. Unlike traditional time series methods that model temporal autocorrelation, diffusion models capture the underlying social dynamics of how innovations spread through a population.

Durbyn implements four classic diffusion models:

| Model | Best For | Key Feature |
|-------|----------|-------------|
| **Bass** | New product forecasting | Separates innovators from imitators |
| **Gompertz** | Biological/market growth | Asymmetric S-curve with early inflection |
| **GSGompertz** | Flexible adoption patterns | Generalized shape control |
| **Weibull** | Reliability/lifetime analysis | Distribution-based flexibility |

---

## Theory and Background

Diffusion of innovations theory (Rogers, 1962) describes how new ideas and technologies spread through social systems. The adoption process typically follows an S-curve pattern:

1. **Introduction phase**: Few early adopters (innovators)
2. **Growth phase**: Rapid adoption as word-of-mouth spreads
3. **Maturity phase**: Market saturation approaches
4. **Saturation**: Adoption levels off at market potential

Diffusion models mathematically capture this dynamic by modeling cumulative adoption as a function of time and market potential.

---

## Model Types

### 1. Bass Diffusion Model

The **Bass model** (Bass, 1969) is the most widely used diffusion model in marketing science. It explicitly models two adoption mechanisms:

- **Innovation effect** (coefficient `p`): External influence from mass media, advertising
- **Imitation effect** (coefficient `q`): Internal influence from word-of-mouth, social contagion

#### Mathematical Formulation

**Cumulative adoption:**
```math
A_t = m \cdot \frac{1 - e^{-(p+q)t}}{1 + \frac{q}{p} e^{-(p+q)t}}
```

**Adoption per period (first differences):**
```math
a_t = A_t - A_{t-1}
```

**Hazard rate interpretation:**
```math
\frac{f(t)}{1-F(t)} = p + q \cdot F(t)
```

where the probability of adoption at time `t` given non-adoption is linear in cumulative adoption.

#### Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Market potential | `m` | Total eventual adopters | > max(cumsum(y)) |
| Innovation coefficient | `p` | External influence rate | 0.001 - 0.1 |
| Imitation coefficient | `q` | Internal influence rate | 0.1 - 0.5 |

**Key insight**: The ratio `q/p` determines the shape of the adoption curve. Higher ratios produce more pronounced peaks.

#### Decomposition

The Bass model uniquely provides decomposition of adoption into innovator and imitator components:

```math
\text{Innovators}_t = p \cdot (m - A_t)
```
```math
\text{Imitators}_t = a_t - \text{Innovators}_t
```

---

### 2. Gompertz Growth Curve

The **Gompertz model** (Gompertz, 1825) produces an asymmetric S-curve with the inflection point occurring earlier than the midpoint. Originally developed for mortality modeling, it's widely used for biological and market growth phenomena.

#### Mathematical Formulation

**Cumulative adoption:**
```math
A_t = m \cdot e^{-a \cdot e^{-b \cdot t}}
```

**Adoption per period:**
```math
a_t = m \cdot a \cdot b \cdot e^{-bt} \cdot e^{-a \cdot e^{-bt}}
```

#### Parameters

| Parameter | Symbol | Description | Effect |
|-----------|--------|-------------|--------|
| Market potential | `m` | Asymptotic maximum | Upper bound of curve |
| Displacement | `a` | X-axis translation | Shifts curve horizontally |
| Growth rate | `b` | Steepness parameter | Controls adoption speed |

**Key property**: The inflection point occurs at `t* = ln(a)/b` when cumulative adoption reaches `m/e ≈ 0.368m`.

---

### 3. Gamma/Shifted Gompertz (GSGompertz)

The **Gamma/Shifted Gompertz** model (Bemmaor, 1994) generalizes the Bass model by allowing heterogeneity in the imitation coefficient across adopters. This produces more flexible curve shapes.

#### Mathematical Formulation

**Cumulative adoption:**
```math
A_t = m \cdot (1 - e^{-bt})(1 + a \cdot e^{-bt})^{-c}
```

#### Parameters

| Parameter | Symbol | Description | Special Cases |
|-----------|--------|-------------|---------------|
| Market potential | `m` | Total eventual adopters | - |
| Shape | `a` | Heterogeneity parameter | `a = p/q` (Bass relation) |
| Scale | `b` | Rate parameter | `b = p + q` (Bass relation) |
| Shift | `c` | Distribution parameter | `c = 1` gives Bass-like curve |

**Connection to Bass**: When `c = 1`, the GSGompertz reduces to a form equivalent to the Bass model with `a = p/q` and `b = p + q`.

---

### 4. Weibull Distribution Model

The **Weibull model** (Sharif & Islam, 1980) uses the Weibull cumulative distribution function for adoption modeling. It's particularly useful when adoption follows reliability/lifetime distribution patterns.

#### Mathematical Formulation

**Cumulative adoption:**
```math
A_t = m \cdot \left(1 - e^{-(t/a)^b}\right)
```

**Adoption per period:**
```math
a_t = \frac{m \cdot b}{a} \cdot \left(\frac{t}{a}\right)^{b-1} \cdot e^{-(t/a)^b}
```

#### Parameters

| Parameter | Symbol | Description | Shape Effect |
|-----------|--------|-------------|--------------|
| Market potential | `m` | Total eventual adopters | Upper bound |
| Scale | `a` | Characteristic life | Stretches/compresses time |
| Shape | `b` | Shape parameter | `b < 1`: decreasing hazard; `b = 1`: exponential; `b > 1`: increasing hazard |

---

## Parameter Initialization

Durbyn provides two initialization strategies for optimization:

### Linearize (Default)

Uses analytical methods to compute starting values:

- **Bass**: Linear regression on `y ~ Y + Y²` where `Y = cumsum(y)`, solving the resulting quadratic
- **Gompertz**: Jukic et al. (2004) three-point method with Bass optimization for `m`
- **GSGompertz**: Derives from Bass parameters using Bemmaor's conversion formulas
- **Weibull**: Median-ranked OLS using Bernard's approximation

### Preset

Uses fixed starting values (useful when analytical methods fail):

| Model | Preset Values |
|-------|---------------|
| Bass | `m=0.5, p=0.5, q=0.5` (scaled by `10*sum(y)` for `m`) |
| Gompertz | `m=1, a=1, b=1` (scaled) |
| GSGompertz | `m=0.5, a=0.5, b=0.5, c=0.5` (scaled) |
| Weibull | `m=0.5, a=0.5, b=0.5` (scaled) |

---

## Usage

### Basic Fitting

```julia
using Durbyn
using Durbyn.Diffusion

# Sample adoption data (units per period)
y = [5, 10, 25, 45, 70, 85, 75, 50, 30, 15]

# Fit Bass diffusion model (default)
fit = diffusion(y)
# Or equivalently:
fit = diffusion(y, model_type=Bass)

# View results
println(fit)
```

**Output:**
```
Diffusion Model (Bass)
─────────────────────────────
Observations: 10
Parameters:
  m: 485.123456
  p: 0.032145
  q: 0.387654
MSE: 12.345678
Loss function: L2
Optimized on: cumulative
```

### Fitting Different Model Types

```julia
# Gompertz model
fit_gomp = diffusion(y, model_type=Gompertz)

# Gamma/Shifted Gompertz
fit_gsg = diffusion(y, model_type=GSGompertz)

# Weibull model
fit_weib = diffusion(y, model_type=Weibull)
```

### Forecasting

```julia
# Fit model
fit = diffusion(y, model_type=Bass)

# Generate 5-period forecast
fc = forecast(fit, h=5)

# Access results
fc.mean        # Point forecasts
fc.lower[1]    # 80% lower prediction bounds
fc.lower[2]    # 95% lower prediction bounds
fc.upper[1]    # 80% upper prediction bounds
fc.upper[2]    # 95% upper prediction bounds

# Plot forecast
plot(fc)
```

### Prediction at Specific Time Points

```julia
# Predict adoption for periods 1-20
fit = diffusion(y, model_type=Bass)
pred = predict(fit, 1:20)

pred.adoption      # Adoption per period
pred.cumulative    # Cumulative adoption
```

### Fixed Parameters

Fix specific parameters while estimating others:

```julia
# Fix market potential at 500, estimate p and q
fit = diffusion(y, model_type=Bass, w=(m=500.0, p=nothing, q=nothing))

# Fix innovation coefficient, estimate m and q
fit = diffusion(y, model_type=Bass, w=(m=nothing, p=0.03, q=nothing))

# Fix all parameters (compute fit only)
fit = diffusion(y, model_type=Bass, w=(m=500.0, p=0.03, q=0.38))
```

### Custom Initial Values

```julia
# Provide numeric initial values
fit = diffusion(y, model_type=Bass, initpar=[500.0, 0.03, 0.38])

# Use preset initialization
fit = diffusion(y, model_type=Bass, initpar="preset")
```

### Loss Functions

```julia
# Mean Squared Error (default)
fit_mse = diffusion(y, loss=2)

# Mean Absolute Error (more robust to outliers)
fit_mae = diffusion(y, loss=1)
```

### Handling Leading Zeros

```julia
# Data with leading zeros (common in early adoption)
y = [0, 0, 0, 5, 10, 25, 45, 70, 85, 75]

# Remove leading zeros before fitting (default)
fit = diffusion(y, cleanlead=true)
println("Offset: $(fit.offset)")  # Number of zeros removed

# Keep leading zeros
fit = diffusion(y, cleanlead=false)
```

### Optimization Options

```julia
# Custom optimization settings
fit = diffusion(y,
    model_type=Bass,
    method="L-BFGS-B",     # Optimization algorithm
    maxiter=1000,          # Maximum iterations
    mscal=true,            # Scale market potential for stability
    cumulative=true        # Optimize on cumulative values
)
```

---

## Accessing Fitted Results

The `DiffusionFit` struct contains comprehensive model information:

```julia
fit = diffusion(y, model_type=Bass)

# Model type
fit.model_type        # Bass, Gompertz, GSGompertz, or Weibull

# Fitted parameters
fit.params            # NamedTuple: (m=..., p=..., q=...) for Bass
fit.params.m          # Market potential
fit.params.p          # Innovation coefficient (Bass)
fit.params.q          # Imitation coefficient (Bass)

# Fitted values
fit.fitted            # Fitted adoption per period
fit.cumulative        # Fitted cumulative adoption
fit.residuals         # Residuals (actual - fitted)

# Diagnostics
fit.mse               # Mean squared error
fit.loss              # Loss function used (1=MAE, 2=MSE)
fit.optim_cumulative  # Whether optimized on cumulative

# Data
fit.y                 # Cleaned data (after removing leading zeros)
fit.y_original        # Original data
fit.offset            # Number of leading zeros removed

# Initialization
fit.init_params       # Initial parameters before optimization
```

---

## Bass Model Decomposition

The Bass model uniquely provides innovator/imitator decomposition:

```julia
using Durbyn.Diffusion

# Generate Bass curve with decomposition
n = 20
m, p, q = 1000.0, 0.03, 0.38

curve = bass_curve(n, m, p, q)

curve.cumulative    # Cumulative adoption
curve.adoption      # Adoption per period
curve.innovators    # Innovation component
curve.imitators     # Imitation component

# Verify: adoption = innovators + imitators
all(curve.adoption .≈ curve.innovators .+ curve.imitators)  # true
```

---

## Curve Generation Functions

Generate theoretical diffusion curves for visualization or analysis:

```julia
using Durbyn.Diffusion

# Bass curve
curve = bass_curve(50, 1000.0, 0.03, 0.38)

# Gompertz curve
curve = gompertz_curve(50, 1000.0, 5.0, 0.3)

# GSGompertz curve
curve = gsgompertz_curve(50, 1000.0, 0.08, 0.41, 1.0)

# Weibull curve
curve = weibull_curve(50, 1000.0, 15.0, 2.5)

# All return NamedTuple with:
# - cumulative: Vector of cumulative adoption
# - adoption: Vector of adoption per period
# - (Bass only) innovators, imitators: Decomposition
```

---

## Model Comparison Example

```julia
using Durbyn
using Durbyn.Diffusion

# Historical smartphone adoption data (millions of units)
y = [2, 8, 25, 55, 95, 140, 175, 195, 205, 210, 212, 213]

# Fit all four models
models = Dict(
    "Bass" => diffusion(y, model_type=Bass),
    "Gompertz" => diffusion(y, model_type=Gompertz),
    "GSGompertz" => diffusion(y, model_type=GSGompertz),
    "Weibull" => diffusion(y, model_type=Weibull)
)

# Compare MSE
for (name, fit) in models
    println("$name MSE: $(round(fit.mse, digits=2))")
end

# Forecast with best model
best_model = argmin(Dict(k => v.mse for (k, v) in models))
fc = forecast(models[best_model], h=5)
println("\nBest model: $best_model")
println("Forecast: $(round.(fc.mean, digits=1))")
```

---

## Use Cases

### New Product Launch Forecasting

```julia
# Early sales data for new product (units/month)
sales = [100, 250, 580, 1200, 2100, 3500, 5200, 6800]

# Fit Bass model
fit = diffusion(sales, model_type=Bass)

# Forecast remaining product lifecycle
fc = forecast(fit, h=24)

# Key insights
println("Estimated market potential: $(round(fit.params.m)) units")
println("Innovation coefficient (p): $(round(fit.params.p, digits=4))")
println("Imitation coefficient (q): $(round(fit.params.q, digits=4))")
println("Peak period (estimated): period $(argmax(fc.mean) + length(sales))")
```

### Technology Adoption Analysis

```julia
# Annual internet user growth (millions)
users = [16, 36, 70, 147, 248, 361, 513, 719, 1018, 1319]

# Compare models
fit_bass = diffusion(users, model_type=Bass)
fit_gomp = diffusion(users, model_type=Gompertz)

# Bass provides adoption dynamics insight
println("Innovation effect (p): $(round(fit_bass.params.p, digits=4))")
println("Imitation effect (q): $(round(fit_bass.params.q, digits=4))")
println("Word-of-mouth multiplier (q/p): $(round(fit_bass.params.q/fit_bass.params.p, digits=1))x")
```

### Market Saturation Analysis

```julia
# EV adoption data
ev_sales = [15, 45, 120, 280, 550, 920, 1400, 1950]

fit = diffusion(ev_sales, model_type=Bass)

# Current penetration
current_cumulative = sum(ev_sales)
market_potential = fit.params.m
penetration = current_cumulative / market_potential * 100

println("Market potential: $(round(market_potential)) units")
println("Current penetration: $(round(penetration, digits=1))%")
println("Remaining market: $(round(market_potential - current_cumulative)) units")
```

---

## API Reference

### Main Functions

```julia
# Fit diffusion model
diffusion(y; model_type=Bass, kwargs...) -> DiffusionFit
fit_diffusion(y; model_type=Bass, kwargs...) -> DiffusionFit

# Generate forecast
forecast(fit::DiffusionFit; h::Int, level=[80, 95]) -> Forecast

# Predict at specific times
predict(fit::DiffusionFit, t::AbstractVector) -> NamedTuple
```

### Curve Generation

```julia
bass_curve(n, m, p, q) -> NamedTuple
gompertz_curve(n, m, a, b) -> NamedTuple
gsgompertz_curve(n, m, a, b, c) -> NamedTuple
weibull_curve(n, m, a, b) -> NamedTuple
```

### Initialization Functions

```julia
bass_init(y) -> NamedTuple
gompertz_init(y; kwargs...) -> NamedTuple
gsgompertz_init(y; kwargs...) -> NamedTuple
weibull_init(y) -> NamedTuple
```

### fit_diffusion Keyword Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_type` | `DiffusionModelType` | `Bass` | Model to fit |
| `cleanlead` | `Bool` | `true` | Remove leading zeros |
| `w` | `NamedTuple` or `Nothing` | `nothing` | Fixed parameters |
| `loss` | `Int` | `2` | Loss function (1=MAE, 2=MSE) |
| `cumulative` | `Bool` | `true` | Optimize on cumulative values |
| `mscal` | `Bool` | `true` | Scale market parameter |
| `maxiter` | `Int` | `500` | Maximum iterations |
| `method` | `String` | `"L-BFGS-B"` | Optimization method |
| `initpar` | `String` or `Vector` | `"linearize"` | Initialization method |

---

## References

- Bass, F.M. (1969). A new product growth for model consumer durables. *Management Science*, 15(5), 215-227.

- Bemmaor, A.C. (1994). Modeling the diffusion of new durable goods: Word-of-mouth effect versus consumer heterogeneity. In G. Laurent et al. (Eds.), *Research Traditions in Marketing*. Kluwer Academic Publishers.

- Gompertz, B. (1825). On the nature of the function expressive of the law of human mortality. *Philosophical Transactions of the Royal Society*, 115, 513-583.

- Jukic, D., Kralik, G., & Scitovski, R. (2004). Least-squares fitting Gompertz curve. *Journal of Computational and Applied Mathematics*, 169, 359-375.

- Rogers, E.M. (1962). *Diffusion of Innovations*. Free Press.

- Sharif, M.N. & Islam, M.N. (1980). The Weibull distribution as a general model for forecasting technological change. *Technological Forecasting and Social Change*, 18(3), 247-256.
