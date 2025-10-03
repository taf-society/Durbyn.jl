
# Intermittent Demand Forecasting

## Overview

Intermittent demand occurs in time series with many zero values and occasional non-zero spikes, commonly found in spare parts inventory, specialty products, and slow-moving items. Traditional forecasting methods like ARIMA or exponential smoothing perform poorly on such data due to the preponderance of zeros and the sporadic nature of demand occurrences.

The Croston family of methods addresses intermittent demand by decomposing the forecasting problem into separate components: demand **size** when it occurs and demand **timing** (intervals or probabilities). This decomposition enables more accurate modeling of the underlying demand process.

## Croston's Method

Croston's method models intermittent demand by maintaining two exponentially smoothed states: demand size ``z_t`` when demand occurs, and inter-demand intervals ``x_t``.

**References:**
- Croston, J. (1972). "Forecasting and stock control for intermittent demands". *Operational Research Quarterly*, 23(3), 289-303.
- Shenstone, L., and Hyndman, R.J. (2005). "Stochastic models underlying Croston's method for intermittent demand forecasting". *Journal of Forecasting*, 24, 389-402.

### Notation

- ``y_t``: observed demand at time ``t`` (often zero)
- ``z_t``: non-zero demand size (observed only when ``y_t > 0``)
- ``x_t``: inter-demand interval (time between non-zero demands)
- ``\hat{z}_t, \hat{x}_t``: exponentially smoothed estimates
- ``\alpha_z, \alpha_x \in (0,1]``: smoothing parameters for size and interval
- ``q``: number of non-zero demands observed up to time ``t``

### Update Equations

The exponential smoothing updates occur only when demand is observed (``y_t > 0``):

```math
\hat{z}_q = \alpha_z z_t + (1-\alpha_z)\hat{z}_{q-1}
```

```math
\hat{x}_q = \alpha_x x_t + (1-\alpha_x)\hat{x}_{q-1}
```

where ``x_t`` is the time since the previous non-zero demand.

### Forecast

The Croston forecast represents the expected demand rate per period:

```math
\hat{y}_{t+h} = \frac{\hat{z}_q}{\hat{x}_q}, \quad h \geq 1
```

This forecast is constant for all future periods (flat forecast profile).

### Implementation Types

The implementation handles four distinct cases based on the data characteristics:

1. **CrostonOne**: All demands are zero - returns zero forecast
2. **CrostonTwo**: Only one non-zero demand and one interval - returns constant forecast
3. **CrostonThree**: Insufficient data (≤1 demand or ≤1 interval) - returns NaN
4. **CrostonFour**: Standard case with multiple demands - applies full Croston method

### Data Structures

#### `CrostonFit`

Stores the fitted Croston model containing:
- `modely`: Simple exponential smoothing model for demand sizes
- `modelp`: Simple exponential smoothing model for inter-demand intervals
- `type`: One of four CrostonType enum values indicating the fitting approach
- `x`: Original demand series
- `y`: Non-zero demands only
- `tt`: Inter-demand intervals
- `m`: Seasonal period (typically 1 for non-seasonal intermittent demand)

#### `CrostonForecast`

Stores forecast output containing:
- `mean`: Forecast values (demand rate per period)
- `model`: The underlying `CrostonFit` object
- `method`: Description string ("Croston's Method")
- `m`: Seasonal period

### Functions

#### `croston(y, m; alpha, options)`

Fits a Croston model to intermittent demand data.

**Arguments:**
- `y::AbstractArray`: Demand time series (may contain zeros)
- `m::Int`: Seasonal period (default: 1)
- `alpha::Union{Float64,Bool,Nothing}`: Smoothing parameter (nothing for automatic optimization)
- `options::NelderMeadOptions`: Optimization settings for parameter estimation

**Returns:**
- `CrostonFit`: Fitted model object

**Example:**
```julia
using Durbyn.ExponentialSmoothing

# Intermittent demand data
demand = [0, 0, 5, 0, 0, 3, 0, 0, 0, 7, 0, 0, 4, 0, 0]

# Fit Croston model with automatic parameter optimization
fit = croston(demand)

# Fit with fixed smoothing parameter
fit_fixed = croston(demand, alpha=0.1)
```

#### `forecast(object::CrostonFit, h::Int)`

Generates forecasts from a fitted Croston model.

**Arguments:**
- `object::CrostonFit`: Fitted Croston model
- `h::Int`: Forecast horizon (number of periods ahead)

**Returns:**
- `CrostonForecast`: Forecast object containing mean forecasts

**Example:**
```julia
# Generate 12-period-ahead forecast
fc = forecast(fit, 12)
println(fc.mean)  # Access forecast values
```

#### `fitted(object::CrostonFit)`

Computes in-sample fitted values using one-step-ahead forecasts.

**Arguments:**
- `object::CrostonFit`: Fitted Croston model

**Returns:**
- `Vector`: Fitted values (same length as original series)

**Note:** The first value is NaN as no forecast is available for the first observation.

**Example:**
```julia
fitted_vals = fitted(fit)
residuals = demand .- fitted_vals
```

#### `plot(forecast::CrostonForecast; show_fitted=false)`

Visualizes the forecast with optional fitted values.

**Arguments:**
- `forecast::CrostonForecast`: Forecast object to plot
- `show_fitted::Bool`: Whether to display in-sample fitted values (default: false)

**Returns:**
- Plots.jl plot object

**Example:**
```julia
# Plot forecast only
plot(fc)

# Plot forecast with fitted values
plot(fc, show_fitted=true)
```

## Syntetos-Boylan Approximation (SBA)

The SBA method applies a bias correction to Croston's forecast. Syntetos and Boylan (2005) showed that Croston's method produces biased forecasts and proposed the following correction:

```math
\hat{y}_{t+h} = \left(1 - \frac{\alpha_x}{2}\right) \frac{\hat{z}_q}{\hat{x}_q}
```

This correction reduces the upward bias inherent in the original Croston method.

## Teunter-Syntetos-Babai (TSB) Method

The TSB method reformulates the intermittent demand problem by modeling demand occurrence probability ``p_t`` and demand size ``z_t`` separately. This method provides an alternative theoretical framework but is not currently implemented in the `IntermittentDemand` module.

### Theoretical Framework

The probability of demand is updated every period:
```math
\hat{p}_t = \alpha_p d_t + (1-\alpha_p)\hat{p}_{t-1}
```

where ``d_t = 1`` if ``y_t > 0``, and ``d_t = 0`` otherwise.

The demand size is updated only when demand occurs:
```math
\hat{z}_q = \alpha_z z_t + (1-\alpha_z)\hat{z}_{q-1}
```

### Forecast

The TSB forecast is:
```math
\hat{y}_{t+h} = \hat{p}_t \cdot \hat{z}_q
```

## Shale-Boylan-Johnston (SBJ) Method

The SBJ method provides an alternative bias correction to Croston's method, particularly suited for Poisson demand arrivals. The correction factor is derived from the theoretical properties of the demand process.

## Optimization and Loss Functions

### Traditional Loss Functions

Classical forecasting metrics often perform poorly for intermittent demand:

- **Mean Squared Error (MSE)**: ``\displaystyle \frac{1}{n}\sum_{t=1}^{n}(y_t-\hat{y}_t)^2``
- **Mean Absolute Error (MAE)**: ``\displaystyle \frac{1}{n}\sum_{t=1}^{n}|y_t-\hat{y}_t|``

These metrics compare forecasts against predominantly zero actual values, leading to downward-biased parameter estimates.

### Rate-Based Loss Functions

Since Croston-type methods produce rate forecasts (expected demand per period), Kourentzes (2014) demonstrated that rate-based loss functions yield superior results:

**Rate residual at time ``t``:**
```math
r_t = \hat{y}_t - \frac{1}{t}\sum_{j=1}^{t} y_j
```

**Mean Absolute Rate error (MAR):**
```math
\text{MAR} = \frac{1}{n}\sum_{t=1}^{n} |r_t|
```

**Mean Squared Rate error (MSR):**
```math
\text{MSR} = \frac{1}{n}\sum_{t=1}^{n} r_t^2
```

### Empirical Findings

Kourentzes (2014) established through extensive simulation that:

1. **MAR and MSR perform equivalently** and both substantially outperform MSE/MAE
2. **Separate smoothing parameters** (``\alpha_z \neq \alpha_x``) improve performance for Croston variants
3. **Initial state optimization** enhances accuracy, particularly for short series
4. **Parameter bounds** should allow values up to 1.0; restrictive upper bounds (e.g., 0.3) can degrade performance
5. **Small smoothing parameters** (typically 0.05-0.2) emerge naturally with proper optimization

## Model Selection

Kourentzes (2014) found that:
- **Simple Croston variants** often perform competitively with complex model selection schemes
- **Bias-corrected methods** (SBA, SBJ) generally outperform the classical Croston method
- **Focus on proper optimization** (using rate-based losses with separate parameters) matters more than complex model selection

## Implementation Details

The `IntermittentDemand` module provides three main functions that implement the Kourentzes (2014) recommendations:

### Available Methods
- `croston_classic()`: Classical Croston method
- `croston_sba()`: Syntetos-Boylan Approximation
- `croston_sbj()`: Shale-Boylan-Johnston bias correction

### Key Parameters

All methods support the following parameters aligned with Kourentzes (2014) findings:

- **`cost_metric`**: Loss function for optimization
  - `"mar"` (recommended): Mean Absolute Rate error
  - `"msr"` (recommended): Mean Squared Rate error
  - `"mae"`: Mean Absolute Error (classical)
  - `"mse"`: Mean Squared Error (classical)

- **`number_of_params`**: Number of smoothing parameters
  - `1`: Single parameter for both size and interval
  - `2` (recommended): Separate parameters (``\alpha_z \neq \alpha_x``)

- **`optimize_init`**: Whether to optimize initial states
  - `true` (recommended): Optimize initial values
  - `false`: Use heuristic initialization

- **`init_strategy`**: Initialization method
  - `"mean"` (default): Use mean of non-zero values and intervals
  - `"naive"`: Use first observed values

### Implementation Notes

1. **Rate-based optimization**: MAR and MSR losses are implemented as recommended
2. **Separate smoothing parameters**: Supported via `number_of_params = 2`
3. **Parameter bounds**: Allow values up to 1.0 (no restrictive caps)
4. **Bias corrections**: SBA and SBJ corrections are built into the respective methods

## Forecasting in Julia

```julia
using Durbyn
using Durbyn.IntermittentDemand

# Intermittent demand data
data = [6, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
        0, 0, 0, 0, 0]

# Classical Croston method (using recommended MAR cost metric)
fit_croston = croston_classic(data, cost_metric = "mar")
fc_croston = forecast(fit_croston, h = 12)

# Syntetos-Boylan Approximation with separate smoothing parameters
fit_sba = croston_sba(data, cost_metric = "mar", number_of_params = 2)
fc_sba = forecast(fit_sba, h = 12)

# Shale-Boylan-Johnston method with initial state optimization
fit_sbj = croston_sbj(data, cost_metric = "mar", optimize_init = true)
fc_sbj = forecast(fit_sbj, h = 12)

# Alternative cost metrics (classical - use with caution)
fit_mse = croston_classic(data, cost_metric = "mse")  # Traditional MSE
fit_mae = croston_classic(data, cost_metric = "mae")  # Traditional MAE
fit_msr = croston_classic(data, cost_metric = "msr")  # Mean Squared Rate

# Visualization
plot(fc_croston, show_fitted = true)
plot(fc_sba, show_fitted = true)
plot(fc_sbj, show_fitted = true)

# Model diagnostics
residuals(fit_croston)
fitted(fit_croston)

# Model comparison
println("Croston weights: ", fit_croston.weights)
println("SBA weights: ", fit_sba.weights)
println("SBJ weights: ", fit_sbj.weights)
```

## Reference

Kourentzes, N. (2014). *On Intermittent Demand Model Optimisation and Selection*. International Journal of Production Economics, 156: 180-190.
