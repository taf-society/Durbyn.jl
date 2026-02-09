"""
    croston_classic(x::AbstractVector; init_strategy::String = "mean", number_of_params::Int = 2,
                    cost_metric::String = "mar", optimize_init::Bool = true, rm_missing::Bool = false)
        -> IntermittentDemandCrostonFit

Forecast intermittent demand using the classical Croston (1972) method.

# Arguments

- `x::AbstractVector`: Input time series vector representing intermittent demand (including zeros and possibly `missing` values).
  Must contain at least two non-zero values.

- `init_strategy::String`: Initialization strategy for smoothing. Default is `"mean"`.
    - `"mean"`: Initialize using mean of non-zero demands and intervals (recommended)
    - `"naive"`: Initialize using first observed demand and interval

- `number_of_params::Int`: Number of smoothing parameters to optimize. Default is `2`.
    - `1`: Single smoothing parameter for both demand size and intervals
    - `2`: Separate smoothing parameters (recommended for better accuracy)

- `cost_metric::String`: Optimization cost function for parameter tuning. Default is `"mar"`.
    - `"mar"`: Mean Absolute Rate (recommended by Kourentzes 2014)
    - `"msr"`: Mean Squared Rate (recommended by Kourentzes 2014)
    - `"mae"`: Mean Absolute Error (classical metric)
    - `"mse"`: Mean Squared Error (classical metric)

- `optimize_init::Bool`: If `true`, initial values are optimized alongside smoothing parameters.
  Default is `true`. Recommended for short series.

- `rm_missing::Bool`: If `true`, `missing` values are removed from `x` before modeling. Default is `false`.

# Returns

`IntermittentDemandCrostonFit` - A struct containing:
- Optimized smoothing weights for demand and interval
- Initial values for the smoothing processes
- Model method identifier
- Original input data

# Description

The classical Croston method is designed for intermittent demand forecasting where demand occurs
sporadically with many zero values. It decomposes the time series into two separate processes:

1. **Demand size (z)**: The non-zero demand magnitudes, smoothed with exponential smoothing
2. **Demand interval (x)**: The time between non-zero demands, also smoothed with exponential smoothing

The forecast is computed as the ratio: `forecast = z / x`

This method is particularly useful for:
- Spare parts inventory management
- Slow-moving items with irregular demand
- Time series with >50% zero values

# Examples

```julia
using Durbyn.IntermittentDemand

# Intermittent demand data with many zeros
demand = [6, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0]

# Fit with default parameters (recommended by Kourentzes 2014)
fit = croston_classic(demand)

# Generate 12-period forecast
fc = forecast(fit, h = 12)
println("Forecast: ", fc.mean)

# Extract fitted values and residuals
fitted_values = fitted(fit)
resids = residuals(fit)

# Custom parameters for specific use cases
fit_custom = croston_classic(demand;
    init_strategy = "naive",
    number_of_params = 1,
    cost_metric = "mse",
    optimize_init = false
)
```

# Grammar Interface

For declarative model specification with grouped data support:

```julia
using Durbyn

spec = CrostonSpec(@formula(demand = croston(method="classic")))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)
```

# See Also

- [`croston_sba`](@ref) - Bias-corrected variant (recommended for most applications)
- [`croston_sbj`](@ref) - Alternative bias correction
- [`CrostonSpec`](@ref) - Grammar interface for model specification
- [`forecast`](@ref) - Generate forecasts from fitted model
- [`fitted`](@ref) - Extract in-sample fitted values
- [`residuals`](@ref) - Compute model residuals

# References

Croston, J. D. (1972). *Forecasting and stock control for intermittent demands*.
Operational Research Quarterly, 23(3), 289-303.

Kourentzes, N. (2014). *On intermittent demand model optimisation and selection*.
International Journal of Production Economics, 156, 180–190.
https://doi.org/10.1016/j.ijpe.2014.06.007
"""

function croston_classic(x::AbstractVector;
    init_strategy::String = "mean",
    number_of_params::Int = 2,
    cost_metric::String = "mar",
    optimize_init::Bool = true,
    rm_missing::Bool = false,
)
    return fit_croston(x, "croston", cost_metric, number_of_params, init_strategy, optimize_init, rm_missing)
end


"""
    croston_sba(x::AbstractVector; init_strategy::String = "mean", number_of_params::Int = 2,
                cost_metric::String = "mar", optimize_init::Bool = true, rm_missing::Bool = false)
        -> IntermittentDemandCrostonFit

Forecast intermittent demand using the Syntetos-Boylan Approximation (SBA) - **Recommended Method**.

# Arguments

- `x::AbstractVector`: Input time series vector representing intermittent demand (including zeros and possibly `missing` values).
  Must contain at least two non-zero values.

- `init_strategy::String`: Initialization strategy for smoothing. Default is `"mean"`.
    - `"mean"`: Initialize using mean of non-zero demands and intervals (recommended)
    - `"naive"`: Initialize using first observed demand and interval

- `number_of_params::Int`: Number of smoothing parameters to optimize. Default is `2`.
    - `1`: Single smoothing parameter for both demand size and intervals
    - `2`: Separate smoothing parameters (recommended for better accuracy)

- `cost_metric::String`: Optimization cost function for parameter tuning. Default is `"mar"`.
    - `"mar"`: Mean Absolute Rate (recommended by Kourentzes 2014)
    - `"msr"`: Mean Squared Rate (recommended by Kourentzes 2014)
    - `"mae"`: Mean Absolute Error (classical metric)
    - `"mse"`: Mean Squared Error (classical metric)

- `optimize_init::Bool`: If `true`, initial values are optimized alongside smoothing parameters.
  Default is `true`. Recommended for short series.

- `rm_missing::Bool`: If `true`, `missing` values are removed from `x` before modeling. Default is `false`.

# Returns

`IntermittentDemandCrostonFit` - A struct containing:
- Optimized smoothing weights for demand and interval
- Initial values for the smoothing processes
- Model method identifier (SBA)
- Original input data

# Description

The Syntetos-Boylan Approximation (SBA) addresses a systematic bias in the classical Croston method.
The classical Croston method tends to over-forecast due to Jensen's inequality when computing the
ratio of smoothed demand to smoothed intervals.

**Bias Correction Formula:**
```
forecast = (1 - α/2) × (z / x)
```
where α is the smoothing parameter for intervals.

**Why SBA is Recommended:**
- Reduces systematic over-forecasting bias in classical Croston
- Improves forecast accuracy, especially for highly intermittent series
- Maintains computational efficiency of the original method
- Empirically validated across multiple studies

**When to Use SBA:**
- Intermittent demand with >50% zeros (recommended)
- Spare parts and slow-moving inventory
- Any scenario where classical Croston would be considered
- Default choice for intermittent demand unless you have specific reasons to use alternatives

# Examples

```julia
using Durbyn.IntermittentDemand

# Highly intermittent demand data
demand = [0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0]

# Fit with recommended SBA method
fit = croston_sba(demand)

# Generate forecast
fc = forecast(fit, h = 12)
println("SBA Forecast: ", fc.mean)

# Compare with classical Croston
fit_classic = croston_classic(demand)
fc_classic = forecast(fit_classic, h = 12)

println("Classical Forecast: ", fc_classic.mean)
println("Difference: ", fc.mean .- fc_classic.mean)

# Extract diagnostics
fitted_values = fitted(fit)
resids = residuals(fit)
```

# Grammar Interface

For declarative model specification with grouped data support:

```julia
using Durbyn

# Recommended: Use SBA for intermittent demand
spec = CrostonSpec(@formula(demand = croston(method="sba")))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)

# With custom parameters
spec = CrostonSpec(@formula(demand = croston(
    method="sba",
    cost_metric="msr",
    number_of_params=2
)))
```

# See Also

- [`croston_classic`](@ref) - Original Croston method (biased)
- [`croston_sbj`](@ref) - Alternative bias correction (SBJ)
- [`CrostonSpec`](@ref) - Grammar interface for model specification
- [`forecast`](@ref) - Generate forecasts from fitted model
- [`fitted`](@ref) - Extract in-sample fitted values
- [`residuals`](@ref) - Compute model residuals

# References

Syntetos, A. A., & Boylan, J. E. (2005). *The accuracy of intermittent demand estimates*.
International Journal of Forecasting, 21(2), 303-314.

Kourentzes, N. (2014). *On intermittent demand model optimisation and selection*.
International Journal of Production Economics, 156, 180–190.
https://doi.org/10.1016/j.ijpe.2014.06.007
"""
function croston_sba(x::AbstractVector;
    init_strategy::String = "mean",
    number_of_params::Int = 2,
    cost_metric::String = "mar",
    optimize_init::Bool = true,
    rm_missing::Bool = false,
)
    return fit_croston(x, "sba", cost_metric, number_of_params, init_strategy, optimize_init, rm_missing)
end

"""
    croston_sbj(x::AbstractVector; init_strategy::String = "mean", number_of_params::Int = 2,
                cost_metric::String = "mar", optimize_init::Bool = true, rm_missing::Bool = false)
        -> IntermittentDemandCrostonFit

Forecast intermittent demand using the Shale-Boylan-Johnston (SBJ) bias correction method.

# Arguments

- `x::AbstractVector`: Input time series vector representing intermittent demand (including zeros and possibly `missing` values).
  Must contain at least two non-zero values.

- `init_strategy::String`: Initialization strategy for smoothing. Default is `"mean"`.
    - `"mean"`: Initialize using mean of non-zero demands and intervals (recommended)
    - `"naive"`: Initialize using first observed demand and interval

- `number_of_params::Int`: Number of smoothing parameters to optimize. Default is `2`.
    - `1`: Single smoothing parameter for both demand size and intervals
    - `2`: Separate smoothing parameters (recommended for better accuracy)

- `cost_metric::String`: Optimization cost function for parameter tuning. Default is `"mar"`.
    - `"mar"`: Mean Absolute Rate (recommended by Kourentzes 2014)
    - `"msr"`: Mean Squared Rate (recommended by Kourentzes 2014)
    - `"mae"`: Mean Absolute Error (classical metric)
    - `"mse"`: Mean Squared Error (classical metric)

- `optimize_init::Bool`: If `true`, initial values are optimized alongside smoothing parameters.
  Default is `true`. Recommended for short series.

- `rm_missing::Bool`: If `true`, `missing` values are removed from `x` before modeling. Default is `false`.

# Returns

`IntermittentDemandCrostonFit` - A struct containing:
- Optimized smoothing weights for demand and interval
- Initial values for the smoothing processes
- Model method identifier (SBJ)
- Original input data

# Description

The Shale-Boylan-Johnston (SBJ) method provides an alternative bias correction to the Syntetos-Boylan
Approximation (SBA). It applies a different correction factor to address the over-forecasting bias in
the classical Croston method.

**Bias Correction Formula:**
```
forecast = (1 - α/(2-α)) × (z / x)
```
where α is the smoothing parameter for intervals.

**SBJ vs SBA:**
- SBJ applies a stronger correction than SBA for the same smoothing parameter
- SBJ correction factor: `1 - α/(2-α)` vs SBA: `1 - α/2`
- For small α, both methods give similar results
- For larger α (more responsive smoothing), SBJ applies stronger downward adjustment

**When to Use SBJ:**
- Alternative to SBA when it consistently over-forecasts
- Very intermittent series with extremely long intervals between demands
- Experimental comparison with SBA to find best method for your data
- Generally, try SBA first as it is more widely validated

**Recommendation:** Use [`croston_sba`](@ref) as the default choice. Only consider SBJ if
SBA shows consistent over-forecasting in your validation studies.

# Examples

```julia
using Durbyn.IntermittentDemand

# Very intermittent demand data
demand = [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0]

# Fit with SBJ method
fit_sbj = croston_sbj(demand)
fc_sbj = forecast(fit_sbj, h = 12)

# Compare with SBA
fit_sba = croston_sba(demand)
fc_sba = forecast(fit_sba, h = 12)

# Compare corrections
println("SBJ Forecast: ", fc_sbj.mean[1])
println("SBA Forecast: ", fc_sba.mean[1])
println("SBJ applies stronger correction: ", fc_sbj.mean[1] < fc_sba.mean[1])

# Extract model diagnostics
fitted_values = fitted(fit_sbj)
resids = residuals(fit_sbj)
```

# Grammar Interface

For declarative model specification with grouped data support:

```julia
using Durbyn

# Use SBJ method
spec = CrostonSpec(@formula(demand = croston(method="sbj")))
fitted = fit(spec, data)
fc = forecast(fitted, h = 12)

# Compare SBA and SBJ in model collection
models = model(
    CrostonSpec(@formula(demand = croston(method="sba"))),
    CrostonSpec(@formula(demand = croston(method="sbj"))),
    names = ["sba", "sbj"]
)
fitted = fit(models, panel_data)
```

# See Also

- [`croston_sba`](@ref) - Recommended bias-corrected method (try this first)
- [`croston_classic`](@ref) - Original Croston method (biased)
- [`CrostonSpec`](@ref) - Grammar interface for model specification
- [`forecast`](@ref) - Generate forecasts from fitted model
- [`fitted`](@ref) - Extract in-sample fitted values
- [`residuals`](@ref) - Compute model residuals

# References

Shale, E. A., Boylan, J. E., & Johnston, F. R. (2006). *Forecasting for intermittent demand:
The estimation of an unbiased average*. Journal of the Operational Research Society, 57(5), 588-592.

Kourentzes, N. (2014). *On intermittent demand model optimisation and selection*.
International Journal of Production Economics, 156, 180–190.
https://doi.org/10.1016/j.ijpe.2014.06.007
"""
function croston_sbj(x::AbstractVector;
    init_strategy::String = "mean",
    number_of_params::Int = 2,
    cost_metric::String = "mar",
    optimize_init::Bool = true,
    rm_missing::Bool = false,
)
    return fit_croston(x, "sbj", cost_metric, number_of_params, init_strategy, optimize_init, rm_missing)
end


