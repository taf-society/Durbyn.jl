"""
    CrostonType

Enumeration for Croston model fitting cases:
- `CrostonOne = 1`: All demands are zero - returns zero forecast
- `CrostonTwo = 2`: Only one non-zero demand and one interval - returns constant forecast
- `CrostonThree = 3`: Insufficient data (≤1 demand or ≤1 interval) - returns NaN
- `CrostonFour = 4`: Standard case with multiple demands - applies full Croston method
"""
@enum CrostonType CrostonOne = 1 CrostonTwo = 2 CrostonThree = 3 CrostonFour = 4

"""
    CrostonFit

Fitted Croston model for intermittent demand forecasting.

# Fields
- `modely::Any`: Simple exponential smoothing model for demand sizes
- `modelp::Any`: Simple exponential smoothing model for inter-demand intervals
- `type::CrostonType`: Type of Croston fitting approach used
- `x::AbstractArray`: Original demand series
- `y::AbstractArray`: Non-zero demands only
- `tt::AbstractArray`: Inter-demand intervals
- `m::Int`: Seasonal period (typically 1 for non-seasonal intermittent demand)

# References
- Croston, J. (1972). "Forecasting and stock control for intermittent demands".
  *Operational Research Quarterly*, 23(3), 289-303.
- Shenstone, L., and Hyndman, R.J. (2005). "Stochastic models underlying Croston's method
  for intermittent demand forecasting". *Journal of Forecasting*, 24, 389-402.

# See also
[`croston`](@ref), [`forecast`](@ref), [`fitted`](@ref)
"""
struct CrostonFit
    modely::Any
    modelp::Any
    type::CrostonType
    x::AbstractArray
    y::AbstractArray
    tt::AbstractArray
    m::Int
end

"""
    CrostonForecast

Croston forecast results for intermittent demand.

# Fields
- `mean::Any`: Forecast values (demand rate per period)
- `model::CrostonFit`: The underlying fitted Croston model
- `method::Any`: Description string ("Croston's Method")
- `m::Int`: Seasonal period

# See also
[`CrostonFit`](@ref), [`forecast`](@ref), [`plot`](@ref)
"""
struct CrostonForecast
    mean::Any
    model::CrostonFit
    method::Any
    m::Int
end


"""
    croston(y, m; alpha=nothing, options=NelderMeadOptions())

Fit a Croston model for intermittent demand forecasting.

Croston's method decomposes intermittent demand into two components: demand size (when it occurs)
and inter-demand intervals. Both components are modeled using simple exponential smoothing, and
the forecast is computed as the ratio of smoothed demand size to smoothed interval.

# Arguments
- `y::AbstractArray`: Demand time series (may contain zeros)
- `m::Int`: Seasonal period (typically 1 for non-seasonal intermittent demand)
- `alpha::Union{Float64,Bool,Nothing}=nothing`: Smoothing parameter. If `nothing`, parameter is
  optimized automatically
- `options::NelderMeadOptions=NelderMeadOptions()`: Optimization settings for parameter estimation

# Returns
- `CrostonFit`: Fitted Croston model object

# Examples
```julia
using Durbyn.ExponentialSmoothing

# Intermittent demand data
demand = [0, 0, 5, 0, 0, 3, 0, 0, 0, 7, 0, 0, 4, 0, 0]

# Fit with automatic parameter optimization
fit = croston(demand, 1)

# Fit with fixed smoothing parameter
fit_fixed = croston(demand, 1, alpha=0.1)

# Generate forecast
fc = forecast(fit, 12)
```

# References
- Croston, J. (1972). "Forecasting and stock control for intermittent demands".
  *Operational Research Quarterly*, 23(3), 289-303.
- Shenstone, L., and Hyndman, R.J. (2005). "Stochastic models underlying Croston's method
  for intermittent demand forecasting". *Journal of Forecasting*, 24, 389-402.

# See also
[`CrostonFit`](@ref), [`forecast`](@ref), [`fitted`](@ref), [`plot`](@ref)
"""
function croston(
    y::AbstractArray,
    m::Int;
    alpha::Union{Float64,Bool,Nothing} = nothing,
    options::NelderMeadOptions = NelderMeadOptions(),
)

    x = copy(y)
    y = [val for val in x if val > 0]

    if isempty(y)
        type = CrostonType(1)
        y_f_struct = nothing
        p_f_struct = nothing
    end

    positions = findall(>(0), x)
    tt = diff(vcat(0, positions))
    tt = na_omit(tt)

    if length(y) == 1 && length(tt) == 1
        type = CrostonType(2)
        y_f_struct = nothing
        p_f_struct = nothing
    elseif length(y) <= 1 || length(tt) <= 1
        type = CrostonType(3)
        y_f_struct = nothing
        p_f_struct = nothing
    else
        y_f_struct = ses(y, m, initial = "simple", alpha = alpha, options = options)
        p_f_struct = ses(tt, m, initial = "simple", alpha = alpha, options = options)
        type = CrostonType(4)
    end
    out = CrostonFit(y_f_struct, p_f_struct, type, x, y, tt, m)
    return (out)
end

"""
    forecast(object::CrostonFit, h::Int)

Generate forecasts from a fitted Croston model.

The forecast represents the expected demand rate per period, computed as the ratio of
smoothed demand size to smoothed inter-demand interval. The forecast is constant for all
future periods (flat forecast profile).

# Arguments
- `object::CrostonFit`: Fitted Croston model
- `h::Int`: Forecast horizon (number of periods ahead)

# Returns
- `CrostonForecast`: Forecast object containing mean forecasts

# Examples
```julia
# Fit model
fit = croston(demand, 1)

# Generate 12-period-ahead forecast
fc = forecast(fit, 12)
println(fc.mean)  # Access forecast values

# Visualize forecast
plot(fc, show_fitted=true)
```

# See also
[`CrostonFit`](@ref), [`croston`](@ref), [`plot`](@ref)
"""
function forecast(object::CrostonFit, h::Int)
    type = object.type
    x = object.x
    y = object.y
    tt = object.tt
    m = object.m
    if type == CrostonOne
        mean = fill(0.0, h)
        y_f = fill(0.0, length(x))
        p_f = fill(0.0, length(x))
    end

    if type == CrostonTwo
        y_f = fill(y[1], h)
        p_f = fill(tt[1], h)
        mean = y_f ./ p_f
    end

    if type == CrostonThree
        y_f = fill(Float64(NaN), h)
        p_f = fill(Float64(NaN), h)
        mean = fill(Float64(NaN), h)
    end

    if type == CrostonFour
        y_f_struct = forecast(object.modely, h = h)
        p_f_struct = forecast(object.modelp, h = h)
        y_f = y_f_struct.mean
        p_f = p_f_struct.mean
        mean = y_f ./ p_f
    end

    return CrostonForecast(mean, object, "Croston's Method", m)
end

"""
    croston(y; alpha=nothing, options=NelderMeadOptions())

Fit a Croston model with default seasonal period m=1 (non-seasonal).

# Arguments
- `y::AbstractArray`: Demand time series (may contain zeros)
- `alpha::Union{Float64,Bool,Nothing}=nothing`: Smoothing parameter
- `options::NelderMeadOptions=NelderMeadOptions()`: Optimization settings

# Returns
- `CrostonFit`: Fitted Croston model object

# Examples
```julia
# Fit non-seasonal Croston model
fit = croston(demand)
fc = forecast(fit, 12)
```

# See also
[`croston(y, m; ...)`](@ref)
"""
function croston(
    y::AbstractArray;
    alpha::Union{Float64,Bool,Nothing} = nothing,
    options::NelderMeadOptions = NelderMeadOptions(),
)
return croston(y, 1, alpha = alpha, options = options)
end


"""
    fitted(object::CrostonFit)

Compute in-sample fitted values using one-step-ahead forecasts.

For each time point t, the fitted value is the one-step-ahead forecast obtained by
fitting the model to all data up to time t-1. The first value is NaN as no forecast
is available for the first observation.

# Arguments
- `object::CrostonFit`: Fitted Croston model

# Returns
- `Vector`: Fitted values (same length as original series)

# Examples
```julia
fit = croston(demand, 1)
fitted_vals = fitted(fit)
residuals = demand .- fitted_vals
```

# See also
[`CrostonFit`](@ref), [`forecast`](@ref)
"""
function fitted(object::CrostonFit)
    x = object.x
    n = length(x)
    m = object.m

    n = length(x)
    fits = fill(Number(NaN), n)
    if n > 1
        for i = 1:(n-1)
            fc = forecast(croston(x[1:i], m; alpha = nothing), 1)
            fits[i+1] = fc.mean[1]
        end
    end
    return fits
end

"""
    plot(forecast::CrostonForecast; show_fitted=false)

Visualize Croston forecast with optional fitted values.

Creates a plot showing historical data, forecast values, and optionally in-sample fitted values.

# Arguments
- `forecast::CrostonForecast`: Forecast object to plot
- `show_fitted::Bool=false`: Whether to display in-sample fitted values

# Returns
- Plots.jl plot object

# Examples
```julia
fit = croston(demand, 1)
fc = forecast(fit, 12)

# Plot forecast only
plot(fc)

# Plot forecast with fitted values
plot(fc, show_fitted=true)
```

# See also
[`CrostonForecast`](@ref), [`forecast`](@ref), [`fitted`](@ref)
"""
function plot(forecast::CrostonForecast; show_fitted::Bool = false)

    history = forecast.model.x
    n_history = length(history)
    mean_fc = forecast.mean
    time_history = 1:n_history
    time_forecast = (n_history+1):(n_history+length(mean_fc))

    p = Plots.plot(
        time_history,
        history,
        label = "Historical Data",
        lw = 2,
        title = forecast.method,
        xlabel = "Time",
        ylabel = "Value",
        linestyle = :dash,
    )
    Plots.plot!(time_forecast, mean_fc, label = "Forecast Mean", lw = 3, color = :blue)

    if show_fitted
        fitted_val = fitted(forecast.model)
        Plots.plot!(
            time_history,
            fitted_val,
            label = "Fitted Values",
            lw = 3,
            linestyle = :dash,
            color = :blue,
        )
    end

    return p
end
