"""
    struct Forecast

Holds the results of a time series forecast from a fitted models.

# Fields

- `model::Any`  
  The original fitted model used to generate the forecast.

- `method::String`  
  The name of the forecasting method used.

- `mean::Vector{Float64}`  
  Point forecasts for the next `h` time steps beyond the end of the training data.

- `level::Union{AbstractVector{<:Real}, Nothing}`
  Confidence levels (e.g., `[80, 95]`) for the prediction intervals. `nothing` when
  prediction intervals are disabled.

- `x::Vector{Float64}`
  The original time series data that the model was trained on.

- `upper::Union{Matrix{Float64}, Nothing}`
  Upper prediction bounds, sized `(h, n_levels)`.
  Each column corresponds to a different confidence level. `nothing` when PI is disabled.

- `lower::Union{Matrix{Float64}, Nothing}`
  Lower prediction bounds, sized `(h, n_levels)`.
  Each column corresponds to a different confidence level. `nothing` when PI is disabled.

- `fitted::Vector{Union{Float64, Missing}}`  
  In-sample fitted values produced by the model.  
  Early values may be `missing` due to lag requirements.

- `residuals::Vector{Union{Float64, Missing}}`  
  In-sample residuals (observed - fitted) for the training data.

# Description

The `Forecast` object is returned by the `forecast` function and encapsulates all relevant 
forecast information, including point predictions, uncertainty intervals, and diagnostics.
 It is designed to support visualization, evaluation, and downstream processing.

# Example

```julia
y = randn(120)
model = arar(y)
fc = forecast(model, h = 12)

fc.mean          # point forecasts
fc.upper[:, 2]   # 95% upper bound
fc.residuals     # model residuals
```
"""
struct Forecast
    model::Any
    method::String
    mean::AbstractVector{<:Real}
    level::Union{AbstractVector{<:Real}, Nothing}
    x::AbstractVector{<:Real}
    upper::Union{AbstractMatrix{<:Real}, Nothing}
    lower::Union{AbstractMatrix{<:Real}, Nothing}
    fitted::AbstractVector
    residuals::AbstractVector
end

function Base.show(io::IO, fc::Forecast)
    println(io, "Forecast from ", fc.method)
    println(io, "-------------------------------")
    println(io, "Forecast horizon: ", length(fc.mean), " steps")
    println(io, "Confidence levels: ", fc.level)

    println(io, "\nFirst 5 point forecasts:")
    println(io, round.(fc.mean[1:min(end, 5)], digits=4))

    println(io, "\nLast 5 observations:")
    num_obs_to_show = min(5, length(fc.x))
    start_idx = max(1, length(fc.x) - num_obs_to_show + 1)
    println(io, round.(fc.x[start_idx:end], digits=4))

    fitted_vals = collect(skipmissing(fc.fitted))
    if length(fitted_vals) >= 5
        println(io, "\nLast 5 fitted values:")
        println(io, round.(fitted_vals[end-4:end], digits=4))
    else
        println(io, "\nNot enough fitted values (only $(length(fitted_vals)))")
    end

    resid_vals = collect(skipmissing(fc.residuals))
    if length(resid_vals) >= 5
        println(io, "\nLast 5 residuals:")
        println(io, round.(resid_vals[end-4:end], digits=4))
    else
        println(io, "\nNot enough residuals (only $(length(resid_vals)))")
    end
end

function forecast end