"""
IntermittentDemandForecast

A container for storing the results of an intermittent demand forecast model.

Fields

mean::Any: The mean forecast values over the prediction horizon, typically a vector of numeric values.

model::Dict: A dictionary containing detailed output from the fitted forecasting model, including in-sample forecasts, smoothing weights, initial values, and model components.

method::Any: A label describing the forecasting method used, such as "Classical Croston Method", "Croston Method with Syntetos-Boylan Approximation", or "Croston Method with Shale-Boylan-Johnston Bias Correction".

Description

This struct serves as a standardized return object for various Croston-based forecasting methods, helping users interpret, validate, or visualize forecast results from intermittent demand models.
"""
struct IntermittentDemandForecast
    mean::Any
    model::Dict
    method::Any
end

"""
croston_classic(x::Vector, h::Int = 10; w::Union{Nothing, Number, Vector{}} = nothing,
init::String = "mean", nop::Int = 2, cost::String = "mar",
init_opt::Bool = true, na_rm::Bool = false) -> IntermittentDemandForecast

Forecast intermittent demand using the classical Croston method.

# Arguments

* `x::Vector`: The input time series vector with intermittent demand (including zeros).
* `h::Int`: Forecast horizon (number of periods ahead to forecast). Defaults to 10.
* `w::Union{Nothing, Number, Vector}`: Smoothing parameter(s). If `nothing`, optimization is performed.
* `init::String`: Initialization method for the model. Accepts "mean" or "naive". Defaults to "mean".
* `nop::Int`: Number of parameters for optimization (1 or 2). Defaults to 2.
* `cost::String`: Optimization cost metric ("mar", "msr", "mae", or "mse"). Defaults to "mar".
* `init_opt::Bool`: Whether to optimize initial values along with smoothing parameters. Defaults to `true`.
* `na_rm::Bool`: If `true`, missing values (`missing`) will be removed from the input series. Defaults to `false`.

# Returns

* `IntermittentDemandForecast`: An object containing the fitted model, forecast, and 
method which is "Classical Croston Method".

# Description

This function applies the classical Croston method, which separates the modeling of demand size and interval for 
    forecasting intermittent demand. It optionally tunes smoothing parameters and initial values based on a 
    selected cost metric.
"""
function croston_classic(x::Vector, h::Int = 10; w::Union{Nothing, Number, Vector{}} = nothing, 
    init::String = "mean", nop::Int = 2, cost::String = "mar", init_opt::Bool = true, 
    na_rm::Bool = false)

    model, mean = crost(x, h, w, init, nop, "croston", cost, init_opt, na_rm)

    return IntermittentDemandForecast(mean, model, "Classical Croston Method")
end

"""
croston_sba(x::Vector, h::Int = 10; w::Union{Nothing, Number, Vector{}} = nothing,
init::String = "mean", nop::Int = 2, cost::String = "mar",
init_opt::Bool = true, na_rm::Bool = false) -> IntermittentDemandForecast

Forecast intermittent demand using the Croston method with Syntetos-Boylan Approximation (SBA).

Arguments

x::Vector: The input time series vector with intermittent demand (including zeros).

h::Int: Forecast horizon (number of periods ahead to forecast). Defaults to 10.

w::Union{Nothing, Number, Vector}: Smoothing parameter(s). If nothing, optimization is performed.

init::String: Initialization method for the model. Accepts "mean" or "naive". Defaults to "mean".

nop::Int: Number of parameters for optimization (1 or 2). Defaults to 2.

cost::String: Optimization cost metric ("mar", "msr", "mae", or "mse"). Defaults to "mar".

init_opt::Bool: Whether to optimize initial values along with smoothing parameters. Defaults to true.

na_rm::Bool: If true, missing values (missing) will be removed from the input series. Defaults to false.

Returns

IntermittentDemandForecast: An object containing the fitted model, forecast, and method, 
labeled as using the "Croston Method with Syntetos-Boylan Approximation".

Description

This function applies the Croston method enhanced with the Syntetos-Boylan Approximation (SBA), 
    a bias correction to improve forecast accuracy for intermittent demand. SBA adjusts Croston's original
         formula by applying a scaling factor to reduce the tendency to overestimate.
"""
function croston_sba(x::Vector, h::Int = 10; w::Union{Nothing, Number, Vector{}} = nothing, 
    init::String = "mean", nop::Int = 2, cost::String = "mar", init_opt::Bool = true, na_rm::Bool = false)
    model, mean = crost(x, h, w, init, nop, "sba", cost, init_opt, na_rm)
    return IntermittentDemandForecast(mean, model, "Croston Method with Syntetos-Boylan Approximation")
end

"""
croston_sbj(x::Vector, h::Int = 10; w::Union{Nothing, Number, Vector{}} = nothing,
init::String = "mean", nop::Int = 2, cost::String = "mar",
init_opt::Bool = true, na_rm::Bool = false) -> IntermittentDemandForecast

Forecast intermittent demand using the Croston method with Shale-Boylan-Johnston (SBJ) bias correction.

Arguments

x::Vector: The input time series vector with intermittent demand (including zeros).

h::Int: Forecast horizon (number of periods ahead to forecast). Defaults to 10.

w::Union{Nothing, Number, Vector}: Smoothing parameter(s). If nothing, optimization is performed.

init::String: Initialization method for the model. Accepts "mean" or "naive". Defaults to "mean".

nop::Int: Number of parameters for optimization (1 or 2). Defaults to 2.

cost::String: Optimization cost metric ("mar", "msr", "mae", or "mse"). Defaults to "mar".

init_opt::Bool: Whether to optimize initial values along with smoothing parameters. Defaults to true.

na_rm::Bool: If true, missing values (missing) will be removed from the input series. Defaults to false.

Returns

IntermittentDemandForecast: An object containing the fitted model, forecast, and metadata, labeled as using the "Croston Method with Shale-Boylan-Johnston Bias Correction".

Description

This function applies the Croston method with the Shale-Boylan-Johnston (SBJ) bias correction. SBJ provides an alternative adjustment to Croston’s formula to reduce bias, particularly for very intermittent demand scenarios. It uses a different correction factor than SBA, and may offer improved performance in certain cases.
"""

function croston_sbj(x::Vector, h::Int = 10; w::Union{Nothing, Number, Vector{}} = nothing, 
    init::String = "mean", nop::Int = 2, cost::String = "mar", init_opt::Bool = true, na_rm::Bool = false)
    model, mean = crost(x, h, w, init, nop, "sbj", cost, init_opt, na_rm)

    return IntermittentDemandForecast(mean, model, "Croston Method with Shale-Boylan-Johnston Bias Correction")
end
