"""
    struct IntermittentDemandCrostonFit

Structure representing the results and configuration for a Croston-based
intermittent demand forecasting method.

# Fields
- `weights::AbstractArray`: Smoothing parameters for demand and interval.

- `initial::AbstractArray`: Initial values for demand and interval smoothing.

- `method::String`: The Croston method used. One of:
    - `"croston"` (Classical Croston Method)
    - `"sba"` (Croston Method with Syntetos-Boylan Approximation)
    - `"sbj"` (Croston-Shale-Boylan-Johnston Bias Correction Method)

- `na_rm::Bool`: Whether missing values were removed prior to fitting. When `true`,
  `x` contains the cleaned series (missings already removed).

- `x::AbstractArray`: The demand time series used for fitting. When `na_rm=true`,
  this is the cleaned series with missing values removed.

"""
struct IntermittentDemandCrostonFit
    weights::AbstractArray
    initial::AbstractArray
    method::String
    na_rm::Bool
    x::AbstractArray
end

"""
    IntermittentDemandForecast

A container for storing the results of an intermittent demand forecast model.

# Fields
- `mean::Any`: The mean forecast values over the prediction horizon, typically a vector
  of numeric values. `nothing` when `h=0`.

- `model::IntermittentDemandCrostonFit`: The fitted model containing smoothing weights,
  initial values, method, and input data.

- `method::Any`: A label describing the forecasting method used
  (`"croston"`, `"sba"`, or `"sbj"`).
"""
struct IntermittentDemandForecast
    mean::Any
    model::IntermittentDemandCrostonFit
    method::Any
end

function show(io::IO, model::IntermittentDemandCrostonFit)
    println(io, "IntermittentDemandCrostonFit:")
    println(io, "  Method:     ", model.method)
    println(io, "  NA Removed: ", model.na_rm)
    println(io, "  Weights:    ", model.weights)
    println(io, "  Initial:    ", model.initial)
    println(io, "  Data Length:", length(model.x))
end


function show(io::IO, forecast::IntermittentDemandForecast)
    println(io, "IntermittentDemandForecast:")
    println(io, "  Method:     ", forecast.method)
    if !isnothing(forecast.mean)
        println(io, "  Mean (first 5 values): ", forecast.mean[1:min(5, end)])
    else
        println(io, "  Mean: not yet forecasted")
    end
    println(io, "  Model Summary:")
    show(io, forecast.model)
end


function croston_opt(x, method, cost, w, nop, init, init_opt)

    nzd = findall(xi -> xi != 0, x)
    k = length(nzd)
    intervals = [nzd[1], nzd[2:k] .- nzd[1:(k-1)]...]

    if isnothing(w) && !init_opt
        p0 = fill(0.05, nop)
        lbound = fill(0, nop)
        ubound = fill(1, nop)

        if nop != 1
            cost_fn = p -> croston_cost(p, x, cost, method, w,
                nop, true, init, init_opt, lbound, ubound)
            options = NelderMeadOptions(maxit=2000)
            result = nelder_mead(cost_fn, p0, options)
            wopt = result.x_opt
        else
            cost_fn = p -> croston_cost([p], x, cost, method, w,
                nop, true, init, init_opt, lbound, ubound)
            result = brent(cost_fn, lbound[1], ubound[1])
            wopt = [result.x_opt]
        end

        wopt = [wopt..., init...]

    elseif isnothing(w) && init_opt
        p0 = [fill(0.05, nop)..., init[1], init[2]]
        lbound = [fill(0, nop)..., 0, 1]
        ubound = [fill(1, nop)..., maximum(x), maximum(intervals)]

        cost_fn = p -> croston_cost(p, x, cost, method, w,
            nop, true, init, true, lbound, ubound)
        options = NelderMeadOptions(maxit=2000)
        result = nelder_mead(cost_fn, p0, options)
        wopt = result.x_opt

    elseif !isnothing(w) && init_opt
        nop = length(w)
        p0 = [init[1], init[2]]
        lbound = [0, 1]
        ubound = [maximum(x), maximum(intervals)]

        cost_fn = p -> croston_cost(p, x, cost, method, w,
            nop, false, init, true, lbound, ubound)
        options = NelderMeadOptions(maxit=2000)
        result = nelder_mead(cost_fn, p0, options)
        wopt = result.x_opt

        wopt = [w..., wopt...]

    else
        # Fixed weights, no init optimization: use weights and init as-is
        wopt = [w..., init...]
    end

    return Dict("w" => wopt[1:nop], "init" => wopt[(nop+1):(nop+2)])
end

function croston_cost(params, x, cost_metric, method, fixed_weights, num_params, optimize_weights,
    initial_values, optimize_inits, lower_bounds, upper_bounds)

    # Check bounds first to avoid computing with invalid params (e.g. division by near-zero interval)
    total_param_count = num_params * optimize_weights + 2 * optimize_inits
    for i in 1:total_param_count
        if params[i] < lower_bounds[i] || params[i] > upper_bounds[i]
            return 9e99
        end
    end

    if optimize_weights
        weights = params[1:num_params]
        inits = optimize_inits ? params[(num_params+1):(num_params+2)] : initial_values
    else
        weights = fixed_weights
        inits = params
    end

    forecast_result = pred_crost(x, 0, collect(weights), inits, method, false)
    forecast_input = forecast_result["frc_in"]

    return evaluation_metrics(x, forecast_input)[cost_metric]
end

function pred_crost(x,h,w,init,method,na_rm)

    if na_rm
        x = collect(filter(!ismissing, x))
    else
        if any(ismissing, x)
            throw(ArgumentError("Input contains missing values. Use rm_missing=true to remove them."))
        end
    end
    n = length(x)

    sum(x .!= 0) >= 2 || throw(ArgumentError("Need at least two non-zero values to model the time series."))
    nzd = findall(xi -> xi != 0, x)
    k = length(nzd)
    z = x[nzd]
    intervals = [nzd[1], nzd[2:k] .- nzd[1:(k-1)]...]

    if !(isa(init, Array))
        init = init == "mean" ? [z[1], mean(intervals)] : [z[1], intervals[1]]
    end

    w = convert(Vector{Float64}, w)
    init = convert(Vector{Float64}, init)

    zfit = zeros(k)
    xfit = zeros(k)
    zfit[1] = init[1]
    xfit[1] = init[2]

    a_demand, a_interval = length(w) == 1 ? (w[1], w[1]) : (w[1], w[2])

    coeff = method == "sba" ? 1 - (a_interval / 2) :
            method == "sbj" ? 1 - a_interval / (2 - a_interval) :
            1

    for i in 2:k
        zfit[i] = zfit[i-1] + a_demand * (z[i] - zfit[i-1])
        xfit[i] = xfit[i-1] + a_interval * (intervals[i] - xfit[i-1])
    end

    cc = coeff .* zfit ./ xfit
    frc_in = zeros(n)
    x_in = zeros(n)
    z_in = zeros(n)
    tv = [nzd .+ 1..., n]

    for i in 1:k
        idx_range = tv[i]:min(tv[i+1], n)
        frc_in[idx_range] .= cc[i]
        x_in[idx_range] .= xfit[i]
        z_in[idx_range] .= zfit[i]
    end

    if h > 0
        frc_out = fill(cc[k], h)
        x_out = fill(xfit[k], h)
        z_out = fill(zfit[k], h)
    else
        frc_out = nothing
        x_out = nothing
        z_out = nothing
    end

    w = length(w) == 1 ? [w[1], w[1]] : w

    c_in = Dict("Demand" => z_in, "Interval" => x_in)
    c_out = h > 0 ? Dict("Demand" => z_out, "Interval" => x_out) : NaN

    comp = Dict("c_in" => c_in, "c_out" => c_out, "coeff" => coeff)
    initial = [zfit[1], xfit[1]]

    return Dict(
        "data" => x,
        "method" => method,
        "frc_in" => frc_in,
        "frc_out" => frc_out,
        "weights" => w,
        "initial" => initial,
        "components" => comp
    )
end

function fit_croston(x, method::String="croston", cost::String="mse", 
                     nop::Int=2, init_strategy::String="mean", 
                     optimize_init::Bool=true, na_rm::Bool=false)

    method = match_arg(method, ["croston", "sba", "sbj"])
    cost = match_arg(cost, ["mar", "msr", "mae", "mse"])
    init_strategy = match_arg(init_strategy, ["naive", "mean"])

    if !(nop in [1, 2])
        @warn "nop can be either 1 or 2. Overriden to 2."
        nop = 2
    end

    if na_rm
        x = collect(filter(!ismissing, x))
    else
        if any(ismissing, x)
            throw(ArgumentError("Input contains missing values. Use rm_missing=true to remove them."))
        end
    end

    sum(x .!= 0) >= 2 || throw(ArgumentError("Need at least two non-zero values to model the time series."))

    nzd = findall(!=(0), x)
    intervals = [nzd[1]; diff(nzd)]

    z = x[nzd]
    init = init_strategy == "mean" ? [z[1], mean(intervals)] : [z[1], intervals[1]]

    opt = croston_opt(x, method, cost, nothing, nop, init, optimize_init)

    return IntermittentDemandCrostonFit(opt["w"], opt["init"], method, na_rm, x)
end

function predict_croston(model::IntermittentDemandCrostonFit, h::Int)

    w = model.weights
    init = model.initial
    method = model.method
    x = model.x

    if any(ismissing, x)
        throw(ArgumentError("model.x contains missing values. Refit with rm_missing=true or provide clean data."))
    end

    return pred_crost(x, h, w, init, method, false)
end


"""
    forecast(object::IntermittentDemandCrostonFit; h::Int = 10) 
        -> IntermittentDemandForecast

Generate forecasts for intermittent demand using a fitted Croston-based model.

# Arguments

- `object::IntermittentDemandCrostonFit`: A fitted intermittent demand model, as returned by 
  functions like [`croston_classic`] or other Croston variants. This object contains the smoothing 
  parameters, initialization strategy, input data, and method type.

- `h::Int`: Forecast horizon (i.e., number of future periods to predict). Default is `10`.

# Returns

- `IntermittentDemandForecast`: An object that encapsulates the forecasted values, the original fitted 
  model (`IntermittentDemandCrostonFit`), and the method used.

# Example

```julia
fit = croston_classic(demand_series)
fc = forecast(fit, h=12)
println(fc.mean)
```

"""
function forecast(object::IntermittentDemandCrostonFit; h::Int = 10)
    out = predict_croston(object, h)
    return IntermittentDemandForecast(out["frc_out"], object, object.method)
end

"""
    fitted(object::IntermittentDemandCrostonFit) -> AbstractVector
    fitted(object::IntermittentDemandForecast) -> AbstractVector

Extract the in-sample fitted values from a Croston-based intermittent demand model.

# Arguments

- `object::IntermittentDemandCrostonFit`: A fitted Croston model containing smoothing parameters, initialization settings, and original data.

- `object::IntermittentDemandForecast`: A forecast object wrapping a `IntermittentDemandCrostonFit` model.

# Returns

- `AbstractVector`: A vector of in-sample fitted values corresponding to the input demand time series. These represent the model's estimates at each time step during training.

# Description

This function returns the in-sample forecasts (`frc_in`) produced by the Croston-based model. It reconstructs the fitted values using the original demand data, smoothing weights, initialization values, and selected Croston method variant.

For `IntermittentDemandForecast` objects, the method delegates to the underlying fitted model.
"""
function fitted(object::IntermittentDemandCrostonFit)
    w = object.weights
    init = object.initial
    method = object.method
    x = object.x

    if any(ismissing, x)
        throw(ArgumentError("model.x contains missing values. Refit with rm_missing=true or provide clean data."))
    end

    return pred_crost(x, 0, w, init, method, false)["frc_in"]
end

function fitted(object::IntermittentDemandForecast)
    return fitted(object.model)
end

"""
    residuals(object::IntermittentDemandCrostonFit) -> AbstractVector
    residuals(object::IntermittentDemandForecast) -> AbstractVector

Compute the residuals from an intermittent demand model.

# Arguments

- `object::IntermittentDemandCrostonFit`: A fitted Croston model object.

- `object::IntermittentDemandForecast`: A forecast object wrapping a fitted Croston model.

# Returns

- `AbstractVector`: A vector of residuals, computed as the difference between observed values and fitted in-sample forecasts.

# Description

Residuals represent the error between the actual demand values and the fitted values produced by the model.
This function subtracts the in-sample forecasts from the original demand time series (`x`) to yield these residuals.

For `IntermittentDemandForecast` objects, residuals are computed by delegating to the underlying model.
"""
function residuals(object::IntermittentDemandCrostonFit)
    return object.x - fitted(object)
end

function residuals(object::IntermittentDemandForecast)
    return residuals(object.model)
end