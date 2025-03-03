export CrostonForecast, CrostonFit, croston, plot, forecast, fitted

@enum CrostonType CrostonOne = 1 CrostonTwo = 2 CrostonThree = 3 CrostonFour = 4

struct CrostonFit
    modely::Any
    modelp::Any
    type::CrostonType
    x::AbstractArray
    y::AbstractArray
    tt::AbstractArray
    m::Int
end

struct CrostonForecast
    mean::Any
    model::CrostonFit
    method::Any
    m::Int
end


function croston(y::AbstractArray, m::Int; alpha::Union{Float64,Bool,Nothing} = nothing)

    x = copy(y)
    y = [val for val in x if val > 0]

    if isempty(y)
        type = CrostonType(1)
        y_f_struct = nothing
        p_f_struct = nothing
    end

    positions = findall(>(0), x)
    tt = diff(vcat(0, positions))

    if length(y) == 1 && length(tt) == 1
        type = CrostonType(2)
        y_f_struct = nothing
        p_f_struct = nothing
    elseif length(y) <= 1 || length(tt) <= 1
        type = CrostonType(3)
        y_f_struct = nothing
        p_f_struct = nothing
    else
        y_f_struct = ses(y, m, initial = "simple", alpha = alpha)
        p_f_struct = ses(tt, m, initial = "simple", alpha = alpha)
        type = CrostonType(4)
    end
    out = CrostonFit(y_f_struct, p_f_struct, type, x, y, tt, m)
    return (out)
end

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
        y_f_struct = forecast_ets_base(object.modely, h = h)
        p_f_struct = forecast_ets_base(object.modelp, h = h)
        y_f = y_f_struct.mean
        p_f = p_f_struct.mean
        mean = y_f ./ p_f
    end

    object = CrostonForecast(mean, object, "Croston's Method", m)
end


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
