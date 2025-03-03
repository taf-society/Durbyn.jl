export CrostonForecast, croston, plot, forecast

@enum CrostonType CrostonOne=1 CrostonTwo=2 CrostonThree=3 CrostonFour=4

struct CrostonFit
    modely::Any
    modelp::Any
    type::CrostonType
    x::AbstractArray
    y::AbstractArray
    tt::AbstractArray
    m::Int
end

struct CrostonMethod
    demand::Any
    period::Any
    model::CrostonFit
end

struct CrostonForecast
    mean::Any
    model::CrostonFit
    method::Any
    m::Int
end


function croston(y::AbstractArray, m::Int; alpha::Union{Float64,Bool,Nothing}=nothing)

    x = copy(y)
    y = [val for val in x if val > 0]

    if isempty(y)
        type=CrostonType(1)
        y_f_struct=nothing
        p_f_struct=nothing
    end
    
    positions = findall(>(0), x)
    tt = diff(vcat(0, positions))

    if length(y) == 1 && length(tt) == 1
        type=CrostonType(2)
        y_f_struct=nothing
        p_f_struct=nothing
    elseif length(y) <= 1 || length(tt) <= 1
        type=CrostonType(3)
        y_f_struct=nothing
        p_f_struct=nothing
    else
        y_f_struct=ses(y, m, initial="simple", alpha=alpha)
        p_f_struct=ses(tt, m, initial="simple", alpha=alpha)
        type=CrostonType(4)
    end
    out=CrostonFit(y_f_struct,p_f_struct,type, x, y, tt,m)
    return(out)
end

function forecast(object::CrostonFit, h::Int)
    type=object.type
    x=object.x
    y=object.y
    tt=object.tt
    m=object.m
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
        y_f_struct = forecast_ets_base(object.modely, h=h)
        p_f_struct = forecast_ets_base(object.modelp, h=h)
        y_f = y_f_struct.mean
        p_f = p_f_struct.mean
        mean = y_f ./ p_f
    end

    object = CrostonForecast(mean, object, "Croston's Method", m)
end


function fitted(object::CrostonFit)
    
    n = length(y)
    ratio, y_f, p_f = croston_fit_fc(y, m; h=h, alpha=alpha)

        n = length(y)
        fits = fill(Number(NaN), n)
        if n > 1
            for i in 1:(n-1)
                tmp, y_f1, p_f1 = croston_fit_fc(y[1:i], m; h=1, alpha=alpha)
                fits[i+1] = tmp[1]
            end
        end

    Croston(Dict("demand" => y_f, "period" => p_f), ratio, fits, "Croston Method", m)
end

function plot(forecast::CrostonForecast)
    history = forecast.model.x
    n_history = length(history)
    mean_fc = forecast.mean
    time_history = 1:n_history
    time_forecast = (n_history + 1):(n_history + length(mean_fc))
    p = Plots.plot(time_history, history, label="Historical Data", lw=2, 
    title=forecast.method, xlabel="Time", ylabel="Value", linestyle=:dash)
    Plots.plot!(time_forecast, mean_fc, label="Forecast Mean", lw=3, color=:blue)
    return p
end
