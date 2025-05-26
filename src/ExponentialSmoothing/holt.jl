export holt
function holt(y::AbstractArray, m::Int; damped::Bool=false, initial::String="optimal", exponential::Bool=false,
    alpha::Union{Float64,Bool,Nothing}=nothing, beta::Union{Float64,Bool,Nothing}=nothing, phi::Union{Float64,Bool,Nothing}=nothing,
    lambda::Union{Float64,Bool,Nothing}=nothing, biasadj::Bool=false)

    initial = match_arg(initial, ["optimal", "simple"])
    model = nothing

    if length(y) <= 1
        throw(ArgumentError("Holt's method needs at least two observations to estimate trend."))
    end

    if initial == "optimal" || damped
        if exponential
            model = ets_base_model(y, m, "MMN", alpha=alpha, beta=beta, phi=phi, damped=damped, opt_crit="mse", lambda=lambda, biasadj=biasadj)
        else
            model = ets_base_model(y, m, "AAN", alpha=alpha, beta=beta, phi=phi, damped=damped, opt_crit="mse", lambda=lambda, biasadj=biasadj)
        end
    else
        model = holt_winters_conventional(y, m, alpha=alpha, beta=beta, gamma=false, phi=phi, exponential=exponential, lambda=lambda, biasadj=biasadj)
    end

    if damped
        method = "Damped Holt's method"
        if initial == "simple"
            @warn "Damped Holt's method requires optimal initialization"
        end
    else
        method = "Holt's method"
    end

    if exponential
        method = method * " with exponential trend"
    end
    return model, method
end