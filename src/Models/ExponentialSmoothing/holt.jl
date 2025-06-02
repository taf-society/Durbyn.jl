export holt, Holt

struct Holt
    fitted::AbstractArray
    residuals::AbstractArray
    components::Vector{Any}
    x::AbstractArray
    par::Any
    loglik::Union{Float64,Int}
    initstate::AbstractArray
    states::AbstractArray
    state_names::Any
    SSE::Union{Float64,Int}
    sigma2::Union{Float64,Int}
    m::Int
    lambda::Union{Float64,Bool,Nothing}
    biasadj::Bool
    aic::Union{Float64,Int}
    bic::Union{Float64,Int}
    aicc::Union{Float64,Int}
    mse::Union{Float64,Int}
    amse::Union{Float64,Int}
    fit::Any
    method::String
end

function holt(
    y::AbstractArray,
    m::Int;
    damped::Bool = false,
    initial::String = "optimal",
    exponential::Bool = false,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    lambda::Union{Float64,Bool,Nothing} = nothing,
    biasadj::Bool = false,
)

    initial = match_arg(initial, ["optimal", "simple"])
    model = nothing

    if length(y) <= 1
        throw(
            ArgumentError(
                "Holt's method needs at least two observations to estimate trend.",
            ),
        )
    end

    if initial == "optimal" || damped
        if exponential
            model = ets_base_model(
                y,
                m,
                "MMN",
                alpha = alpha,
                beta = beta,
                phi = phi,
                damped = damped,
                opt_crit = "mse",
                lambda = lambda,
                biasadj = biasadj,
            )
        else
            model = ets_base_model(
                y,
                m,
                "AAN",
                alpha = alpha,
                beta = beta,
                phi = phi,
                damped = damped,
                opt_crit = "mse",
                lambda = lambda,
                biasadj = biasadj,
            )
        end
    else
        model = holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = beta,
            gamma = false,
            phi = phi,
            exponential = exponential,
            lambda = lambda,
            biasadj = biasadj,
        )
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

    return Holt(
        model.fitted,
        model.residuals,
        model.components,
        model.x,
        model.par,
        model.loglik,
        model.initstate,
        model.states,
        model.state_names,
        model.SSE,
        model.sigma2,
        model.m,
        model.lambda,
        model.biasadj,
        model.aic,
        model.bic,
        model.aicc,
        model.mse,
        model.amse,
        model.fit,
        method,
    )
end