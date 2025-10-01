export holt_winters, HoltWinters

struct HoltWinters
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

function holt_winters(
    y::AbstractArray,
    m::Int;
    seasonal::String = "additive",
    damped::Bool = false,
    initial::String = "optimal",
    exponential::Bool = false,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    gamma::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    lambda::Union{Float64,Bool,Nothing} = nothing,
    biasadj::Bool = false,
    options::NelderMeadOptions = NelderMeadOptions(),
)

    initial = match_arg(initial, ["optimal", "simple"])
    seasonal = match_arg(seasonal, ["additive", "multiplicative"])

    if m <= 1
        throw(ArgumentError("The time series should have frequency greater than 1."))
    end

    if length(y) <= m + 3
        throw(ArgumentError("I need at least $(m + 3) observations to estimate seasonality."))
    end

    if seasonal == "additive" && exponential
        throw(ArgumentError("Forbidden model combination: additive seasonality with exponential trend."))
    end

    model_code = ""

    if initial == "optimal" || damped
        if seasonal == "additive"
            model_code = exponential ? "ANA" : "AAA"
        else
            model_code = exponential ? "MMM" : "MAM"
        end

        model = ets_base_model(
            y,
            m,
            model_code,
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            phi = phi,
            damped = damped,
            opt_crit = "mse",
            lambda = lambda,
            biasadj = biasadj,
            options = options,
        )
    else
        model = holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            phi = phi,
            seasonal = seasonal,
            exponential = exponential,
            lambda = lambda,
            biasadj = biasadj,
            options = options,
        )
    end

    method = damped ? "Damped Holt-Winters'" : "Holt-Winters'"
    method *= seasonal == "additive" ? " additive method" : " multiplicative method"
    if exponential
        method *= " with exponential trend"
    end

    if damped && initial == "simple"
        @warn "Damped Holt-Winters' method requires optimal initialization"
    end

    # Handle fields that don't exist in HoltWintersConventional
    loglik = hasfield(typeof(model), :loglik) ? model.loglik : NaN
    aic = hasfield(typeof(model), :aic) ? model.aic : NaN
    bic = hasfield(typeof(model), :bic) ? model.bic : NaN
    aicc = hasfield(typeof(model), :aicc) ? model.aicc : NaN
    mse = hasfield(typeof(model), :mse) ? model.mse : NaN
    amse = hasfield(typeof(model), :amse) ? model.amse : NaN
    fit = hasfield(typeof(model), :fit) ? model.fit : Float64[]

    return HoltWinters(
        model.fitted,
        model.residuals,
        model.components,
        model.x,
        model.par,
        loglik,
        model.initstate,
        model.states,
        model.state_names,
        model.SSE,
        model.sigma2,
        model.m,
        model.lambda,
        model.biasadj,
        aic,
        bic,
        aicc,
        mse,
        amse,
        fit,
        method,
    )
end
