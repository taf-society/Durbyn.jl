
function ets(
    y::AbstractArray,
    m::Int,
    model::Union{String,ETS};
    damped::Union{Bool,Nothing} = nothing,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    gamma::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    additive_only::Bool = false,
    lambda::Union{Float64,Bool,Nothing,String} = nothing,
    biasadj::Bool = false,
    lower::AbstractArray = [0.0001, 0.0001, 0.0001, 0.8],
    upper::AbstractArray = [0.9999, 0.9999, 0.9999, 0.98],
    opt_crit::String = "lik",
    nmse::Int = 3,
    bounds::String = "both",
    ic::String = "aicc",
    restrict::Bool = true,
    allow_multiplicative_trend::Bool = false,
    use_initial_values::Bool = false,
    na_action_type::String = "na_contiguous",
    opt_method::String = "Nelder-Mead",
    iterations::Int = 2000,
    kwargs...,
)

    if model == "ZZZ" && is_constant(y)
        return ses(y, alpha = 0.99999, initial = "simple")
    end

    out = ets_base_model(
        y,
        m,
        model,
        damped = damped,
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        phi = phi,
        additive_only = additive_only,
        lambda = lambda,
        biasadj = biasadj,
        lower = lower,
        upper = upper,
        opt_crit = opt_crit,
        nmse = nmse,
        bounds = bounds,
        ic = ic,
        restrict = restrict,
        allow_multiplicative_trend = allow_multiplicative_trend,
        use_initial_values = use_initial_values,
        na_action_type = na_action_type,
        opt_method = opt_method,
        iterations = iterations,
        kwargs...,
    )

    return out
end
