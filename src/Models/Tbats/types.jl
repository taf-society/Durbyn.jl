# ─── TBATS type definitions ───────────────────────────────────────────────────
#
# Container types for fitted TBATS models and optimization metadata.

"""
    TBATSModel

Container for a fitted TBATS model (Box-Cox transformation, ARMA errors,
trend, and multiple seasonal components via Fourier terms) following
De Livera, Hyndman & Snyder (2011). Fields store smoothing parameters, ARMA
coefficients, state matrices (`x`/`seed_states`), fitted values, errors,
likelihood, information criteria, and metadata needed to forecast without
refitting. The descriptor `TBATS(omega, {p,q}, phi, <m1,k1>,...,<mJ,kJ>)`
uses the paper's notation, where `omega` is the Box-Cox lambda, `{p,q}` the ARMA
orders, `phi` the damping parameter, and `<m,k>` pairs define seasonal
periods and Fourier orders.

# References
- De Livera, A.M., Hyndman, R.J., & Snyder, R.D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. Journal of the American Statistical Association, 106(496), 1513-1527.
"""
mutable struct TBATSModel
    lambda::Union{Float64,Nothing}
    alpha::Float64
    beta::Union{Float64,Nothing}
    damping_parameter::Union{Float64,Nothing}
    gamma_one_values::Union{Vector{Float64},Nothing}
    gamma_two_values::Union{Vector{Float64},Nothing}
    ar_coefficients::Union{Vector{Float64},Nothing}
    ma_coefficients::Union{Vector{Float64},Nothing}
    seasonal_periods::Union{Vector{<:Real},Nothing}
    k_vector::Union{Vector{Int},Nothing}
    fitted_values::Vector{Float64}
    errors::Vector{Float64}
    x::Matrix{Float64}
    seed_states::AbstractArray{Float64}
    variance::Float64
    aic::Union{Float64,Nothing}
    likelihood::Float64
    optim_return_code::Int
    y::Vector{Float64}
    parameters::Dict{Symbol,Any}
    method::String
    biasadj::Bool
end

# Internal switch for the guarded final refinement (see fitting.jl); exists
# so evaluation harnesses can A/B the refinement against the reference
# single-run behaviour. Always true in normal use.
const _REFINE = Ref{Bool}(true)

_aic_val(model::Nothing) = Inf
_aic_val(model) = isnothing(model.aic) ? Inf : model.aic

function tbats_descriptor(
    lambda::Union{Float64,Nothing},
    ar_coefficients::Union{Vector{Float64},Nothing},
    ma_coefficients::Union{Vector{Float64},Nothing},
    damping_parameter::Union{Float64,Nothing},
    seasonal_periods::Union{Vector{<:Real},Nothing},
    k_vector::Union{Vector{Int},Nothing},
)
    lambda_str = isnothing(lambda) ? "1" : string(round(lambda, digits = 3))
    ar_count = isnothing(ar_coefficients) ? 0 : length(ar_coefficients)
    ma_count = isnothing(ma_coefficients) ? 0 : length(ma_coefficients)
    damping_str = isnothing(damping_parameter) ? "-" : string(round(damping_parameter, digits = 3))

    buffer = IOBuffer()
    print(buffer, "TBATS(", lambda_str, ", {", ar_count, ",", ma_count, "}, ", damping_str, ", ")

    if isnothing(seasonal_periods) || isempty(seasonal_periods)
        print(buffer, "{-})")
    else
        print(buffer, "{")
        for (i, (m, k)) in enumerate(zip(seasonal_periods, k_vector))
            print(buffer, "<", m, ",", k, ">")
            if i < length(seasonal_periods)
                print(buffer, ", ")
            end
        end
        print(buffer, "})")
    end

    return String(take!(buffer))
end

tbats_descriptor(model::TBATSModel) = tbats_descriptor(
    model.lambda,
    model.ar_coefficients,
    model.ma_coefficients,
    model.damping_parameter,
    model.seasonal_periods,
    model.k_vector,
)

Base.string(model::TBATSModel) = tbats_descriptor(model)

struct TBATSParameterControl
    use_box_cox::Bool
    use_beta::Bool
    use_damping::Bool
    length_gamma::Int
    p::Int
    q::Int
end
