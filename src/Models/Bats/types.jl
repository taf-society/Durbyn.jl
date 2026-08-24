# ─── BATS type definitions ────────────────────────────────────────────────────
#
# Container types for fitted BATS models and optimization metadata.

"""
    BATSModel

Container for fitted BATS models (Box-Cox transformation, ARMA errors,
trend and seasonal components) following De Livera, Hyndman & Snyder (2011).
Each field stores the estimated smoothing parameters, state matrices,
diagnostics and metadata required to regenerate forecasts without re-fitting.

Fields capture Box-Cox lambda, level/trend/seasonal coefficients,
ARMA coefficients, the state matrices `x`/`seed_states`, innovation
variance, information criteria, the original series, optimizer metadata
and a descriptive `method` string such as `BATS(λ, {p,q}, φ, {m…})`.
"""
mutable struct BATSModel
    lambda::Union{Float64,Nothing}
    alpha::Float64
    beta::Union{Float64,Nothing}
    damping_parameter::Union{Float64,Nothing}
    gamma_values::Union{Vector{Float64},Nothing}
    ar_coefficients::Union{Vector{Float64},Nothing}
    ma_coefficients::Union{Vector{Float64},Nothing}
    seasonal_periods::Union{Vector{Int},Nothing}
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

"""
    ParameterControl

Tracks which parameter groups are active in a packed parameter vector,
enabling parameterise/unparameterise to roundtrip without ambiguity.
"""
struct ParameterControl
    use_box_cox::Bool
    use_beta::Bool
    use_damping::Bool
    length_gamma::Int
    p::Int
    q::Int
end

function bats_descriptor(
    lambda::Union{Float64,Nothing},
    ar_coefficients::Union{Vector{Float64},Nothing},
    ma_coefficients::Union{Vector{Float64},Nothing},
    damping_parameter::Union{Float64,Nothing},
    seasonal_periods::Union{Vector{Int},Nothing},
)
    lambda_str = isnothing(lambda) ? "1" : string(round(lambda, digits = 3))
    ar_count = isnothing(ar_coefficients) ? 0 : length(ar_coefficients)
    ma_count = isnothing(ma_coefficients) ? 0 : length(ma_coefficients)
    damping_str = isnothing(damping_parameter) ? "-" : string(round(damping_parameter, digits = 3))

    buffer = IOBuffer()
    print(buffer, "BATS(", lambda_str, ", {", ar_count, ",", ma_count, "}, ", damping_str, ", ")

    if isnothing(seasonal_periods) || isempty(seasonal_periods)
        print(buffer, "-)")
    else
        print(buffer, "{", join(seasonal_periods, ","), "})")
    end

    return String(take!(buffer))
end

bats_descriptor(model::BATSModel) = bats_descriptor(
    model.lambda,
    model.ar_coefficients,
    model.ma_coefficients,
    model.damping_parameter,
    model.seasonal_periods,
)

Base.string(model::BATSModel) = bats_descriptor(model)
