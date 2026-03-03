"""
    PDQ(p, d, q)

ARIMA order specification: autoregressive (`p`), differencing (`d`), and moving average (`q`) terms.

# Fields
- `p::Int`: Number of autoregressive (AR) terms.
- `d::Int`: Degree of differencing.
- `q::Int`: Number of moving average (MA) terms.
"""
struct PDQ
    p::Int
    d::Int
    q::Int

    function PDQ(p::Int, d::Int, q::Int)
        if p < 0 || d < 0 || q < 0
            throw(
                ArgumentError(
                    "All PDQ parameters must be non-negative integers. Got: p=$p, d=$d, q=$q",
                ),
            )
        end
        new(p, d, q)
    end
end

mutable struct ArimaStateSpace
    phi::AbstractVector
    theta::AbstractVector
    Delta::AbstractVector
    Z::AbstractVector
    a::AbstractVector
    P::AbstractMatrix
    T::AbstractMatrix
    V::Matrix{Float64}
    h::Real
    Pn::AbstractMatrix
end

mutable struct KalmanWorkspace
    anew::Vector{Float64}
    M::Vector{Float64}
    mm::Union{Matrix{Float64},Nothing}
    rsResid::Union{Vector{Float64},Nothing}
end

function KalmanWorkspace(rd::Int, n::Int, d::Int, give_resid::Bool)
    anew = zeros(rd)
    M = zeros(rd)
    mm = d > 0 ? zeros(rd, rd) : nothing
    rsResid = give_resid ? zeros(n) : nothing
    return KalmanWorkspace(anew, M, mm, rsResid)
end

function reset!(ws::KalmanWorkspace)
    fill!(ws.anew, 0.0)
    fill!(ws.M, 0.0)
    if !isnothing(ws.mm)
        fill!(ws.mm, 0.0)
    end
    if !isnothing(ws.rsResid)
        fill!(ws.rsResid, 0.0)
    end
    return ws
end

"""
    StateSpaceModel

Abstract supertype for all state-space models in the ARIMA module.
"""
abstract type StateSpaceModel end

"""
    SARIMAOrder

Compact representation of a SARIMA(p,d,q)(P,D,Q)[s] model order.

# Fields
- `p::Int`: Non-seasonal AR order.
- `d::Int`: Non-seasonal differencing order.
- `q::Int`: Non-seasonal MA order.
- `P::Int`: Seasonal AR order.
- `D::Int`: Seasonal differencing order.
- `Q::Int`: Seasonal MA order.
- `s::Int`: Seasonal period.
- `r::Int`: State dimension for ARMA part: `max(p + s*P, q + s*Q + 1)`.
- `n_diff::Int`: Total differencing: `d + D*s`.
- `rd::Int`: Full state dimension: `r + n_diff`.
"""
struct SARIMAOrder
    p::Int
    d::Int
    q::Int
    P::Int
    D::Int
    Q::Int
    s::Int
    r::Int
    n_diff::Int
    rd::Int

    function SARIMAOrder(p::Int, d::Int, q::Int, P::Int, D::Int, Q::Int, s::Int)
        p >= 0 || throw(ArgumentError("p must be non-negative, got $p"))
        d >= 0 || throw(ArgumentError("d must be non-negative, got $d"))
        q >= 0 || throw(ArgumentError("q must be non-negative, got $q"))
        P >= 0 || throw(ArgumentError("P must be non-negative, got $P"))
        D >= 0 || throw(ArgumentError("D must be non-negative, got $D"))
        Q >= 0 || throw(ArgumentError("Q must be non-negative, got $Q"))
        s >= 0 || throw(ArgumentError("s must be non-negative, got $s"))

        p_full = p + s * P
        q_full = q + s * Q
        r = max(p_full, q_full + 1)

        Delta = [1.0]
        for _ in 1:d
            Delta = _ts_conv(Delta, [1.0, -1.0])
        end
        for _ in 1:D
            seasonal_filter = [1.0; zeros(s - 1); -1.0]
            Delta = _ts_conv(Delta, seasonal_filter)
        end
        n_diff = length(Delta) - 1
        rd = r + n_diff

        new(p, d, q, P, D, Q, s, r, n_diff, rd)
    end
end

function _ts_conv(a::Vector{Float64}, b::Vector{Float64})
    na = length(a)
    nb = length(b)
    nab = na + nb - 1
    ab = zeros(Float64, nab)
    @inbounds for i in 1:na
        for j in 1:nb
            ab[i+j-1] += a[i] * b[j]
        end
    end
    return ab
end

SARIMAOrder(order::PDQ, seasonal::PDQ, s::Int) =
    SARIMAOrder(order.p, order.d, order.q, seasonal.p, seasonal.d, seasonal.q, s)

"""
    HyperParameters{Fl}

Mutable container for SARIMA model parameters.

Holds both the constrained (model-space) and unconstrained (optimizer-space)
representations of the parameter vector, along with a mask indicating which
parameters are free.

# Fields
- `constrained::Vector{Fl}`: Parameters in model space (AR/MA/xreg coefficients).
- `unconstrained::Vector{Fl}`: Parameters in unconstrained space (for optimizer).
- `mask::Vector{Bool}`: `true` for free parameters, `false` for fixed.
- `names::Vector{String}`: Human-readable parameter names.
- `fixed::Vector{Fl}`: Fixed parameter values; `NaN` marks free parameters.
"""
mutable struct HyperParameters{Fl<:AbstractFloat}
    constrained::Vector{Fl}
    unconstrained::Vector{Fl}
    mask::Vector{Bool}
    names::Vector{String}
    fixed::Vector{Fl}
end

"""
    SARIMASystem{Fl}

State-space matrices for a SARIMA model.

Same field names as `ArimaStateSpace` so that Kalman filter functions work via duck typing.

# Fields
- `phi::Vector{Fl}`: AR coefficients.
- `theta::Vector{Fl}`: MA coefficients.
- `Delta::Vector{Fl}`: Differencing polynomial coefficients.
- `Z::Vector{Fl}`: Observation vector.
- `T::Matrix{Fl}`: Transition matrix.
- `V::Matrix{Fl}`: Process noise covariance (RQR').
- `h::Fl`: Observation noise variance.
- `a::Vector{Fl}`: Current state estimate.
- `P::Matrix{Fl}`: Current state covariance.
- `Pn::Matrix{Fl}`: Prior state covariance (initial/predicted).
"""
mutable struct SARIMASystem{Fl<:AbstractFloat}
    phi::Vector{Fl}
    theta::Vector{Fl}
    Delta::Vector{Fl}
    Z::Vector{Fl}
    T::Matrix{Fl}
    V::Matrix{Fl}
    h::Fl
    a::Vector{Fl}
    P::Matrix{Fl}
    Pn::Matrix{Fl}
end

"""
    SARIMAResults{Fl}

Results from fitting a SARIMA model.

# Fields
- `coef::NamedMatrix`: Estimated coefficients with names.
- `sigma2::Float64`: Estimated innovation variance.
- `var_coef::Matrix{Float64}`: Variance-covariance matrix of coefficients.
- `loglik::Float64`: Maximized log-likelihood.
- `aic::Union{Float64,Nothing}`: Akaike information criterion.
- `bic::Union{Float64,Nothing}`: Bayesian information criterion.
- `aicc::Union{Float64,Nothing}`: Corrected AIC.
- `ic::Union{Float64,Nothing}`: Selected information criterion for comparison.
- `residuals::Vector{Float64}`: Model residuals.
- `fitted_values::Vector{Float64}`: In-sample fitted values.
- `convergence_code::Bool`: `true` if optimizer converged successfully.
- `n_cond::Int`: Number of conditioning observations.
- `nobs::Int`: Number of observations used in estimation.
"""
mutable struct SARIMAResults{Fl<:AbstractFloat}
    coef::NamedMatrix
    sigma2::Float64
    var_coef::Matrix{Float64}
    loglik::Float64
    aic::Union{Float64,Nothing}
    bic::Union{Float64,Nothing}
    aicc::Union{Float64,Nothing}
    ic::Union{Float64,Nothing}
    residuals::Vector{Float64}
    fitted_values::Vector{Float64}
    convergence_code::Bool
    n_cond::Int
    nobs::Int
end

"""
    SARIMA{Fl} <: StateSpaceModel

Main SARIMA model struct following the `fit!(model)` pattern.

# Usage
```julia
model = SARIMA(y, 12; order=PDQ(1,1,1), seasonal=PDQ(0,1,1))
fit!(model)
forecast(model; h=12)
```

# Fields
- `y::Vector{Fl}`: Working copy of the series (possibly transformed).
- `y_orig::Vector{Fl}`: Original series.
- `order::SARIMAOrder`: Model order specification.
- `hyperparameters::HyperParameters{Fl}`: Parameter vectors.
- `system::Union{Nothing,SARIMASystem{Fl}}`: State-space matrices (populated during fit).
- `workspace::Union{Nothing,KalmanWorkspace}`: Pre-allocated Kalman filter workspace.
- `results::Union{Nothing,SARIMAResults{Fl}}`: Fit results (populated after fit!).
- `xreg::Union{Nothing,NamedMatrix}`: Exogenous regressors.
- `xreg_original::Union{Nothing,NamedMatrix}`: Original xreg before SVD rotation.
- `include_mean::Bool`: Whether to include an intercept term.
- `include_drift::Bool`: Whether to include a drift term.
- `lambda::Union{Real,Nothing}`: Box-Cox transformation parameter.
- `biasadj::Union{Bool,Nothing}`: Bias adjustment flag for Box-Cox.
- `method::Symbol`: Estimation method (`:css`, `:ml`, `:css_ml`).
- `transform_pars::Bool`: Whether to use parameter transformations.
- `SSinit::Symbol`: Covariance initialization method.
- `optim_method::Symbol`: Optimization method.
- `optim_control::Dict`: Optimizer control parameters.
- `kappa::Real`: Prior variance for diffuse states.
- `n_cond::Int`: Number of conditioning observations.
- `fixed::Union{Nothing,AbstractArray}`: Fixed parameter specification.
- `init::Union{Nothing,AbstractArray}`: Initial parameter values.
"""
mutable struct SARIMA{Fl<:AbstractFloat} <: StateSpaceModel
    y::Vector{Fl}
    y_orig::Vector{Fl}
    order::SARIMAOrder
    hyperparameters::HyperParameters{Fl}
    system::Union{Nothing,SARIMASystem{Fl}}
    workspace::Union{Nothing,KalmanWorkspace}
    results::Union{Nothing,SARIMAResults{Fl}}
    xreg::Union{Nothing,NamedMatrix}
    xreg_original::Union{Nothing,NamedMatrix}
    include_mean::Bool
    include_drift::Bool
    lambda::Union{Real,Nothing}
    biasadj::Union{Bool,Nothing}
    method::Symbol
    transform_pars::Bool
    SSinit::Symbol
    optim_method::Symbol
    optim_control::Dict
    kappa::Real
    n_cond::Int
    fixed::Union{Nothing,AbstractArray}
    init::Union{Nothing,AbstractArray}
end

"""
    ArimaPredictions

Result container for `predict_arima` output.

# Fields
- `prediction::Vector{Float64}`: Point forecasts.
- `se::Vector{Float64}`: Standard errors.
- `y::AbstractVector`: Original series.
- `fitted::Vector{Float64}`: Fitted values.
- `residuals::Vector{Float64}`: Residuals.
- `method::String`: Method description.
"""
struct ArimaPredictions
    prediction::Vector{Float64}
    se::Vector{Float64}
    y::AbstractVector
    fitted::Vector{Float64}
    residuals::Vector{Float64}
    method::String
end
