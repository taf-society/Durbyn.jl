function initialize_arima_state(phi::Vector{Float64}, theta::Vector{Float64}, Delta::Vector{Float64}; kappa::Float64=1e6)
    p = length(phi)
    q = length(theta)
    r = max(p, q + 1)
    d = length(Delta)
    rd = r + d
    Z = vcat([1.0], zeros(r - 1), Delta)
    T = zeros(Float64, rd, rd)
    if p > 0
        for i = 1:p
            T[i, 1] = phi[i]
        end
    end
    if r > 1
        for i = 2:r
            T[i-1, i] = 1.0
        end
    end
    if d > 0
        T[r+1, :] = Z'
        if d > 1
            for i = 2:d
                T[r+i, r+i-1] = 1.0
            end
        end
    end
    if q < r - 1
        theta = vcat(theta, zeros(r - 1 - q))
    end
    R = vcat([1.0], theta, zeros(d))
    V = R * R'
    h = 0.0
    a = zeros(Float64, rd)
    P = zeros(Float64, rd, rd)
    Pn = zeros(Float64, rd, rd)
    Pn[1:r, 1:r] = _solve_stationary_covariance(phi, theta)
    if d > 0
        for i = r+1:r+d
            Pn[i, i] = kappa
        end
    end
    return ArimaStateSpace(phi, theta, Delta, Z, a, P, T, V, h, Pn)
end

function update_arima(mod::Union{ArimaStateSpace,SARIMASystem}, phi, theta; kappa::Float64=1e6)
    p = length(phi)
    q = length(theta)
    r = max(p, q + 1)
    rd = size(mod.T, 1)
    d = rd - r

    mod.phi = phi
    mod.theta = theta

    # Update transition matrix T: AR coefficients in first column
    if p > 0
        mod.T[1:p, 1] .= phi
    end

    # Update process noise covariance V = R * R' where R = [1, θ₁, ..., θ_{r-1}, 0, ..., 0]
    theta_full = q < r - 1 ? vcat(theta, zeros(r - 1 - q)) : theta
    R = vcat([1.0], theta_full, zeros(d))
    @inbounds for i in 1:rd
        for j in 1:rd
            mod.V[i, j] = R[i] * R[j]
        end
    end

    # Reset full Pn: ARMA covariance block + diffuse initialization
    Pn = mod.Pn
    Pn[1:r, 1:r] .= _solve_stationary_covariance(phi, theta)
    # Reset diffuse block (cross-covariance + prior variance)
    if d > 0
        Pn[1:r, r+1:rd] .= 0.0
        Pn[r+1:rd, 1:r] .= 0.0
        Pn[r+1:rd, r+1:rd] .= 0.0
        for i in r+1:rd
            Pn[i, i] = kappa
        end
    end

    mod.a .= 0.0
    return mod
end

"""
    SARIMA(y, m; order, seasonal, kwargs...)

Construct a SARIMA model ready to be fitted with `fit!`.

# Arguments
- `y::AbstractVector`: Time series data.
- `m::Int`: Seasonal period (1 for non-seasonal).

# Keyword Arguments
- `order::PDQ`: Non-seasonal (p,d,q) orders. Default `PDQ(0,0,0)`.
- `seasonal::PDQ`: Seasonal (P,D,Q) orders. Default `PDQ(0,0,0)`.
- `xreg`: Optional exogenous regressors (`NamedMatrix` or `nothing`).
- `include_mean::Bool`: Include intercept for undifferenced series.
- `include_drift::Bool`: Include drift term.
- `lambda`: Box-Cox parameter.
- `biasadj::Bool`: Bias adjustment for Box-Cox.
- `method::Symbol`: `:css`, `:ml`, or `:css_ml`.
- `transform_pars::Bool`: Use parameter transformations.
- `fixed`: Fixed parameter values.
- `init`: Initial parameter values.
- `optim_method::Symbol`: Optimizer (`:bfgs`, `:nelder_mead`, `:lbfgsb`).
- `optim_control::Dict`: Optimizer control parameters.
- `kappa::Real`: Prior variance for diffuse states.
"""
function SARIMA(
    y::AbstractVector,
    m::Int;
    order::PDQ=PDQ(0, 0, 0),
    seasonal::PDQ=PDQ(0, 0, 0),
    xreg::Union{Nothing,NamedMatrix}=nothing,
    include_mean::Bool=true,
    include_drift::Bool=false,
    lambda::Union{Real,Nothing}=nothing,
    biasadj::Union{Bool,Nothing}=nothing,
    method::Symbol=:css_ml,
    transform_pars::Bool=true,
    fixed::Union{Nothing,AbstractArray}=nothing,
    init::Union{Nothing,AbstractArray}=nothing,
    optim_method::Symbol=:bfgs,
    optim_control::Dict=Dict(),
    kappa::Real=1e6,
)
    _check_arg(method, (:css_ml, :ml, :css), "method")

    x = [ismissing(yi) ? NaN : Float64(yi) for yi in y]
    y_orig = copy(x)
    sarima_order = SARIMAOrder(order, seasonal, m)
    narma = n_arma_params(sarima_order)

    if isnothing(fixed)
        fixed_vec = fill(NaN, narma)
    else
        fixed_vec = Float64.(fixed)
    end

    hp = HyperParameters{Float64}(
        zeros(Float64, narma),
        zeros(Float64, narma),
        Bool[],
        String[],
        fixed_vec,
    )

    return SARIMA{Float64}(
        x,
        y_orig,
        sarima_order,
        hp,
        nothing,
        nothing,
        nothing,
        xreg,
        xreg,
        include_mean,
        include_drift,
        lambda,
        biasadj,
        method,
        transform_pars,
        optim_method,
        optim_control,
        kappa,
        0,
        fixed,
        init,
    )
end

function initialize_system!(model::SARIMA{Fl}) where {Fl}
    order = model.order
    Delta = build_delta(order)

    hp = model.hyperparameters
    phi, theta = transform_arima_parameters(hp.constrained, order, false)

    ss = initialize_arima_state(
        phi, theta, Delta;
        kappa=Float64(model.kappa),
    )

    model.system = SARIMASystem{Fl}(
        Fl.(ss.phi), Fl.(ss.theta), Fl.(ss.Delta),
        Fl.(ss.Z), Fl.(ss.T), Fl.(ss.V),
        Fl(ss.h), Fl.(ss.a), Fl.(ss.P), Fl.(ss.Pn),
    )
    return model
end

function update_system!(model::SARIMA, phi::Vector{Float64}, theta::Vector{Float64})
    sys = model.system
    update_arima(sys, phi, theta; kappa=Float64(model.kappa))
    return model
end
