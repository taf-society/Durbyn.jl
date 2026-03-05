# --- State prediction: Harvey 1989 §3.3.3 companion form ---
#
# Computes a_{t|t-1} = T * a_{t-1} for the SARIMA companion-form transition matrix.
# The state vector has two blocks:
#   [1:r]     ARMA companion block — shift register + AR coefficients
#   [r+1:rd]  Differencing block   — observation equation + shift register
#
# Signature preserved for simulate.jl compatibility.
function state_prediction!(anew::AbstractArray, a::AbstractArray,
                           p::Int, r::Int, d::Int, rd::Int,
                           phi::AbstractArray, delta::AbstractArray)
    # ARMA companion block (Harvey 1989 eq 3.3.1):
    #   rows 1..r-1: shift a[i+1] forward, plus AR contribution phi[i]*a[1]
    #   row r:       AR contribution only (no shift source)
    @inbounds for i in 1:r
        anew[i] = (i < r ? a[i + 1] : 0.0) + (i <= p ? phi[i] * a[1] : 0.0)
    end

    # Differencing block (Harvey 1989 eq 3.3.6):
    #   row r+1:     Z'*a = a[1] + Delta'*a[r+1:rd]
    #   rows r+2..rd: shift register
    if d > 0
        anew[r + 1] = a[1]
        @inbounds for i in 1:d
            anew[r + 1] += delta[i] * a[r + i]
        end
        @inbounds for i in (r + 2):rd
            anew[i] = a[i - 1]
        end
    end
end

# --- Covariance prediction: Durbin & Koopman 2012 eq (4.16) ---
#
# P_{t|t-1} = T * P_{t-1|t-1} * T' + V
#
# Unified for all models (with or without differencing). The transition matrix T
# and noise covariance V already encode the full state-space structure.
function _predict_covariance!(Pnew::Matrix{Float64}, T::AbstractMatrix,
                              P::Matrix{Float64}, V::Matrix{Float64},
                              work::Matrix{Float64})
    mul!(work, T, P)      # work = T * P
    mul!(Pnew, work, T')  # Pnew = T * P * T'
    Pnew .+= V            # Pnew += V (process noise covariance)
    return nothing
end

# --- Kalman filter: Durbin & Koopman 2012 Algorithm 4.3 ---
#
# Computes the concentrated log-likelihood for a Gaussian state-space model:
#   ℓ_c = -n/2 * log(σ²) - 1/2 * Σ log(F_t)
# where σ² = (1/n) Σ v_t²/F_t  (Harvey 1989 eq 3.3.16).
#
# Returns sufficient statistics [Σ v²/F, Σ log(F), n_valid] so the caller
# can form the likelihood.
function compute_arima_likelihood(
    y::Vector{Float64},
    model::Union{ArimaStateSpace,SARIMASystem},
    update_start::Int,
    give_resid::Bool;
    workspace::Union{KalmanWorkspace,Nothing}=nothing
)
    Z = model.Z
    T = model.T
    V = model.V
    a = model.a
    P = model.P
    Pnew = model.Pn

    n = length(y)
    rd = length(a)
    d = length(model.Delta)
    r = rd - d

    # Likelihood accumulators
    sum_sq_resid = 0.0
    sum_log_gain = 0.0
    n_valid = 0

    # Workspace: pre-allocated buffers for the filter
    if isnothing(workspace)
        anew = zeros(rd)
        M = zeros(rd)
        work = zeros(rd, rd)
        standardized_residuals = give_resid ? zeros(n) : nothing
    else
        anew = workspace.anew
        M = workspace.M
        work = workspace.work
        if give_resid
            if isnothing(workspace.standardized_residuals) || length(workspace.standardized_residuals) != n
                standardized_residuals = zeros(n)
            else
                standardized_residuals = workspace.standardized_residuals
                fill!(standardized_residuals, 0.0)
            end
        else
            standardized_residuals = nothing
        end
    end

    # Diffuse threshold (D&K 2012 §5.2): for models with differencing, the prior
    # variance of diffuse states is kappa (≈ 1e6). Observations whose innovation
    # variance F_t ≥ kappa/100 are still in the diffuse phase and excluded from
    # the concentrated likelihood.
    if d > 0
        diffuse_threshold = maximum(Pnew[r + i, r + i] for i in 1:d) / 100
    else
        diffuse_threshold = Inf  # stationary model: all observations contribute
    end

    @inbounds for l = 1:n
        # --- Prediction step: D&K eq (4.16) ---
        mul!(anew, T, a)  # a_{t|t-1} = T * a_{t-1}

        if l > update_start + 1
            _predict_covariance!(Pnew, T, P, V, work)  # P_{t|t-1} = T*P*T' + V
        end

        if !isnan(y[l])
            # --- Innovation: D&K eq (4.14) ---
            resid = y[l] - dot(Z, anew)  # v_t = y_t - Z' * a_{t|t-1}

            # --- Innovation variance: F_t = Z' * P_{t|t-1} * Z ---
            mul!(M, Pnew, Z)     # M = P_{t|t-1} * Z (Kalman gain numerator)
            gain = dot(Z, M)     # F_t = Z' * M

            # --- Likelihood accumulation (Harvey 1989 eq 3.3.16) ---
            if gain < diffuse_threshold
                n_valid += 1
                sum_sq_resid += resid^2 / gain
                sum_log_gain += gain > 0 ? log(gain) : NaN
            end

            # --- Standardized residuals ---
            if give_resid
                standardized_residuals[l] = gain > 0 ? resid / sqrt(gain) : NaN
            end

            # --- State update: a_{t} = a_{t|t-1} + M * v_t / F_t  (D&K eq 4.15) ---
            inv_gain = 1.0 / gain
            for i in 1:rd
                a[i] = anew[i] + M[i] * resid * inv_gain
            end

            # --- Covariance update (rank-1 downdate): P_t = P_{t|t-1} - M*M'/F_t ---
            for i in 1:rd
                for j in 1:rd
                    P[i, j] = Pnew[i, j] - M[i] * M[j] * inv_gain
                end
            end
        else
            # Missing observation: skip update, copy prediction forward (D&K §4.10)
            a .= anew
            copyto!(P, Pnew)
            if give_resid
                standardized_residuals[l] = NaN
            end
        end
    end

    result_stats = (sum_sq_resid, sum_log_gain, Float64(n_valid))

    if give_resid
        return (stats = result_stats, residuals = standardized_residuals)
    else
        return result_stats
    end
end

# --- Forecast: Durbin & Koopman 2012 §4.7 ---
#
# Iterates the prediction step without updates to produce h-step-ahead forecasts.
#   ŷ_{T+j} = Z' * a_{T+j|T}
#   Var(ŷ_{T+j}) = Z' * P_{T+j|T} * Z
function kalman_forecast(n_ahead::Int, mod::Union{ArimaStateSpace,SARIMASystem}; update::Bool=false)
    Z = mod.Z
    T = mod.T
    V = mod.V
    a = copy(mod.a)
    P = copy(mod.P)

    rd = length(a)

    forecasts = Vector{Float64}(undef, n_ahead)
    variances = Vector{Float64}(undef, n_ahead)

    anew = similar(a)
    Pnew = similar(P)
    work = similar(P)

    for l in 1:n_ahead
        # Predict state
        mul!(anew, T, a)
        a .= anew

        # Point forecast
        forecasts[l] = dot(Z, a)

        # Predict covariance
        _predict_covariance!(Pnew, T, P, V, work)

        # Forecast variance
        variances[l] = dot(Z, Pnew, Z)

        P .= Pnew
    end

    result = (pred = forecasts, var = variances)
    if update
        updated_mod = deepcopy(mod)
        updated_mod.a .= a
        updated_mod.P .= P
        result = merge(result, (; mod = updated_mod))
    end
    return result
end
