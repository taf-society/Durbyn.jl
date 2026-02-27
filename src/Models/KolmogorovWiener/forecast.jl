"""
    forecast(model::KWFilterResult; h, level, nlags, ridge)

Compute Wiener-optimal multi-step-ahead forecasts from a fitted KW filter result.

Uses the Kolmogorov-Wiener prediction formula: for each horizon j, solve

    b_j = Γ⁻¹ g_j

where Γ is the p×p Toeplitz autocovariance matrix of the stationary component and
g_j is the cross-covariance vector between the regressor block and the future value
at lag j. Each horizon gets its own optimal weight vector (direct multi-step prediction).

For integrated series (d ≥ 1 or D ≥ 1), forecasts are computed on the fully differenced
(stationary) series using `(1-L)^d (1-L^s)^D`, then cumulated back to levels through
the inverse operations — first undoing nonseasonal differences, then seasonal differences.

# Arguments
- `model::KWFilterResult`: Result from [`kolmogorov_wiener`](@ref)

# Keyword Arguments
- `h::Int`: Forecast horizon (required)
- `level::Vector{<:Real}=[80, 95]`: Confidence levels for prediction intervals
- `nlags::Union{Int,Nothing}=nothing`: Number of lags for the regressor block.
  Default uses all available observations (min of T and length of autocovariance).
- `ridge::Float64=0.0`: Tikhonov regularization parameter added to the diagonal of Γ

# Returns
A [`Forecast`](@ref) struct with point forecasts, prediction intervals, fitted values
and residuals.

# Examples
```julia
y = air_passengers()
r = kolmogorov_wiener(y, :hp; m=12)
fc = forecast(r; h=12)
fc.mean          # 12 point forecasts
fc.upper[:, 2]   # 95% upper bounds
```
"""
function forecast(model::KWFilterResult;
    h::Int,
    level::Vector{<:Real} = [80, 95],
    nlags::Union{Int,Nothing} = nothing,
    ridge::Float64 = 0.0)

    h >= 1 || throw(ArgumentError("Forecast horizon h must be >= 1, got $h"))
    ridge >= 0.0 || throw(ArgumentError("Ridge parameter must be >= 0, got $ridge"))
    all(l -> 0 < l < 100, level) || throw(ArgumentError("Confidence levels must be in (0, 100)"))

    y = model.y
    gamma = model.gamma
    d = model.d
    T = length(y)

    # Extract seasonal structure from the ARIMA model
    d_ns, D_s, s = _extract_differencing(model)
    p = _effective_nlags(T, d_ns, D_s, s, nlags)

    if d == 0
        fc_mean, fc_var = _wiener_forecast_stationary(y, gamma, h, p, ridge)
    else
        fc_mean, fc_var = _wiener_forecast_integrated(y, gamma, d_ns, D_s, s, h, p, ridge)
    end

    # Clamp variance to non-negative (numerical noise)
    fc_var = max.(fc_var, 0.0)

    # Build prediction intervals
    n_levels = length(level)
    upper = zeros(h, n_levels)
    lower = zeros(h, n_levels)
    for (j, lev) in enumerate(level)
        alpha = 1.0 - lev / 100.0
        z = dist_quantile(Normal(), 1.0 - alpha / 2.0)
        se = sqrt.(fc_var)
        upper[:, j] = fc_mean .+ z .* se
        lower[:, j] = fc_mean .- z .* se
    end

    # Use the ARIMA model's in-sample fitted values so the Forecast's fitted/residuals
    # are at the same scale as the original series (and forecast means), regardless of
    # whether the KW filter extracted cycle or trend.
    if !isnothing(model.arima_model)
        arima_fv = Float64.(fitted(model.arima_model))
        # If Box-Cox was applied, fitted values are on the transformed scale — invert.
        bc_lambda = model.arima_model.lambda
        if !isnothing(bc_lambda)
            arima_fv = inv_box_cox(arima_fv; lambda=bc_lambda)
        end
        arima_res = y .- arima_fv
    else
        arima_fv = fitted(model)
        arima_res = residuals(model)
    end

    return Forecast(
        model,
        "Kolmogorov-Wiener",
        fc_mean,
        level,
        y,
        upper,
        lower,
        arima_fv,
        arima_res,
    )
end

"""
    forecast(model::Decomposition; h, level, nlags, ridge)

Forecast the original series from a KW decomposition using Wiener-optimal
prediction on the underlying ARIMA structure, then return the result as
fitted trend + future values.

Constructs a minimal `KWFilterResult` from the decomposition metadata and
delegates to `forecast(::KWFilterResult; ...)`.

Only supports decompositions with `method == :kw`.
"""
function forecast(model::Decomposition;
    h::Int,
    level::Vector{<:Real} = [80, 95],
    nlags::Union{Int,Nothing} = nothing,
    ridge::Float64 = 0.0)

    model.method === :kw || throw(ArgumentError(
        "forecast(::Decomposition) currently only supports method=:kw, got $(model.method)"))

    md = model.metadata

    # Build a KWFilterResult for the trend component
    r_trend = KWFilterResult(
        model.trend,           # filtered = trend
        zeros(0, 0),           # weights not needed for forecast
        Float64[],             # ideal_coefs not needed for forecast
        md[:filter_type],
        md[:arima_model],
        md[:gamma],
        md[:d],
        model.data,
        :trend,
        md[:params],
    )

    return forecast(r_trend; h=h, level=level, nlags=nlags, ridge=ridge)
end

"""
    _extract_differencing(model::KWFilterResult) -> (d_ns, D_s, s)

Extract nonseasonal integration order `d_ns`, seasonal integration order `D_s`,
and seasonal period `s` from the ARIMA model stored in a KWFilterResult.

Falls back to treating all integration as nonseasonal if no ARIMA model is available.
"""
function _extract_differencing(model::KWFilterResult)
    if !isnothing(model.arima_model)
        arma = model.arima_model.arma
        d_ns = arma[6]   # nonseasonal d
        D_s = arma[7]    # seasonal D
        s = arma[5]      # seasonal period
        return d_ns, D_s, s
    else
        return model.d, 0, 1
    end
end

"""
    _effective_nlags(T, d_ns, D_s, s, nlags) -> Int

Determine the number of lags to use for the regressor block.
Clamps to [1, T_eff] where T_eff is the length of the fully differenced series:
T_eff = T - D_s * s - d_ns.
"""
function _effective_nlags(T::Int, d_ns::Int, D_s::Int, s::Int, nlags::Union{Int,Nothing})
    T_eff = T - D_s * s - d_ns
    T_eff >= 1 || throw(ArgumentError(
        "Series too short for differencing (T=$T, d=$d_ns, D=$D_s, s=$s)"))
    if isnothing(nlags)
        return T_eff
    end
    nlags >= 1 || throw(ArgumentError("nlags must be >= 1, got $nlags"))
    return min(nlags, T_eff)
end

"""
    _apply_differencing(y, d_ns, D_s, s) -> Vector{Float64}

Apply `(1-L^s)^D_s` then `(1-L)^d_ns` to produce the stationary component.
"""
function _apply_differencing(y::AbstractVector, d_ns::Int, D_s::Int, s::Int)
    z = Float64.(y)
    for _ in 1:D_s
        z = z[(s+1):end] .- z[1:(end-s)]
    end
    for _ in 1:d_ns
        z = diff(z)
    end
    return z
end

"""
    _build_prediction_rhs(gamma, step, p) -> Vector{Float64}

Build the cross-covariance vector g_h where g_h[i] = γ(step + i - 1) for i = 1:p.
This is the covariance between y_{T-i+1} and y_{T+step}.
"""
function _build_prediction_rhs(gamma::AbstractVector, step::Int, p::Int)
    g = zeros(p)
    maxlag = length(gamma) - 1
    for i in 1:p
        lag = step + i - 1
        g[i] = lag <= maxlag ? gamma[lag + 1] : 0.0
    end
    return g
end

"""
    _wiener_forecast_stationary(y, gamma, h, p, ridge) -> (mean, var)

Stationary (d=0) Wiener prediction. For each horizon j=1:h, solves Γ b_j = g_j.

Returns vectors of length h: point forecasts and forecast MSE.
"""
function _wiener_forecast_stationary(y::AbstractVector, gamma::AbstractVector,
    h::Int, p::Int, ridge::Float64)

    T = length(y)

    # Regressor: z = [y_T, y_{T-1}, ..., y_{T-p+1}]
    z = y[T:-1:max(T - p + 1, 1)]
    if length(z) < p
        z = vcat(z, zeros(p - length(z)))
    end

    # Build Toeplitz Γ (p × p) and Cholesky-factor once
    Gamma = build_toeplitz_gamma_hat(gamma, p)
    if ridge > 0.0
        Gamma = Symmetric(Matrix(Gamma) + ridge * I)
    end
    C = cholesky(Gamma)

    fc_mean = zeros(h)
    fc_var = zeros(h)

    for j in 1:h
        g_j = _build_prediction_rhs(gamma, j, p)
        b_j = C \ g_j
        fc_mean[j] = dot(b_j, z)
        fc_var[j] = gamma[1] - dot(g_j, b_j)
    end

    return fc_mean, fc_var
end

"""
    _wiener_forecast_integrated(y, gamma, d_ns, D_s, s, h, p, ridge) -> (mean, var)

Integrated Wiener prediction. Applies `(1-L^s)^D_s (1-L)^d_ns` to produce the
stationary component, forecasts it, then cumulates back to levels.

Returns vectors of length h: point forecasts and forecast MSE at level scale.
"""
function _wiener_forecast_integrated(y::AbstractVector, gamma::AbstractVector,
    d_ns::Int, D_s::Int, s::Int, h::Int, p::Int, ridge::Float64)

    # Proper differencing: seasonal then nonseasonal
    z = _apply_differencing(y, d_ns, D_s, s)

    # Forecast the differenced series
    diff_mean, _ = _wiener_forecast_stationary(z, gamma, h, p, ridge)

    # Build B matrix (h × p) of weight vectors for variance cumulation
    Gamma = build_toeplitz_gamma_hat(gamma, p)
    if ridge > 0.0
        Gamma = Symmetric(Matrix(Gamma) + ridge * I)
    end
    C = cholesky(Gamma)

    B_mat = zeros(h, p)
    for j in 1:h
        g_j = _build_prediction_rhs(gamma, j, p)
        B_mat[j, :] = C \ g_j
    end

    # Cumulate forecasts back to levels
    fc_mean = _cumulate_forecasts(y, diff_mean, d_ns, D_s, s)

    # Cumulate variance
    fc_var = _cumulate_variance(gamma, B_mat, d_ns, D_s, s, h, p)

    return fc_mean, fc_var
end

"""
    _cumulate_forecasts(y, diff_fc, d_ns, D_s, s) -> Vector{Float64}

Convert forecasts of the fully differenced series back to level forecasts.

Undoes nonseasonal differencing first (cumsum with anchors from the seasonally
differenced series), then undoes seasonal differencing (adding back values
s steps behind from the original series or previously reconstructed forecasts).
"""
function _cumulate_forecasts(y::AbstractVector, diff_fc::AbstractVector,
    d_ns::Int, D_s::Int, s::Int)

    h = length(diff_fc)
    fc = copy(diff_fc)

    # Build chain of seasonally differenced series for anchoring
    seasonal_diffs = Vector{Vector{Float64}}(undef, D_s + 1)
    seasonal_diffs[1] = Float64.(y)
    for k in 1:D_s
        sd = seasonal_diffs[k]
        seasonal_diffs[k + 1] = sd[(s+1):end] .- sd[1:(end-s)]
    end

    # w = fully seasonally differenced series (before nonseasonal diffs)
    w = seasonal_diffs[D_s + 1]

    # Build chain of nonseasonally differenced w for anchoring
    nonseasonal_diffs = Vector{Vector{Float64}}(undef, d_ns + 1)
    nonseasonal_diffs[1] = w
    for k in 1:d_ns
        nonseasonal_diffs[k + 1] = diff(nonseasonal_diffs[k])
    end

    # Undo nonseasonal differencing (d_ns times)
    for k in d_ns:-1:1
        anchor = nonseasonal_diffs[k][end]
        fc = cumsum(fc) .+ anchor
    end
    # Now fc = forecasts of w (the seasonally differenced series)

    # Undo seasonal differencing (D_s times)
    for k in D_s:-1:1
        prev_series = seasonal_diffs[k]  # series at level (k-1)
        T_prev = length(prev_series)

        new_fc = zeros(h)
        for j in 1:h
            back_idx = T_prev + j - s
            if back_idx >= 1 && back_idx <= T_prev
                # Value is known from historical data
                new_fc[j] = fc[j] + prev_series[back_idx]
            elseif back_idx > T_prev
                # Value is a previously reconstructed forecast
                new_fc[j] = fc[j] + new_fc[back_idx - T_prev]
            else
                # Not enough history (edge case)
                new_fc[j] = fc[j]
            end
        end
        fc = new_fc
    end

    return fc
end

"""
    _cumulate_variance(gamma, B_mat, d_ns, D_s, s, h, p) -> Vector{Float64}

Compute forecast MSE at the level scale after cumulation.

The h×h error covariance of the differenced forecasts is:
    Σ_diff[j,k] = γ(|j-k|) - b_j' g_k

The cumulation matrix C = C_seasonal^{D_s} * C_nonseasonal^{d_ns} transforms
differenced errors to level errors:
- C_nonseasonal = S^{d_ns} where S is the lower-triangular ones (cumsum operator)
- C_seasonal[i,j] = 1 iff (i-j) is a non-negative multiple of s

    Var(level) = diag(C Σ_diff C')
"""
function _cumulate_variance(gamma::AbstractVector, B_mat::AbstractMatrix,
    d_ns::Int, D_s::Int, s::Int, h::Int, p::Int)

    maxlag = length(gamma) - 1

    # Build Σ_diff: h × h covariance of differenced forecast errors
    Sigma_diff = zeros(h, h)
    for j in 1:h
        for k in j:h
            lag = abs(j - k)
            gamma_jk = lag <= maxlag ? gamma[lag + 1] : 0.0
            g_k = _build_prediction_rhs(gamma, k, p)
            cross = dot(B_mat[j, :], g_k)
            Sigma_diff[j, k] = gamma_jk - cross
            Sigma_diff[k, j] = Sigma_diff[j, k]
        end
    end

    # Build cumulation matrix C = C_seasonal * C_nonseasonal
    # Nonseasonal: S^{d_ns} where S is lower-triangular ones
    S_ns = LowerTriangular(ones(h, h))
    C = Matrix{Float64}(I, h, h)
    for _ in 1:d_ns
        C = Matrix(S_ns) * C
    end

    # Seasonal: S_s^{D_s} where S_s[i,j] = 1 iff (i-j) % s == 0 and i >= j
    if D_s > 0
        S_s = zeros(h, h)
        for i in 1:h
            for j in 1:i
                if (i - j) % s == 0
                    S_s[i, j] = 1.0
                end
            end
        end
        for _ in 1:D_s
            C = S_s * C
        end
    end

    # Cumulated covariance
    Sigma_level = C * Sigma_diff * C'

    return diag(Sigma_level)
end
