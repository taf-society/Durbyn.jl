"""
    ArarmaModel

Struct holding the fitted ARARMA model.

# Fields
- `psi::Vector{Float64}`
  Composite AR prefilter polynomial (Ψ) accumulated during the
  non-stationary AR reduction stage (ARAR). Encodes the composed filters used
  to remove long-memory/persistent behavior before the short-memory ARMA fit.

- `Sbar::Float64`
  Sample mean used to form the AR intercept
  (`c = (1 - sum(lag_phi)) * Sbar`).

- `gamma::Vector{Float64}`
  Sample autocovariances `γ(0:max_lag)` of the
  prefiltered & demeaned series, used for selecting the best AR lags.

- `lag_phi::Vector{Float64}`
  AR coefficients (4 terms) associated with the chosen lag
  tuple in `best_lag` from the Yule-Walker equations. Used to build the
  composite filter xi via `compute_xi(psi, lag_phi, best_lag)`.

- `sigma2::Float64`
  Innovation variance (σ²) from the short-memory ARMA(p, q) stage.

- `best_lag::Tuple{Int, Int, Int, Int}`
  Selected AR lag tuple `(1, i, j, k)` minimizing
  the implied variance proxy in the AR stage.

- `y_original::Vector{Float64}`
  The original input series. Fitted values and residuals are aligned back to this.

- `arma_phi::Vector{Float64}`
  AR coefficients from the ARMA(p, q) fit on AR residuals.

- `arma_theta::Vector{Float64}`
  MA coefficients (θ) from the ARMA(p, q) fit on AR residuals.

- `ma_order::Int`, `ar_order::Int`
  Orders `q` and `p` used in the ARMA stage.

- `aic::Float64`, `bic::Float64`, `loglik::Float64`
  Information criteria (AIC, BIC) and log-likelihood from the ARMA stage.

"""
struct ArarmaModel
    psi::Vector{Float64}
    Sbar::Float64
    gamma::Vector{Float64}
    lag_phi::Vector{Float64}
    sigma2::Float64
    best_lag::Tuple{Int, Int, Int, Int}
    y_original::Vector{Float64}
    arma_phi::Vector{Float64}
    arma_theta::Vector{Float64}
    ma_order::Int
    ar_order::Int
    aic::Float64
    bic::Float64
    loglik::Float64
end

function Base.show(io::IO, model::ArarmaModel)
    println(io, "ARARMA Model Summary")
    println(io, "--------------------")
    println(io, "Number of observations: ", length(model.y_original))
    println(io, "Model order: ARARMA($(model.ar_order), $(model.ma_order))")
    println(io, "Selected AR lags: ", model.best_lag)
    println(io, "Yule-Walker AR coefficients: ", round.(model.lag_phi, digits=4))
    println(io, "ARMA-stage AR coefficients (ϕ): ", round.(model.arma_phi, digits=4))
    println(io, "ARMA-stage MA coefficients (θ): ", round.(model.arma_theta, digits=4))
    println(io, "Innovation variance (σ²): ", round(model.sigma2, digits=4))
    println(io, "Mean of shortened series (S̄): ", round(model.Sbar, digits=4))
    println(io, "Length of memory-shortening filter (Ψ): ", length(model.psi))
    println(io, "Log-likelihood: ", round(model.loglik, digits=2))
    println(io, "AIC: ", round(model.aic, digits=2))
    println(io, "BIC: ", round(model.bic, digits=2))
end



function safe_slice(x::Vector{T}, idxs::AbstractVector{Int}) where T
    out = similar(idxs, T)
    n = length(x)
    for (i, idx) in enumerate(idxs)
        out[i] = (1 <= idx <= n) ? x[idx] : zero(T)
    end
    return out
end


function compute_xi(Ψ::Vector{Float64}, ϕ::Vector{Float64}, lags::NTuple{4, Int})
    _, i, j, k_lag = lags
    xi = [Ψ; zeros(k_lag)]
    ϕ_pad = vcat(ϕ, zeros(4 - length(ϕ)))

    xi .-= ϕ_pad[1] .* vcat(0.0, Ψ, zeros(k_lag - 1))
    xi .-= ϕ_pad[2] .* vcat(zeros(i), Ψ, zeros(k_lag - i))
    xi .-= ϕ_pad[3] .* vcat(zeros(j), Ψ, zeros(k_lag - j))
    xi .-= ϕ_pad[4] .* vcat(zeros(k_lag), Ψ)
    return xi
end

"""
    is_arma_stable(ϕ, θ) -> Bool

Check if ARMA coefficients satisfy stability constraints.
Returns true if sum(|ϕ|) < 1 and sum(|θ|) < 1.
"""
function is_arma_stable(ϕ::Vector{Float64}, θ::Vector{Float64})
    if !isempty(ϕ) && sum(abs.(ϕ)) >= 1.0
        return false
    end
    if !isempty(θ) && sum(abs.(θ)) >= 1.0
        return false
    end
    return true
end

# Fits ARMA(p, q) model using MLE, returns (ϕ, θ, σ²).
# Numerically stable: σ² is log-parametrized.

function fit_arma(p::Int, q::Int, y::Vector{Float64}, options::NelderMeadOptions=NelderMeadOptions())
    n = length(y)
    μ = mean(y)
    y_demeaned = y .- μ
    max_lag = max(p, q)
    function arma_loss(params)
        ϕ = params[1:p]
        θ = params[p+1:p+q]
        logσ2 = params[end]
        σ2 = exp(logσ2)
        ε = zeros(n)
        for t in (max_lag + 1):n
            ar_idx = reverse((t-p):(t-1))
            ma_idx = reverse((t-q):(t-1))
            ar_part = dot(ϕ, safe_slice(y_demeaned, ar_idx))
            ma_part = dot(θ, safe_slice(ε, ma_idx))
            ε[t] = y_demeaned[t] - (ar_part + ma_part)
        end
        # Use effective sample size (excluding burn-in)
        n_eff = n - max_lag
        nll = sum(ε.^2) / σ2 + n_eff * log(σ2 + 1e-8)

        # Log-barrier penalty to enforce stability (sum|coef| < 0.95)
        sum_phi = sum(abs.(ϕ))
        sum_theta = sum(abs.(θ))
        penalty = 0.0
        if sum_phi > 0.95
            penalty += 1e6 * (sum_phi - 0.95)^2
        elseif sum_phi > 0.8
            penalty -= 100.0 * log(0.95 - sum_phi + 1e-8)
        end
        if sum_theta > 0.95
            penalty += 1e6 * (sum_theta - 0.95)^2
        elseif sum_theta > 0.8
            penalty -= 100.0 * log(0.95 - sum_theta + 1e-8)
        end

        return nll + penalty
    end
    init = [zeros(p + q); log(var(y))]

    parscale = max.(abs.(init), 0.1)

    est_params = descaler(
        nmmin(θ -> arma_loss(descaler(θ, parscale)), scaler(init, parscale), options).x_opt,
        parscale
    )

    return (est_params[1:p], est_params[p+1:p+q], exp(est_params[end]))
end

function autocovariances(x::Vector{Float64}, maxlag::Int)
    n = length(x)
    return [dot(x[1:(n-i)], x[(i+1):n]) / n for i in 0:maxlag]
end

"""
    ararma(y::Vector{<:Real};
           max_ar_depth::Int=26,
           max_lag::Int=40,
           p::Int=4, q::Int=1,
           options::NelderMeadOptions=NelderMeadOptions()) -> ArarmaModel

Fit an **ARARMA** model to a univariate numeric series `y`.

# Algorithm (as implemented)

1. **Adaptive non-stationary AR prefilter (ARAR stage).**
   Up to three iterations:
   - For delays `τ = 1:15`, fit OLS AR(1) at lag `τ` and score by a relative
     error criterion Err(τ). If Err(τ*) ≤ 8/n or the estimated coefficient
     is strongly persistent (`φ̂(τ*) ≥ 0.9` with `τ > 2`), filter the series and
     compose the filter into `psi`.
   - If persistence is high (`φ̂ ≥ 0.9`) but `τ* ≤ 2`, fit a 2-lag AR instead,
     filter, and update `psi`.
   - Otherwise, stop early.

2. **Best-lag AR selection.**  
   Compute autocovariances `γ(0:max_lag)` of the prefiltered, demeaned series.
   Over triplets `(1,i,j,k)` with `1 < i < j < k ≤ max_ar_depth`, solve a 4×4
   system to estimate `phi`. Select the tuple minimizing  
   `σ² = γ(0) - φ₁γ(1) - φ₂γ(i) - φ₃γ(j) - φ₄γ(k)`.

3. **Composite AR kernel.**  
   Build `xi = compute_xi(psi, best_phi, best_lag)`.  
   Use intercept `c = (1 - sum(best_phi)) * mean(y')`, where `y'` is the
   prefiltered series.

4. **Residuals from AR stage.**  
   Compute AR-only fitted values starting at index `k = length(xi)` and form
   residuals `r`.

5. **Short-memory ARMA(p, q) by MLE.**  
   Optimize `[phi_AR[1:p]; theta_MA[1:q]; log(sigma2)]` via Nelder-Mead on the
   conditional Gaussian likelihood. Parameterize variance as `log(sigma2)` for
   numerical stability.

6. **Information criteria.**
   Compute `loglik`, `AIC = 2k - 2ℓ`, `BIC = log(n)k - 2ℓ`, with `k = p + q + 1`
   (including σ²) and `n` the effective residual length.

# Returns
An [`ArarmaModel`](@ref) bundling fitted values, residuals, and
forecasting components.

# Assumptions
- Univariate, numeric input with no missing values.
- Prefilter is capped at three reductions and favors strong persistence.
- Best-lag search uses up to 4 AR terms `(1,i,j,k)`.
- The ARMA optimizer uses log-barrier penalties to enforce stability (sum|φ| < 0.95, sum|θ| < 0.95).

# Example
```julia
using Durbyn
using Durbyn.Ararma

y = air_passengers();

m = ararma(y)
ŷ = fitted(m)
r = residuals(m)
fc = forecast(m; h=12, level=[80,95])
plot(fc)
```

Reference
Parzen, E. (1982). ARARMA Models for Time Series Analysis and Forecasting.
Journal of Forecasting, 1(1), 67-82.
"""
function ararma(y::Vector{<:Real}; max_ar_depth::Int=26, max_lag::Int=40, p::Int=4, q::Int=1, 
    options::NelderMeadOptions=NelderMeadOptions())
    Y = copy(y)
    Ψ = [1.0]

    for _ in 1:3
        n = length(y)
        ϕ = [sum(y[(τ+1):n] .* y[1:(n-τ)]) / sum(y[1:(n-τ)].^2) for τ in 1:15]
        err = [sum((y[(τ+1):n] - ϕ[τ]*y[1:(n-τ)]).^2) / sum(y[(τ+1):n].^2) for τ in 1:15]
        τ = argmin(err)
        if err[τ] <= 8/n || (ϕ[τ] >= 0.9 && τ > 2)
            y = y[(τ+1):n] .- ϕ[τ] * y[1:(n-τ)]
            Ψ = [Ψ; zeros(τ)] .- ϕ[τ] .* [zeros(τ); Ψ]
        elseif ϕ[τ] >= 0.9
            A = zeros(2,2)
            A[1,1] = sum(y[2:(n-1)].^2)
            A[1,2] = A[2,1] = sum(y[1:(n-2)] .* y[2:(n-1)])
            A[2,2] = sum(y[1:(n-2)].^2)
            b = [sum(y[3:n] .* y[2:(n-1)]), sum(y[3:n] .* y[1:(n-2)])]
            ϕ_2 = A \ b
            y = y[3:n] .- ϕ_2[1]*y[2:(n-1)] .- ϕ_2[2]*y[1:(n-2)]
            Ψ = vcat(Ψ, 0.0, 0.0) .- ϕ_2[1]*vcat(0.0, Ψ, 0.0) .- ϕ_2[2]*vcat(0.0, 0.0, Ψ)
        else
            break
        end
    end

    Sbar = mean(y)
    X = y .- Sbar
    n = length(X)
    # Autocovariances of already-demeaned series (no need to demean again)
    gamma = [sum(X[1:(n-i)] .* X[(i+1):n]) / n for i in 0:max_lag]

    best_σ2 = Inf
    best_lag = (1, 0, 0, 0)
    best_phi = zeros(4)
    A = fill(gamma[1], 4, 4)
    b = zeros(4)
    for i in 2:(max_ar_depth-2), j in (i+1):(max_ar_depth-1), k in (j+1):max_ar_depth
        A[1,2] = A[2,1] = gamma[i]
        A[1,3] = A[3,1] = gamma[j]
        A[2,3] = A[3,2] = gamma[j-i+1]
        A[1,4] = A[4,1] = gamma[k]
        A[2,4] = A[4,2] = gamma[k-i+1]
        A[3,4] = A[4,3] = gamma[k-j+1]
        b .= [gamma[2], gamma[i+1], gamma[j+1], gamma[k+1]]
        # Skip singular or near-singular matrices
        cond_A = cond(A)
        if !isfinite(cond_A) || cond_A > 1e12
            continue
        end
        ϕ = A \ b
        σ2 = gamma[1] - ϕ[1]*gamma[2] - ϕ[2]*gamma[i+1] - ϕ[3]*gamma[j+1] - ϕ[4]*gamma[k+1]
        if σ2 < best_σ2
            best_σ2 = σ2
            best_phi = copy(ϕ)
            best_lag = (1, i, j, k)
        end
    end

    xi = compute_xi(Ψ, best_phi, best_lag)
    k = length(xi)
    n = length(Y)
    c = (1 - sum(best_phi)) * Sbar

    fitted_vals = fill(NaN, n)
    for t in k:n
        idxs = t .- (1:(k-1))
        ar_pred = -sum(xi[2:k] .* Y[idxs]) + c
        fitted_vals[t] = ar_pred
    end

    residuals = Y .- fitted_vals
    ϕ_ar, θ_ma, σ2_hat = fit_arma(p, q, residuals[k:end], options)

    arma_resid = residuals[k:end]
    n_eff = length(arma_resid)
    k_param = p + q + 1  # +1 for σ²

    loglik = -0.5 * n_eff * (log(2π * σ2_hat) + 1)
    aic = 2 * k_param - 2 * loglik
    bic = log(n_eff) * k_param - 2 * loglik

    return ArarmaModel(Ψ, Sbar, gamma, best_phi, σ2_hat, best_lag, Y, ϕ_ar, θ_ma, q, p, aic, bic, loglik)
end


"""
    fitted(model)

Returns fitted values for the full ARARMA model.
"""
function fitted(model::ArarmaModel)
    y = model.y_original
    xi = compute_xi(model.psi, model.lag_phi, model.best_lag)
    n = length(y)
    k = length(xi)
    p = model.ar_order
    q = model.ma_order
    c = (1 - sum(model.lag_phi)) * model.Sbar
    ϕ = model.arma_phi
    θ = model.arma_theta

    # Only use ARMA coefficients if stable
    use_arma = is_arma_stable(ϕ, θ)

    fitted_vals = fill(NaN, n)
    ar_resid = zeros(n)  # AR-stage residuals
    ε = zeros(n)         # ARMA innovations

    # First compute AR-only fitted values and residuals
    for t in k:n
        idxs = t .- (1:(k-1))
        ar_pred = -sum(xi[2:k] .* safe_slice(y, idxs)) + c
        ar_resid[t] = y[t] - ar_pred
    end

    # Now compute full fitted values incorporating ARMA on AR residuals
    for t in k:n
        idxs = t .- (1:(k-1))
        ar_pred = -sum(xi[2:k] .* safe_slice(y, idxs)) + c

        arma_adj = 0.0
        if use_arma
            # AR part of ARMA: applied to AR residuals
            if t > p
                ar_idxs = (t-p):(t-1)
                arma_adj += dot(ϕ, reverse(safe_slice(ar_resid, ar_idxs)))
            end
            # MA part of ARMA: applied to innovations
            if t > q
                ma_idxs = (t-q):(t-1)
                arma_adj += dot(θ, reverse(safe_slice(ε, ma_idxs)))
            end
        end

        fitted_vals[t] = ar_pred + arma_adj
        ε[t] = y[t] - fitted_vals[t]
    end
    return fitted_vals
end

"""
    residuals(model)

Returns residuals for the ARARMA model.
"""
residuals(model::ArarmaModel) = model.y_original .- fitted(model)

"""
    forecast(model, h; level=[80,95])

Returns h-step-ahead forecasts and confidence intervals.
"""
function forecast(model::ArarmaModel; h::Int, level::Vector{Int}=[80,95])
    y = copy(model.y_original)
    n = length(y)
    xi = compute_xi(model.psi, model.lag_phi, model.best_lag)
    k = length(xi)
    p = model.ar_order
    q = model.ma_order
    ϕ = model.arma_phi
    θ = model.arma_theta
    σ2 = model.sigma2
    Sbar = model.Sbar
    c = (1 - sum(model.lag_phi)) * Sbar

    # Only use ARMA coefficients if stable
    use_arma = is_arma_stable(ϕ, θ)

    # Compute historical AR residuals for ARMA AR part
    ar_resid = zeros(n)
    for t in k:n
        idxs = t .- (1:(k-1))
        ar_pred = -sum(xi[2:k] .* safe_slice(y, idxs)) + c
        ar_resid[t] = y[t] - ar_pred
    end

    y_ext = [y; zeros(h)]
    ar_resid_ext = [ar_resid; zeros(h)]
    ε_ext = zeros(n + h)
    forecasts = zeros(h)

    for t in 1:h
        idx = n + t
        idxs = idx .- (1:(k-1))
        ar_part = -sum(xi[2:k] .* safe_slice(y_ext, idxs)) + c

        arma_adj = 0.0
        if use_arma
            # AR part of ARMA (decays for future)
            ar_arma_idxs = (idx-p):(idx-1)
            arma_adj += dot(ϕ, reverse(safe_slice(ar_resid_ext, ar_arma_idxs)))
            # MA part of ARMA (only for first q steps)
            if t <= q
                ma_idxs = (idx-q):(idx-1)
                arma_adj += dot(θ, reverse(safe_slice(ε_ext, ma_idxs)))
            end
        end

        forecasts[t] = ar_part + arma_adj
        ε_ext[idx] = 0.0
        ar_resid_ext[idx] = 0.0  # Future AR residuals are zero (unknown)
        y_ext[idx] = forecasts[t]
    end

    τ = zeros(h)
    τ[1] = 1.0
    len_xi = length(xi)
    for j in 2:h
        # MA(∞) recursion: τ_j = -Σ τ_{j-i} * ξ_i for i = 1 to min(j-1, len_xi-1)
        max_i = min(j - 1, len_xi - 1)
        τ[j] = -sum(τ[j-i] * xi[i+1] for i in 1:max_i)
    end
    se = sqrt.(σ2 .* [sum(τ[1:j].^2) for j in 1:h])
    z = level .|> l -> quantile(Normal(), 0.5 + l/200)

    upper = reduce(hcat,[forecasts .+ zi .* se for zi in z])
    lower = reduce(hcat,[forecasts .- zi .* se for zi in z])

    fits = fitted(model)
    res = residuals(model)
    method = "Ararma($(model.ar_order), $(model.ma_order))"

    return Forecast(model, method, forecasts, level, model.y_original, upper, lower, fits, res)
end

"""
    auto_ararma(y::Vector{<:Real};
                max_p::Int=4, max_q::Int=2,
                crit::Symbol=:aic,            # or :bic
                max_ar_depth::Int=26,
                max_lag::Int=40,
                options::NelderMeadOptions=NelderMeadOptions()) -> Union{ArarmaModel,Nothing}

Automatic order selection wrapper around [`ararma`](@ref).

# Behavior
- Tries `(p, q)` across `p ∈ 0:max_p`, `q ∈ 0:max_q`, skipping `(0,0)`.
- Calls `ararma(y; p, q, ...)` for each pair, catching failures and continuing.
- Scores by AIC (default) or BIC using the ARMA-stage likelihood.
- Returns the best-scoring [`ArarmaModel`](@ref), or `nothing` if all attempts fail.

# Choosing a criterion
- `:aic` often yields richer models with better short-horizon forecasts.
- `:bic` favors more parsimonious models.

# Example
```julia
using Durbyn
using Durbyn.Ararma

ap = air_passengers();
model = auto_ararma(y; max_p=4, max_q=2, crit=:bic)
ŷ = fitted(m)
r = residuals(m)
fc = forecast(m; h=12, level=[80,95])
plot(fc)
````

References: 
Parzen, E. (1982). *ARARMA Models for Time Series Analysis and Forecasting*. Journal of Forecasting, 1(1), 67-82.
"""
function auto_ararma(y::Vector{<:Real}; max_p::Int=4, max_q::Int=2, 
    crit::Symbol=:aic, max_ar_depth::Int=26, max_lag::Int=40, 
    options::NelderMeadOptions=NelderMeadOptions())

    best_score = Inf
    best_model = nothing
    for p in 0:max_p, q in 0:max_q
        if p == 0 && q == 0
            continue
        end
        try
            model = ararma(y; p=p, q=q, max_ar_depth=max_ar_depth, max_lag=max_lag, options=options)
            score = crit === :aic ? model.aic : model.bic
            if score < best_score
                best_score = score
                best_model = model
            end
        catch err
            @info "Model selection issue: $err"
        end
    end
    return best_model
end
