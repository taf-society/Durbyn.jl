"""
    simulate(model::ArimaFit, h::Int; kwargs...) -> Vector{Float64}

Simulate a future sample path of length `h` from a fitted ARIMA model.

Starting from the final Kalman-filtered state, propagates the state-space
representation forward with stochastic innovations. The regression component
(intercept, drift, exogenous regressors) is added to the state-space output.

# Arguments
- `model::ArimaFit`: A fitted ARIMA model.
- `h::Int`: Number of steps to simulate.

# Keyword Arguments
- `xreg::Union{Nothing,NamedMatrix}=nothing`: Future exogenous regressors
  (must match training columns). Already aligned by the caller (e.g., `forecast`).
- `lambda::Union{Real,Nothing}=nothing`: Box-Cox parameter. If not `nothing`,
  inverse Box-Cox is applied to return values on the original scale.
- `bootstrap::Bool=false`: If `true`, innovations are resampled from the model
  residuals (with replacement, de-meaned). Otherwise Gaussian N(0, σ²).
- `innov::Union{Nothing,AbstractVector}=nothing`: Optional pre-generated
  innovation vector of length `h` (overrides `bootstrap`).

# Returns
A `Vector{Float64}` of length `h` with simulated future values.
"""
function simulate(model::ArimaFit, h::Int;
    xreg::Union{Nothing,NamedMatrix}=nothing,
    lambda::Union{Real,Nothing}=nothing,
    bootstrap::Bool=false,
    innov::Union{Nothing,AbstractVector}=nothing)

    mod = model.model
    phi = mod.phi
    theta = mod.theta
    delta = mod.Delta
    Z = mod.Z

    # Start from the final Kalman-filtered state
    a = copy(mod.a)

    d = length(delta)
    rd = length(a)
    r = rd - d
    p_ar = length(phi)

    R_vec = vcat([1.0], theta, zeros(d))

    if !isnothing(innov)
        length(innov) == h || throw(ArgumentError("innov must have length h=$h"))
        e = Float64.(innov)
    elseif bootstrap
        res = model.residuals
        n_cond = model.n_cond
        valid = filter(!isnan, @view(res[(n_cond+1):end]))
        isempty(valid) && throw(ArgumentError("No valid residuals for bootstrap resampling"))
        valid = valid .- mean(valid)
        e = valid[rand(1:length(valid), h)]
    else
        e = randn(h) .* sqrt(model.sigma2)
    end

    anew = similar(a)
    sim = Vector{Float64}(undef, h)

    @inbounds for t in 1:h
        state_prediction!(anew, a, p_ar, r, d, rd, phi, delta)
        et = e[t]
        for i in 1:rd
            anew[i] += R_vec[i] * et
        end
        sim[t] = dot(Z, anew)
        a .= anew
    end

    coefs = vec(model.coef.data)
    coef_names = model.coef.colnames
    narma = sum(model.arma[1:4])
    ncoefs = length(coefs)

    if ncoefs > narma
        has_intercept = coef_names[narma+1] == "intercept"

        newxreg_data = if !isnothing(xreg) && model.xreg isa NamedMatrix
            align_columns(xreg, model.xreg.colnames).data
        else
            nothing
        end

        usexreg = if has_intercept
            intercept_col = ones(h, 1)
            isnothing(newxreg_data) ? intercept_col : hcat(intercept_col, newxreg_data)
        else
            newxreg_data
        end

        reg_coefs = coefs[(narma+1):ncoefs]
        if !isnothing(usexreg)
            sim .+= vec(usexreg * reg_coefs)
        end
    end

    if !isnothing(lambda)
        sim = inv_box_cox(sim; lambda=lambda, biasadj=false)
    end

    return sim
end
