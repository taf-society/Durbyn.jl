# --- CSS residual computation: Box-Jenkins 2015 §7.1.3 ---
#
# Applies differencing via order.d, order.D, order.s, then computes
# conditional residuals from the ARMA recursion.

function compute_css_residuals!(
    differenced::Vector{Float64},
    residuals::Vector{Float64},
    y::AbstractVector,
    order::SARIMAOrder,
    phi::AbstractVector,
    theta::AbstractVector,
    n_conditioning::Int,
)
    n = length(y)
    length(differenced) == n ||
        throw(ArgumentError("differenced buffer length ($(length(differenced))) must match y length ($n)"))
    length(residuals) == n ||
        throw(ArgumentError("residual buffer length ($(length(residuals))) must match y length ($n)"))

    p = length(phi)
    q = length(theta)

    @inbounds for i in 1:n
        differenced[i] = Float64(y[i])
    end

    # Non-seasonal differencing: (1-B)^d
    @inbounds for _ in 1:order.d
        for l in n:-1:2
            differenced[l] -= differenced[l - 1]
        end
    end

    # Seasonal differencing: (1-B^s)^D
    s = order.s
    @inbounds for _ in 1:order.D
        for l in n:-1:(s + 1)
            differenced[l] -= differenced[l - s]
        end
    end

    @inbounds for i in 1:min(n_conditioning, n)
        residuals[i] = 0.0
    end

    ssq = 0.0
    nu = 0

    @inbounds for l in (n_conditioning + 1):n
        tmp = differenced[l]

        ar_terms = min(p, l - 1)
        for j in 1:ar_terms
            tmp -= phi[j] * differenced[l - j]
        end

        ma_terms = min(q, l - n_conditioning)
        for j in 1:ma_terms
            tmp -= theta[j] * residuals[l - j]
        end

        residuals[l] = tmp
        if !isnan(tmp)
            nu += 1
            ssq += tmp * tmp
        end
    end

    sigma2 = nu > 0 ? ssq / nu : NaN
    return (sigma2 = sigma2, residuals = residuals)
end

function compute_css_residuals(
    y::AbstractArray,
    order::SARIMAOrder,
    phi::AbstractArray,
    theta::AbstractArray,
    n_conditioning::Int,
)
    n = length(y)
    differenced = Vector{Float64}(undef, n)
    residuals = Vector{Float64}(undef, n)
    return compute_css_residuals!(differenced, residuals, y, order, phi, theta, n_conditioning)
end

# Backward-compatible overload for forecast-derived code (arima_rjh, simulate, etc.)
function compute_css_residuals(
    y::AbstractArray,
    arma::Vector{Int},
    phi::AbstractArray,
    theta::AbstractArray,
    n_conditioning::Int,
)
    # Build a minimal SARIMAOrder from the arma vector
    order = SARIMAOrder(arma[1], arma[6], arma[2], arma[3], arma[7], arma[4], arma[5])
    return compute_css_residuals(y, order, phi, theta, n_conditioning)
end

function process_xreg(xreg::Union{NamedMatrix,Nothing}, n::Int)
    if isnothing(xreg)
        xreg_mat = Matrix{Float64}(undef, n, 0)
        n_xreg_cols = 0
        xreg_names = String[]
    else
        if size(xreg.data, 1) != n
            throw(ArgumentError("time series length ($n) does not match xreg row count ($(size(xreg.data, 1)))"))
        end
        xreg_mat = xreg.data
        if !(eltype(xreg_mat) <: Float64)
            xreg_mat = Float64.(xreg_mat)
        end
        n_xreg_cols = size(xreg_mat, 2)
        xreg_names = xreg.colnames
    end
    return xreg_mat, n_xreg_cols, xreg_names
end

# --- Initial parameter estimation (OLS on differenced data) ---
#
# Replaces regress_and_update!. ARMA params start at zero (standard recommendation,
# Box-Jenkins 2015 §7.1). Xreg params from OLS on differenced data.
# Parameter scaling uses max(1, |β|) instead of R's 10*se heuristic.

function compute_initial_params(
    x::AbstractArray,
    xreg::Matrix,
    n_arma::Int,
    n_xreg_cols::Int,
    order::SARIMAOrder,
    Delta::AbstractArray,
)
    init0 = zeros(n_arma)
    parscale = ones(n_arma)

    dx = copy(x)
    dxreg = copy(xreg)
    if order.d > 0
        dx = diff(dx; lag_steps=1, difference_order=order.d)
        dxreg = diff(dxreg; lag_steps=1, difference_order=order.d)
        dx, dxreg = dropmissing(dx, dxreg)
    end
    if order.s > 1 && order.D > 0
        dx = diff(dx; lag_steps=order.s, difference_order=order.D)
        dxreg = diff(dxreg; lag_steps=order.s, difference_order=order.D)
        dx, dxreg = dropmissing(dx, dxreg)
    end

    if length(dx) > size(dxreg, 2)
        try
            fit = Stats.ols(dx, dxreg)
            fit_rank = rank(dxreg)
        catch e
            @warn "Fitting OLS to difference data failed: $e"
            fit = nothing
            fit_rank = 0
        end
    else
        @debug "Not enough observations to fit OLS" length_dx=length(dx) predictors=size(dxreg, 2)
        fit = nothing
        fit_rank = 0
    end

    if fit_rank == 0
        x_clean, xreg_clean = dropmissing(x, xreg)
        fit = Stats.ols(x_clean, xreg_clean)
    end

    n_observations = length(x)
    n_predictors = size(xreg, 2)
    n_missing_rows = 0
    @inbounds for row in 1:n_observations
        row_has_missing = isnan(x[row])
        if !row_has_missing
            for col in 1:n_predictors
                if isnan(xreg[row, col])
                    row_has_missing = true
                    break
                end
            end
        end
        n_missing_rows += row_has_missing
    end
    n_used = (n_observations - n_missing_rows) - length(Delta)
    model_coefs = Stats.coefficients(fit)
    init0 = append!(init0, model_coefs)

    # Parameter scaling: max(1, |β|) for xreg coefficients
    xreg_scales = [max(1.0, abs(c)) for c in model_coefs]
    parscale = append!(parscale, xreg_scales)

    return init0, parscale, n_used
end

# --- Xreg preprocessing: column-wise L2 normalization ---
#
# Scales each xreg column to unit L2 norm for numerical conditioning.
# Returns the scaling factors for undoing the transformation after fitting.

function preprocess_xreg(xreg_mat::Matrix{Float64}, mask_xreg::AbstractVector{Bool})
    n_cols = size(xreg_mat, 2)
    if n_cols == 0
        return xreg_mat, ones(0)
    end

    # Only normalize columns where all xreg params are free
    scales = ones(n_cols)
    xreg_scaled = copy(xreg_mat)

    if all(mask_xreg)
        n_rows = size(xreg_mat, 1)
        @inbounds for col in 1:n_cols
            sum_sq = 0.0
            for row in 1:n_rows
                value = xreg_mat[row, col]
                if isfinite(value)
                    sum_sq += value * value
                end
            end
            if sum_sq > 0.0
                scale = sqrt(sum_sq)
                scales[col] = scale
                inv_scale = inv(scale)
                for row in 1:n_rows
                    xreg_scaled[row, col] = xreg_mat[row, col] * inv_scale
                end
            end
        end
    end

    return xreg_scaled, scales
end

function unscale_xreg_coefs!(coef::AbstractVector, scales::Vector{Float64},
                              xreg_range::UnitRange{Int})
    for (i, idx) in enumerate(xreg_range)
        coef[idx] /= scales[i]
    end
end

# --- Coefficient name formatting ---

function format_coef_names(order::SARIMAOrder, coef::AbstractArray,
                           cn::Vector{String}, n_xreg_cols::Int)
    names = String[]
    if order.p > 0
        append!(names, ["ar$(i)" for i in 1:order.p])
    end
    if order.q > 0
        append!(names, ["ma$(i)" for i in 1:order.q])
    end
    if order.P > 0
        append!(names, ["sar$(i)" for i in 1:order.P])
    end
    if order.Q > 0
        append!(names, ["sma$(i)" for i in 1:order.Q])
    end
    if n_xreg_cols > 0
        append!(names, cn)
    end
    mat = reshape(coef, 1, :)
    return NamedMatrix(mat, names)
end

mutable struct ObjectiveWorkspace
    parameter_buffer::Vector{Float64}
    xreg_linear_predictor::Vector{Float64}
    x_work::Vector{Float64}
    css_differenced::Vector{Float64}
    css_residuals::Vector{Float64}
    phi_coefficients::Vector{Float64}
    theta_coefficients::Vector{Float64}
    nonseasonal_ar_transform_work::Vector{Float64}
    seasonal_ar_transform_work::Vector{Float64}
end

function ObjectiveWorkspace(n_parameters::Int, n_observations::Int, order::SARIMAOrder)
    phi_length = order.s > 0 ? (order.p + order.s * order.P) : order.p
    theta_length = order.s > 0 ? (order.q + order.s * order.Q) : order.q
    return ObjectiveWorkspace(
        Vector{Float64}(undef, n_parameters),
        Vector{Float64}(undef, n_observations),
        Vector{Float64}(undef, n_observations),
        Vector{Float64}(undef, n_observations),
        Vector{Float64}(undef, n_observations),
        Vector{Float64}(undef, phi_length),
        Vector{Float64}(undef, theta_length),
        Vector{Float64}(undef, max(1, order.p)),
        Vector{Float64}(undef, max(1, order.P)),
    )
end

@inline function _prepare_regression_adjusted_series!(
    workspace::ObjectiveWorkspace,
    x::Vector{Float64},
    xreg_mat::Matrix{Float64},
    params::Vector{Float64},
    xreg_range::UnitRange{Int},
)
    mul!(workspace.xreg_linear_predictor, xreg_mat, @view(params[xreg_range]))
    @inbounds @simd for i in eachindex(x)
        workspace.x_work[i] = x[i] - workspace.xreg_linear_predictor[i]
    end
    return workspace.x_work
end

@inline function _scatter_free_parameters!(
    parameter_buffer::Vector{Float64},
    base_parameters::Vector{Float64},
    free_parameters::AbstractVector,
    free_parameter_indices::Vector{Int},
)
    copyto!(parameter_buffer, base_parameters)
    @inbounds for i in eachindex(free_parameter_indices)
        parameter_buffer[free_parameter_indices[i]] = free_parameters[i]
    end
    return parameter_buffer
end

@inline function _as_typed_sarima_system(state::ArimaStateSpace)
    return SARIMASystem{Float64}(
        state.phi,
        state.theta,
        state.Delta,
        state.Z,
        state.T,
        state.V,
        Float64(state.h),
        state.a,
        state.P,
        state.Pn,
    )
end

# --- ML objective function ---
#
# Concentrated Gaussian log-likelihood (Harvey 1989 eq 3.3.16):
#   ℓ_c(θ) = -n/2 * [log(σ²(θ)) + (1/n)Σlog(F_t)]
# We minimize: 0.5 * [log(σ²) + Σlog(F)/n]
#
# Takes explicit arguments instead of capturing mutable closure state.

function _ml_objective(
    free_params::AbstractVector,
    full_coef::Vector{Float64},
    free_parameter_indices::Vector{Int},
    order::SARIMAOrder,
    x::Vector{Float64},
    xreg_mat::Matrix{Float64},
    n_xreg_cols::Int,
    xreg_range::UnitRange{Int},
    objective_workspace::ObjectiveWorkspace,
    mod::Union{ArimaStateSpace,SARIMASystem},
    transform_pars::Bool,
    kappa::Float64,
    kalman_workspace::Union{KalmanWorkspace,Nothing},
)
    parameter_buffer = objective_workspace.parameter_buffer
    _scatter_free_parameters!(parameter_buffer, full_coef, free_params, free_parameter_indices)
    transform_arima_parameters!(
        objective_workspace.phi_coefficients,
        objective_workspace.theta_coefficients,
        parameter_buffer,
        order,
        transform_pars,
        objective_workspace.nonseasonal_ar_transform_work,
        objective_workspace.seasonal_ar_transform_work,
    )

    try
        update_arima(mod, objective_workspace.phi_coefficients, objective_workspace.theta_coefficients; kappa=kappa)
    catch
        return typemax(Float64)
    end

    x_work = if n_xreg_cols > 0
        _prepare_regression_adjusted_series!(
            objective_workspace,
            x,
            xreg_mat,
            parameter_buffer,
            xreg_range,
        )
    else
        x
    end

    resss = compute_arima_likelihood(x_work, mod, 0, false; workspace=kalman_workspace)

    nu = resss[3]
    nu <= 0 && return typemax(Float64)
    s2 = resss[1] / nu
    (s2 < 0 || isnan(s2) || s2 == Inf) && return typemax(Float64)
    result = 0.5 * (log(s2) + resss[2] / nu)
    return isnan(result) || result == Inf ? typemax(Float64) : result
end

# --- CSS objective function ---
#
# Conditional sum of squares (Box-Jenkins 2015 §7.1.3):
# Minimize: 0.5 * log(σ²_CSS)

function _css_objective(
    free_params::AbstractVector,
    full_coef::Vector{Float64},
    free_parameter_indices::Vector{Int},
    order::SARIMAOrder,
    x::Vector{Float64},
    xreg_mat::Matrix{Float64},
    n_xreg_cols::Int,
    n_conditioning::Int,
    xreg_range::UnitRange{Int},
    objective_workspace::ObjectiveWorkspace,
)
    parameter_buffer = objective_workspace.parameter_buffer
    _scatter_free_parameters!(parameter_buffer, full_coef, free_params, free_parameter_indices)
    transform_arima_parameters!(
        objective_workspace.phi_coefficients,
        objective_workspace.theta_coefficients,
        parameter_buffer,
        order,
        false,
        objective_workspace.nonseasonal_ar_transform_work,
        objective_workspace.seasonal_ar_transform_work,
    )

    x_work = if n_xreg_cols > 0
        _prepare_regression_adjusted_series!(
            objective_workspace,
            x,
            xreg_mat,
            parameter_buffer,
            xreg_range,
        )
    else
        x
    end

    ross = compute_css_residuals!(
        objective_workspace.css_differenced,
        objective_workspace.css_residuals,
        x_work,
        order,
        objective_workspace.phi_coefficients,
        objective_workspace.theta_coefficients,
        n_conditioning,
    )
    sigma2 = ross[:sigma2]
    (sigma2 < 0 || isnan(sigma2) || sigma2 == Inf) && return typemax(Float64)
    result = 0.5 * log(sigma2)
    return isnan(result) || result == Inf ? typemax(Float64) : result
end

"""
    fit!(model::SARIMA)

Fit a SARIMA model in-place using the method specified in `model.method`.

Supported methods: `:css`, `:ml`, `:css_ml` (default).

After fitting, `model.results` contains a `SARIMAResults` and `model.system`
contains the final state-space representation.

Returns the model itself (for chaining).
"""
function fit!(model::SARIMA{Fl}) where {Fl}
    order = model.order
    n_arma = n_arma_params(order)
    Delta = arima_differencing_delta(order.d, order.D, order.s)
    method = model.method
    transform_pars = model.transform_pars
    optim_method = model.optim_method
    optim_control = model.optim_control
    kappa = Float64(model.kappa)
    x = model.y
    n = length(x)
    y_save = copy(x)

    nd = order.d + order.D
    n_used = length(dropmissing(x)) - length(Delta)

    xreg_original = model.xreg_original

    xreg = model.xreg
    if model.include_mean && (nd == 0)
        if isnothing(xreg)
            xreg = NamedMatrix(zeros(n, 0), String[])
        end
        xreg = add_drift_term(xreg, ones(n), "intercept")
    end

    xreg_mat, n_xreg_cols, xreg_names = process_xreg(xreg, n)

    if method === :css_ml
        has_missing = xi -> (ismissing(xi) || isnan(xi))
        anyna = any(has_missing, x)
        if n_xreg_cols > 0
            anyna |= any(has_missing, xreg_mat)
        end
        if anyna
            method = :ml
        end
    end

    if method in (:css, :css_ml)
        n_conditioning = css_conditioning(order)
    else
        n_conditioning = 0
    end

    # Parameter mask: NaN = free, finite = fixed
    fixed = if isnothing(model.fixed)
        fill(NaN, n_arma + n_xreg_cols)
    else
        f = Float64.(model.fixed)
        if length(f) != n_arma + n_xreg_cols
            throw(ArgumentError("'fixed' has length $(length(f)), expected $(n_arma + n_xreg_cols)"))
        end
        f
    end
    mask = isnan.(fixed)
    free_parameter_indices = findall(mask)
    no_optim = !any(mask)

    if no_optim
        transform_pars = false
    end

    if transform_pars
        sar_idx = sar_indices(order)
        ar_idx = ar_indices(order)
        if any(.!mask[ar_idx]) || (!isempty(sar_idx) && any(.!mask[sar_idx]))
            @warn "Some AR parameters were fixed: Setting transform_pars = false"
            transform_pars = false
        end
    end

    # Xreg preprocessing: column-wise L2 normalization for numerical conditioning
    xreg_range = xreg_indices(order, n_xreg_cols)
    mask_xreg = n_xreg_cols > 0 ? mask[xreg_range] : Bool[]
    xreg_mat, xreg_scales = preprocess_xreg(xreg_mat, mask_xreg)
    objective_workspace = ObjectiveWorkspace(length(fixed), n, order)

    if n_xreg_cols > 0
        init0, parscale, n_used =
            compute_initial_params(x, xreg_mat, n_arma, n_xreg_cols, order, Delta)
    else
        init0 = zeros(n_arma)
        parscale = ones(n_arma)
    end

    if n_used <= 0
        throw(ArgumentError("insufficient non-missing observations for estimation"))
    end

    init = model.init
    if !isnothing(init)
        init = copy(init)
        if length(init) != length(init0)
            throw(ArgumentError("'init' is of the wrong length"))
        end
        ind = map(xi -> isnan(xi) || ismissing(xi), init)
        if any(ind)
            init[ind] .= init0[ind]
        end
        if method === :ml
            ar_idx = ar_indices(order)
            sar_idx = sar_indices(order)
            if order.p > 0 && !ar_check(init[ar_idx])
                throw(ArgumentError("AR polynomial has roots inside the unit circle"))
            end
            if order.P > 0 && !ar_check(init[sar_idx])
                throw(ArgumentError("seasonal AR polynomial has roots inside the unit circle"))
            end
        end
    else
        init = copy(init0)
    end

    coef = copy(Float64.(fixed))
    kalman_ws = Ref{Union{KalmanWorkspace,Nothing}}(nothing)

    # Likelihood evaluation with residuals (for final model evaluation)
    _ssl(y_in, mod) = compute_arima_likelihood(y_in, mod, 0, true)

    # CSS objective (standalone function, not closure over mutable state)
    css_fn = p -> _css_objective(
        p,
        fixed,
        free_parameter_indices,
        order,
        x,
        xreg_mat,
        n_xreg_cols,
        n_conditioning,
        xreg_range,
        objective_workspace,
    )

    if method === :css
        if no_optim
            res = (converged=true, minimizer=zeros(0), minimum=css_fn(zeros(0)))
        else
            opt = optimize(css_fn, init[mask], optim_method;
                param_scale = parscale[mask],
                step_sizes = get(optim_control, "ndeps", fill(1e-2, sum(mask))),
                max_iterations = get(optim_control, "maxit", 500))
            res = (converged=opt.converged, minimizer=opt.minimizer, minimum=opt.minimum)
        end

        if !res.converged
            @warn "CSS optimization convergence issue"
        end

        coef[mask] .= res.minimizer
        trarma = transform_arima_parameters(coef, order, false)
        mod = _as_typed_sarima_system(
            initialize_arima_state(trarma[1], trarma[2], Delta; kappa=kappa),
        )

        if n_xreg_cols > 0
            x = x - xreg_mat * coef[xreg_range]
        end
        _ssl(x, mod)
        val = compute_css_residuals(x, order, trarma[1], trarma[2], n_conditioning)
        sigma2 = val[:sigma2]

        if no_optim
            var = zeros(0)
        else
            hessian = numerical_hessian(css_fn, res.minimizer)
            var = inv(hessian * n_used)
        end

    else
        # --- CSS→ML or ML-only flow (Harvey 1989 §3.4.2) ---

        if method in (:css_ml, :ml)
            if method === :ml
                n_conditioning = css_conditioning(order)
            end

            if no_optim
                res = (converged=true, minimizer=zeros(sum(mask)), minimum=css_fn(zeros(0)))
            else
                opt = optimize(css_fn, init[mask], optim_method;
                    param_scale = parscale[mask],
                    step_sizes = get(optim_control, "ndeps", fill(1e-2, sum(mask))),
                    max_iterations = get(optim_control, "maxit", 500))
                res = (converged=opt.converged, minimizer=opt.minimizer, minimum=opt.minimum)
            end

            if res.converged
                init[mask] .= res.minimizer
            end

            # Stationarity validation of CSS solution (mathematical requirement)
            ar_idx = ar_indices(order)
            sar_idx = sar_indices(order)
            if order.p > 0 && !ar_check(init[ar_idx])
                throw(ArgumentError("CSS initialization produced non-stationary AR polynomial"))
            end
            if order.P > 0 && !ar_check(init[sar_idx])
                throw(ArgumentError("CSS initialization produced non-stationary seasonal AR polynomial"))
            end

            n_conditioning = 0
        end

        # Transform to unconstrained space (Jones 1980) for ML optimization
        if transform_pars
            init = inverse_arima_parameter_transform(init, order)
            ma_idx = ma_indices(order)
            sma_idx = sma_indices(order)
            if order.q > 0
                init[ma_idx] .= arima_reflect_ma_roots(init[ma_idx])
            end
            if order.Q > 0
                init[sma_idx] .= arima_reflect_ma_roots(init[sma_idx])
            end
        end

        trarma = transform_arima_parameters(init, order, transform_pars)
        mod = _as_typed_sarima_system(
            initialize_arima_state(trarma[1], trarma[2], Delta; kappa=kappa),
        )

        rd = length(mod.a)
        d_len = length(mod.Delta)
        kalman_ws[] = KalmanWorkspace(rd, n, d_len, false)

        # ML objective (standalone function)
        ml_fn = p -> _ml_objective(
            p,
            coef,
            free_parameter_indices,
            order,
            x,
            xreg_mat,
            n_xreg_cols,
            xreg_range,
            objective_workspace,
            mod,
            transform_pars,
            kappa,
            kalman_ws[],
        )

        if no_optim
            res = (converged=true, minimizer=zeros(0), minimum=ml_fn(zeros(0)))
        else
            opt = optimize(ml_fn, init[mask], optim_method;
                param_scale = parscale[mask],
                step_sizes = get(optim_control, "ndeps", nothing),
                max_iterations = get(optim_control, "maxit", nothing))
            res = (converged=opt.converged, minimizer=opt.minimizer, minimum=opt.minimum)
        end

        if !res.converged
            @warn "Optimizer may not have fully converged"
        end

        coef[mask] .= res.minimizer

        # Variance-covariance via Hessian + Delta method (standard statistics)
        if transform_pars
            ma_idx = ma_indices(order)
            sma_idx = sma_indices(order)
            if order.q > 0 && all(mask[ma_idx])
                coef[ma_idx] .= arima_reflect_ma_roots(coef[ma_idx])
            end
            if order.Q > 0 && all(mask[sma_idx])
                coef[sma_idx] .= arima_reflect_ma_roots(coef[sma_idx])
            end

            # Re-evaluate at final point to get accurate Hessian
            if any(coef[mask] .!= res.minimizer)
                ml_fn_eval = p -> _ml_objective(p, coef, free_parameter_indices, order, x, xreg_mat, n_xreg_cols,
                                                xreg_range, objective_workspace, mod, true, kappa, kalman_ws[])
                # Evaluate objective at the new parameter values
                opt = optimize(ml_fn_eval, coef[mask], optim_method;
                    param_scale = parscale[mask],
                    max_iterations = 0)
                res = (converged=opt.converged, minimizer=opt.minimizer, minimum=opt.minimum)
                hessian = numerical_hessian(ml_fn_eval, res.minimizer)
                coef[mask] .= res.minimizer
            else
                hessian = numerical_hessian(ml_fn, res.minimizer)
            end

            # Delta method: transform Hessian from unconstrained to constrained space
            A = compute_arima_transform_gradient(coef, order)
            A = A[mask, mask]
            var = A' * ((hessian * n_used) \ A)
            coef = undo_arima_parameter_transform(coef, order)
        else
            if no_optim
                var = zeros(0)
            else
                ml_fn_hess = p -> _ml_objective(p, coef, free_parameter_indices, order, x, xreg_mat, n_xreg_cols,
                                                xreg_range, objective_workspace, mod, true, kappa, kalman_ws[])
                hessian = numerical_hessian(ml_fn_hess, res.minimizer)
                var = inv(hessian * n_used)
            end
        end

        trarma = transform_arima_parameters(coef, order, false)
        mod = _as_typed_sarima_system(
            initialize_arima_state(trarma[1], trarma[2], Delta; kappa=kappa),
        )

        val = if n_xreg_cols > 0
            _ssl(x - xreg_mat * coef[xreg_range], mod)
        else
            _ssl(x, mod)
        end
        sigma2 = val.stats[1] / n_used
    end

    value = 2 * n_used * res.minimum + n_used + n_used * log(2 * π)
    aic = method !== :css ? value + 2 * sum(mask) + 2 : nothing
    loglik = -0.5 * value

    # Undo xreg column scaling
    if n_xreg_cols > 0 && all(mask_xreg)
        unscale_xreg_coefs!(coef, xreg_scales, xreg_range)
        # Transform variance-covariance accordingly
        if length(var) > 0
            A = Matrix{Float64}(I, n_arma + n_xreg_cols, n_arma + n_xreg_cols)
            for (i, idx) in enumerate(xreg_range)
                A[idx, idx] = 1.0 / xreg_scales[i]
            end
            A = A[mask, mask]
            var = A * var * transpose(A)
        end
    end

    arima_coef = format_coef_names(order, coef, xreg_names, n_xreg_cols)
    resid = val.residuals
    fitted_vals = y_save .- resid

    if size(var) == (0,)
        var = reshape(var, 0, 0)
    end

    o = order
    model.system = SARIMASystem{Fl}(
        Fl.(mod.phi), Fl.(mod.theta), Fl.(mod.Delta),
        Fl.(mod.Z), Fl.(mod.T), Fl.(mod.V),
        Fl(mod.h), Fl.(mod.a), Fl.(mod.P), Fl.(mod.Pn),
    )

    model.hyperparameters = HyperParameters{Fl}(
        Fl.(coef), Fl.(coef), mask, arima_coef.colnames, Fl.(fixed),
    )

    model.results = SARIMAResults{Fl}(
        arima_coef,
        sigma2,
        var,
        loglik,
        aic,
        nothing,
        nothing,
        nothing,
        resid,
        fitted_vals,
        res.converged,
        n_conditioning,
        n_used,
    )

    model.method = method
    model.n_cond = n_conditioning
    model.xreg = xreg_original
    model.transform_pars = transform_pars

    return model
end

"""
    arima(x, m; kwargs...) -> ArimaFit

Fit an ARIMA model. This is a wrapper that constructs a [`SARIMA`](@ref) model,
calls [`fit!`](@ref), and returns an [`ArimaFit`](@ref) for backward compatibility.
"""
function arima(
    x::AbstractArray,
    m::Int;
    order::PDQ = PDQ(0, 0, 0),
    seasonal::PDQ = PDQ(0, 0, 0),
    xreg::Union{Nothing, NamedMatrix} = nothing,
    include_mean::Bool = true,
    transform_pars::Bool = true,
    fixed::Union{Nothing, AbstractArray} = nothing,
    init::Union{Nothing, AbstractArray} = nothing,
    method::Symbol = :css_ml,
    n_cond::Union{Nothing, AbstractArray} = nothing,
    optim_method::Symbol = :bfgs,
    optim_control::Dict = Dict(),
    kappa::Real = 1e6,
)
    x_clean = [ismissing(xi) ? NaN : Float64(xi) for xi in x]
    model = SARIMA(
        x_clean, m;
        order=order, seasonal=seasonal, xreg=xreg,
        include_mean=include_mean, transform_pars=transform_pars,
        fixed=fixed, init=init, method=method,
        optim_method=optim_method,
        optim_control=optim_control, kappa=kappa,
    )

    fit!(model)

    o = model.order
    r = model.results
    has_xreg_coefs = !isnothing(r) && length(r.coef.colnames) > n_arma_params(o)

    fit_method = if has_xreg_coefs
        "Regression with ARIMA($(o.p),$(o.d),$(o.q))($(o.P),$(o.D),$(o.Q))[$(o.s)] errors"
    else
        "ARIMA($(o.p),$(o.d),$(o.q))($(o.P),$(o.D),$(o.Q))[$(o.s)]"
    end

    return ArimaFit(model, fit_method)
end
