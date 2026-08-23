# ─── TBATS model fitting and selection ────────────────────────────────────────
#
# Core fitting routines: fit_specific_tbats (single configuration),
# filter_tbats_specifics (add ARMA errors), fit_previous_tbats_model
# (refit with frozen parameters), create_constant_tbats_model.

function fit_specific_tbats(
    y::AbstractVector{<:Real};
    use_box_cox::Bool,
    use_beta::Bool,
    use_damping::Bool,
    seasonal_periods::Union{Vector{<:Real},Nothing} = nothing,
    k_vector::Union{Vector{Int},Nothing} = nothing,
    starting_params = nothing,
    x_nought = nothing,
    ar_coefs::Union{AbstractVector{<:Real},Nothing} = nothing,
    ma_coefs::Union{AbstractVector{<:Real},Nothing} = nothing,
    init_box_cox = nothing,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
    biasadj::Bool = false,
    kwargs...,
)
    y = collect(float.(y))

    if seasonal_periods !== nothing
        perm = sortperm(seasonal_periods)
        seasonal_periods = seasonal_periods[perm]
        k_vector = k_vector[perm]
    end

    if starting_params === nothing
        p = ar_coefs === nothing ? 0 : length(ar_coefs)
        q = ma_coefs === nothing ? 0 : length(ma_coefs)

        alpha = 0.09

        if use_beta
            beta_v = 0.05
            b = 0.00
            small_phi = use_damping ? 0.999 : 1.0
        else
            beta_v = nothing
            b = nothing
            small_phi = nothing
            use_damping = false
        end

        if seasonal_periods !== nothing && k_vector !== nothing
            gamma_one_v = zeros(length(k_vector))
            gamma_two_v = zeros(length(k_vector))
            s_vector = zeros(2 * sum(k_vector))
        else
            gamma_one_v = nothing
            gamma_two_v = nothing
            s_vector = nothing
        end

        if use_box_cox
            if init_box_cox !== nothing
                lambda = init_box_cox
            else
                bc_period = (seasonal_periods === nothing || isempty(seasonal_periods)) ? 1 : round(Int, first(seasonal_periods))
                lambda = box_cox_lambda(y, bc_period; lower = bc_lower, upper = bc_upper)
            end
        else
            lambda = nothing
        end
    else
        paramz = unparameterise_tbats(starting_params.vect, starting_params.control)
        lambda = paramz.lambda
        alpha = paramz.alpha
        beta_v = paramz.beta
        b = isnothing(paramz.beta) ? nothing : 0.0
        small_phi = paramz.small_phi
        gamma_one_v = paramz.gamma_one_v
        gamma_two_v = paramz.gamma_two_v

        if seasonal_periods !== nothing && k_vector !== nothing
            s_vector = zeros(2 * sum(k_vector))
        else
            s_vector = nothing
        end

        p = ar_coefs === nothing ? 0 : length(ar_coefs)
        q = ma_coefs === nothing ? 0 : length(ma_coefs)
    end

    if x_nought === nothing
        d_vector = ar_coefs === nothing ? nothing : zeros(length(ar_coefs))
        epsilon_vector = ma_coefs === nothing ? nothing : zeros(length(ma_coefs))
        x_nought_result = make_xmatrix_tbats(0.0, b, s_vector, d_vector, epsilon_vector)
        x_nought = x_nought_result.x
    else
        x_nought = reshape(collect(float.(x_nought)), :, 1)
    end

    param_result = parameterise_tbats(alpha, beta_v, small_phi, gamma_one_v, gamma_two_v, lambda, ar_coefs, ma_coefs)
    param_vector = param_result.vect
    control = param_result.control
    par_scale = make_parscale_tbats(control)

    tau = (seasonal_periods === nothing || k_vector === nothing) ? 0 : 2 * sum(k_vector)

    w = make_tbats_wmatrix(small_phi, k_vector, ar_coefs, ma_coefs, tau)
    g_result = make_tbats_gmatrix(alpha, beta_v, gamma_one_v, gamma_two_v, k_vector, p, q)
    F = make_tbats_fmatrix(
        alpha,
        beta_v,
        small_phi,
        seasonal_periods,
        k_vector,
        g_result.gamma_bold_matrix,
        ar_coefs,
        ma_coefs,
    )
    D = F .- g_result.g * w.w_transpose

    if use_box_cox
        y_transformed, lambda = box_cox(y, 1; lambda=lambda)
        fitted = calc_model_tbats(y_transformed, vec(x_nought), F, g_result.g, w)
    else
        fitted = calc_model_tbats(y, vec(x_nought), F, g_result.g, w)
    end
    y_tilda = fitted.e

    n = length(y)
    k_dim = size(w.w_transpose, 2)
    w_tilda_transpose = zeros(n, k_dim)
    w_tilda_transpose[1, :] .= w.w_transpose[1, :]

    for i = 2:n
        w_tilda_transpose[i, :] = vec(transpose(w_tilda_transpose[i-1, :]) * D)
    end

    if (p != 0) || (q != 0)
        end_cut = size(w_tilda_transpose, 2)
        start_cut = end_cut - (p + q) + 1
        keep_cols = 1:(start_cut-1)
        w_tilda_cut = w_tilda_transpose[:, keep_cols]
    else
        w_tilda_cut = w_tilda_transpose
    end

    coefs = w_tilda_cut \ y_tilda
    x_core = reshape(collect(float.(coefs)), :, 1)

    if (p != 0) || (q != 0)
        arma_seed_states = zeros(p + q, 1)
        x_nought = vcat(x_core, arma_seed_states)
    else
        x_nought = x_core
    end

    opt_env = Dict{Symbol,Any}()
    opt_env[:F] = F
    opt_env[:w_transpose] = w.w_transpose
    opt_env[:g] = g_result.g
    opt_env[:gamma_bold_matrix] = g_result.gamma_bold_matrix
    opt_env[:k_vector] = k_vector
    opt_env[:y] = reshape(y, 1, :)
    opt_env[:y_hat] = zeros(1, n)
    opt_env[:e] = zeros(1, n)
    opt_env[:x] = zeros(size(x_nought, 1), n)

    opt_env[:box_cox_buffer_x] = zeros(size(x_nought, 1))
    opt_env[:box_cox_buffer_y] = zeros(n)

    state_dim = size(x_nought, 1)
    opt_env[:Fx_buffer] = zeros(state_dim)
    opt_env[:g_scaled] = zeros(state_dim)

    if use_box_cox
        x_nought_untransformed = inv_box_cox(x_nought; lambda=lambda)
        opt_env[:x_nought_untransformed] = x_nought_untransformed

        original_objective = pvec -> calc_likelihood_tbats(
            pvec,
            opt_env;
            use_beta = use_beta,
            use_small_phi = use_damping,
            seasonal_periods = seasonal_periods,
            k_vector = k_vector,
            p = p,
            q = q,
            tau = tau,
            bc_lower = bc_lower,
            bc_upper = bc_upper,
        )

        scaled_param0 = param_vector ./ par_scale
        objective_scaled = θs -> original_objective(θs .* par_scale)

        maxit = 100 * length(param_vector)^2
        opt_result = optimize(
            objective_scaled,
            scaled_param0,
            :nelder_mead;
            max_iterations = maxit,
        )

        opt_par_scaled = opt_result.minimizer
        opt_par = opt_par_scaled .* par_scale

        paramz = unparameterise_tbats(opt_par, control)
        lambda = paramz.lambda
        alpha = paramz.alpha
        beta_v = paramz.beta
        small_phi = paramz.small_phi
        gamma_one_v = paramz.gamma_one_v
        gamma_two_v = paramz.gamma_two_v
        ar_coefs = paramz.ar_coefs
        ma_coefs = paramz.ma_coefs

        x_nought_vec, lambda = box_cox(vec(opt_env[:x_nought_untransformed]), 1; lambda=lambda)
        x_nought = reshape(x_nought_vec, :, 1)

        w = make_tbats_wmatrix(small_phi, k_vector, ar_coefs, ma_coefs, tau)
        g_result = make_tbats_gmatrix(alpha, beta_v, gamma_one_v, gamma_two_v, k_vector, p, q)
        F = make_tbats_fmatrix(
            alpha,
            beta_v,
            small_phi,
            seasonal_periods,
            k_vector,
            g_result.gamma_bold_matrix,
            ar_coefs,
            ma_coefs,
        )

        y_transformed, lambda = box_cox(y, 1; lambda=lambda)
        fitted_values_and_errors = calc_model_tbats(y_transformed, vec(x_nought), F, g_result.g, w)
        e = fitted_values_and_errors.e
        variance = sum(e .^ 2) / length(y)

        fitted_values = inv_box_cox(fitted_values_and_errors.y_hat; lambda=lambda, biasadj=biasadj, fvar=variance)
    else
        original_objective = pvec -> calc_likelihood_tbats_notransform(
            pvec,
            opt_env,
            x_nought;
            use_beta = use_beta,
            use_small_phi = use_damping,
            seasonal_periods = seasonal_periods,
            k_vector = k_vector,
            p = p,
            q = q,
            tau = tau,
        )

        scaled_param0 = param_vector ./ par_scale
        objective_scaled = θs -> original_objective(θs .* par_scale)

        if length(param_vector) > 1
            maxit = 100 * length(param_vector)^2
            opt_result = optimize(
                objective_scaled,
                scaled_param0,
                :nelder_mead;
                max_iterations = maxit,
            )
        else
            opt_result = optimize(
                objective_scaled,
                scaled_param0,
                :bfgs,
            )
        end

        opt_par_scaled = opt_result.minimizer
        opt_par = opt_par_scaled .* par_scale

        paramz = unparameterise_tbats(opt_par, control)
        lambda = paramz.lambda
        alpha = paramz.alpha
        beta_v = paramz.beta
        small_phi = paramz.small_phi
        gamma_one_v = paramz.gamma_one_v
        gamma_two_v = paramz.gamma_two_v
        ar_coefs = paramz.ar_coefs
        ma_coefs = paramz.ma_coefs

        w = make_tbats_wmatrix(small_phi, k_vector, ar_coefs, ma_coefs, tau)
        g_result = make_tbats_gmatrix(alpha, beta_v, gamma_one_v, gamma_two_v, k_vector, p, q)
        F = make_tbats_fmatrix(
            alpha,
            beta_v,
            small_phi,
            seasonal_periods,
            k_vector,
            g_result.gamma_bold_matrix,
            ar_coefs,
            ma_coefs,
        )

        fitted_values_and_errors = calc_model_tbats(y, vec(x_nought), F, g_result.g, w)
        e = fitted_values_and_errors.e
        fitted_values = fitted_values_and_errors.y_hat
        variance = sum(e .^ 2) / length(y)
    end

    likelihood = opt_result.minimum
    aic = likelihood + 2 * (length(param_vector) + size(x_nought, 1))

    model = (
        lambda = lambda,
        alpha = alpha,
        beta = beta_v,
        damping_parameter = small_phi,
        gamma_one_values = gamma_one_v,
        gamma_two_values = gamma_two_v,
        ar_coefficients = ar_coefs,
        ma_coefficients = ma_coefs,
        likelihood = likelihood,
        optim_return_code = opt_result.converged ? 0 : 1,
        variance = variance,
        aic = aic,
        parameters = (vect = opt_par, control = control),
        seed_states = x_nought,
        fitted_values = collect(fitted_values),
        errors = collect(e),
        x = fitted_values_and_errors.x,
        seasonal_periods = seasonal_periods,
        k_vector = k_vector,
        y = y,
        biasadj = biasadj,
    )

    return model
end

function filter_tbats_specifics(
    y::AbstractVector{<:Real},
    box_cox::Bool,
    trend::Bool,
    damping::Bool,
    seasonal_periods::Vector{<:Real},
    k_vector::Vector{Int},
    use_arma_errors::Bool;
    aux_model::Union{TBATSModel,NamedTuple,Nothing} = nothing,
    init_box_cox::Union{Nothing,Real} = nothing,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
    biasadj::Bool = false,
    arima_kwargs...,
)

    first_model = if aux_model === nothing
        try
            fit_specific_tbats(
                Float64.(y);
                use_box_cox = box_cox,
                use_beta = trend,
                use_damping = damping,
                seasonal_periods = seasonal_periods,
                k_vector = k_vector,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
            )
        catch e
            @warn "fit_specific_tbats in filter_tbats_specifics failed: $e"
            nothing
        end
    else
        aux_model
    end

    if first_model === nothing
        return nothing
    end

    if !use_arma_errors
        return first_model
    end

    arma = try
        auto_arima(collect(Float64, first_model.errors), 1; d = 0, arima_kwargs...)
    catch e
        @warn "auto_arima in filter_tbats_specifics failed: $e"
        nothing
    end

    if arma === nothing
        return first_model
    end

    p = arma.arma[1]
    q = arma.arma[2]

    if p == 0 && q == 0
        return first_model
    end

    ar_coefs = p > 0 ? zeros(Float64, p) : nothing
    ma_coefs = q > 0 ? zeros(Float64, q) : nothing

    starting_params = first_model.parameters

    second_model = try
        fit_specific_tbats(
            Float64.(y);
            use_box_cox = box_cox,
            use_beta = trend,
            use_damping = damping,
            seasonal_periods = seasonal_periods,
            k_vector = k_vector,
            starting_params = starting_params,
            ar_coefs = ar_coefs,
            ma_coefs = ma_coefs,
            init_box_cox = init_box_cox,
            bc_lower = bc_lower,
            bc_upper = bc_upper,
            biasadj = biasadj,
        )
    catch e
        @warn "fit_specific_tbats with ARMA in filter_tbats_specifics failed: $e"
        nothing
    end

    aic_first = _aic_val(first_model)
    aic_second = _aic_val(second_model)

    if aic_second < aic_first
        return second_model
    else
        return first_model
    end
end

function fit_previous_tbats_model(y::Vector{Float64}; model::TBATSModel)
    # Handle constant model edge case
    if isempty(model.parameters)
        return create_constant_tbats_model(y)
    end

    # Extract frozen parameters
    paramz = unparameterise_tbats(model.parameters[:vect], model.parameters[:control])
    seasonal_periods = model.seasonal_periods
    k_vector = model.k_vector
    p = isnothing(paramz.ar_coefs) ? 0 : length(paramz.ar_coefs)
    q = isnothing(paramz.ma_coefs) ? 0 : length(paramz.ma_coefs)
    tau = isnothing(k_vector) ? 0 : 2 * sum(k_vector)

    # Apply Box-Cox if old model used it
    if !isnothing(paramz.lambda)
        if any(yi -> yi <= 0, y)
            @warn "New data has non-positive values but old model used Box-Cox (lambda=$(paramz.lambda)). Results may contain NaN."
        end
        y_transformed, _ = box_cox(y, 1; lambda=paramz.lambda)
    else
        y_transformed = y
    end

    # Rebuild matrices from frozen parameters
    w = make_tbats_wmatrix(paramz.small_phi, k_vector, paramz.ar_coefs, paramz.ma_coefs, tau)
    g_result = make_tbats_gmatrix(paramz.alpha, paramz.beta, paramz.gamma_one_v, paramz.gamma_two_v, k_vector, p, q)
    F = make_tbats_fmatrix(
        paramz.alpha,
        paramz.beta,
        paramz.small_phi,
        seasonal_periods,
        k_vector,
        g_result.gamma_bold_matrix,
        paramz.ar_coefs,
        paramz.ma_coefs,
    )

    # Seed states are already stored in Box-Cox transformed space
    x_nought = collect(Float64, vec(model.seed_states))

    # Run calc_model_tbats on new data
    result = calc_model_tbats(y_transformed, x_nought, F, g_result.g, w)

    # Compute variance and back-transform
    variance = sum(abs2, result.e) / length(y)
    fitted_values = !isnothing(paramz.lambda) ?
        inv_box_cox(result.y_hat; lambda=paramz.lambda, biasadj=model.biasadj, fvar=variance) : result.y_hat

    # Compute likelihood and AIC
    n = length(y)
    if !isnothing(paramz.lambda)
        likelihood = n * log(sum(abs2, result.e)) - 2 * (paramz.lambda - 1) * sum(log.(y))
    else
        likelihood = n * log(sum(abs2, result.e))
    end
    n_params = length(model.parameters[:vect]) + size(model.seed_states, 1)
    aic = likelihood + 2 * n_params

    method_label = tbats_descriptor(paramz.lambda, paramz.ar_coefs, paramz.ma_coefs,
                                     paramz.small_phi, seasonal_periods, k_vector)

    return TBATSModel(
        paramz.lambda, paramz.alpha, paramz.beta, paramz.small_phi,
        paramz.gamma_one_v, paramz.gamma_two_v,
        paramz.ar_coefs, paramz.ma_coefs, seasonal_periods, k_vector,
        collect(fitted_values), collect(result.e), result.x,
        model.seed_states, variance, aic, likelihood,
        0,  # optim_return_code (no optimization)
        y,  # caller swaps in orig_y
        Dict{Symbol,Any}(:vect => model.parameters[:vect],
                          :control => model.parameters[:control]),
        method_label,
        model.biasadj,
    )
end

function create_constant_tbats_model(y::Vector{Float64})
    n = length(y)
    y_mean = mean(y)

    return TBATSModel(
        nothing,
        0.9999,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        fill(y_mean, n),
        zeros(n),
        fill(y_mean, 1, n),
        [y_mean],
        0.0,
        nothing,
        -Inf,
        0,
        y,
        Dict{Symbol,Any}(),
        tbats_descriptor(nothing, nothing, nothing, nothing, nothing, nothing),
        false,
    )
end
