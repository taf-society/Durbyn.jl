# ─── BATS model fitting and selection ─────────────────────────────────────────
#
# Core fitting routines: fit_specific_bats (single configuration),
# filter_specifics (compare seasonal vs non-seasonal, add ARMA errors),
# fit_previous_bats_model (refit with frozen parameters),
# create_constant_model (trivial flat-series model).

function fit_specific_bats(
    y::AbstractVector{<:Real};
    use_box_cox::Bool,
    use_beta::Bool,
    use_damping::Bool,
    seasonal_periods::Union{Vector{Int},Nothing} = nothing,
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
        seasonal_periods = sort!(Int.(seasonal_periods))
    end


    if starting_params === nothing

        p = ar_coefs === nothing ? 0 : length(ar_coefs)
        q = ma_coefs === nothing ? 0 : length(ma_coefs)

        sp_sum = seasonal_periods === nothing ? 0 : sum(seasonal_periods)


        alpha = sp_sum > 16 ? 1e-6 : 0.09


        if use_beta
            beta_v = sp_sum > 16 ? 5e-7 : 0.05
            b = 0.00
            small_phi = use_damping ? 0.999 : 1.0
        else
            beta_v = nothing
            b = nothing
            small_phi = nothing

            use_damping = false
        end


        if seasonal_periods !== nothing
            gamma_v = fill(0.001, length(seasonal_periods))
            s_vector = zeros(sum(seasonal_periods))
        else
            gamma_v = nothing
            s_vector = nothing
        end


        if use_box_cox
            if init_box_cox !== nothing
                lambda = init_box_cox
            else

                bc_period = (seasonal_periods === nothing || isempty(seasonal_periods)) ? 1 : first(seasonal_periods)
                lambda = box_cox_lambda(y, bc_period; lower = bc_lower, upper = bc_upper)
            end
        else
            lambda = nothing
        end

    else

        paramz = unparameterise(starting_params.vect, starting_params.control)

        lambda = paramz.lambda
        alpha = paramz.alpha
        beta_v = paramz.beta
        b = isnothing(paramz.beta) ? nothing : 0.0
        small_phi = paramz.small_phi
        gamma_v = paramz.gamma_v

        if seasonal_periods !== nothing
            s_vector = zeros(sum(seasonal_periods))
        else
            s_vector = nothing
        end


        p = ar_coefs === nothing ? 0 : length(ar_coefs)
        q = ma_coefs === nothing ? 0 : length(ma_coefs)
    end


    if x_nought === nothing

        d_vector = ar_coefs === nothing ? nothing : zeros(length(ar_coefs))

        epsilon_vector = ma_coefs === nothing ? nothing : zeros(length(ma_coefs))


        x_nought_result = make_xmatrix(0.0, b, s_vector, d_vector, epsilon_vector)

        x_nought = x_nought_result.x
    else
        x_nought = reshape(collect(float.(x_nought)), :, 1)
    end



    param_result =
        parameterise(alpha, beta_v, small_phi, gamma_v, lambda, ar_coefs, ma_coefs)
    param_vector = param_result.vect
    control = param_result.control
    par_scale = make_parscale_bats(control)


    w = make_wmatrix(small_phi, seasonal_periods, ar_coefs, ma_coefs)
    g = make_gmatrix(alpha, beta_v, gamma_v, seasonal_periods, p, q)
    F = make_fmatrix(
        alpha,
        beta_v,
        small_phi,
        seasonal_periods,
        g.gamma_bold_matrix,
        ar_coefs,
        ma_coefs,
    )
    D = F .- reshape(g.g, :, 1) * w.w_transpose


    if use_box_cox
        y_transformed, lambda = box_cox(y, 1; lambda=lambda)
        fitted = calc_model(y_transformed, vec(x_nought), F, g.g, w)
    else
        fitted = calc_model(y, vec(x_nought), F, g.g, w)
    end
    y_tilda = fitted.e


    n = length(y)
    k = size(w.w_transpose, 2)
    w_tilda_transpose = zeros(n, k)
    w_tilda_transpose[1, :] .= w.w_transpose[1, :]

    for i = 2:n

        w_tilda_transpose[i, :] = vec(transpose(w_tilda_transpose[i-1, :]) * D)
    end


    if seasonal_periods !== nothing

        list_cut_w = cut_w(use_beta, w_tilda_transpose, seasonal_periods, p, q)
        w_tilda_cut = list_cut_w.matrix
        mask_vector = list_cut_w.mask_vector



        coefs = w_tilda_cut \ y_tilda
        x_nought =
            calc_seasonal_seeds(use_beta, coefs, seasonal_periods, mask_vector, p, q)

    else

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
    end


    opt_env = Dict{Symbol,Any}()
    opt_env[:F] = F
    opt_env[:w_transpose] = w.w_transpose
    opt_env[:g] = reshape(g.g, :, 1)
    opt_env[:gamma_bold_matrix] = g.gamma_bold_matrix
    opt_env[:y] = reshape(y, 1, :)
    opt_env[:y_hat] = zeros(1, n)
    opt_env[:e] = zeros(1, n)
    opt_env[:x] = zeros(size(x_nought, 1), n)

    opt_env[:y_vec_buffer] = Vector{Float64}(undef, n)
    opt_env[:x_nought_buffer] = Vector{Float64}(undef, size(x_nought, 1))
    opt_env[:y_transformed_mat] = Matrix{Float64}(undef, 1, n)

    state_dim = size(x_nought, 1)
    opt_env[:Fx_buffer] = zeros(state_dim)

    tau = seasonal_periods === nothing ? 0 : sum(seasonal_periods)


    if use_box_cox

        x_nought_untransformed = inv_box_cox(x_nought; lambda=lambda)
        opt_env[:x_nought_untransformed] = x_nought_untransformed


        original_objective =
            pvec -> calc_likelihood(
                pvec,
                opt_env;
                use_beta = use_beta,
                use_small_phi = use_damping,
                seasonal_periods = seasonal_periods,
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


        paramz = unparameterise(opt_par, control)

        lambda = paramz.lambda
        alpha = paramz.alpha
        beta_v = paramz.beta
        small_phi = paramz.small_phi
        gamma_v = paramz.gamma_v
        ar_coefs = paramz.ar_coefs
        ma_coefs = paramz.ma_coefs


        x_nought_vec, lambda = box_cox(vec(opt_env[:x_nought_untransformed]), 1; lambda=lambda)
        x_nought = reshape(x_nought_vec, :, 1)


        w = make_wmatrix(small_phi, seasonal_periods, ar_coefs, ma_coefs)
        g = make_gmatrix(alpha, beta_v, gamma_v, seasonal_periods, p, q)
        F = make_fmatrix(
            alpha,
            beta_v,
            small_phi,
            seasonal_periods,
            g.gamma_bold_matrix,
            ar_coefs,
            ma_coefs,
        )


        y_transformed, lambda = box_cox(y, 1; lambda=lambda)
        fitted_values_and_errors = calc_model(y_transformed, vec(x_nought), F, g.g, w)
        e = fitted_values_and_errors.e
        variance = sum(abs2, e) / length(y)

        fitted_values = inv_box_cox(fitted_values_and_errors.y_hat; lambda=lambda, biasadj=biasadj, fvar=variance)


    else


        original_objective =
            pvec -> calc_likelihood2(
                pvec,
                opt_env,
                x_nought;
                use_beta = use_beta,
                use_small_phi = use_damping,
                seasonal_periods = seasonal_periods,
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

        paramz = unparameterise(opt_par, control)

        lambda = paramz.lambda
        alpha = paramz.alpha
        beta_v = paramz.beta
        small_phi = paramz.small_phi
        gamma_v = paramz.gamma_v
        ar_coefs = paramz.ar_coefs
        ma_coefs = paramz.ma_coefs


        w = make_wmatrix(small_phi, seasonal_periods, ar_coefs, ma_coefs)
        g = make_gmatrix(alpha, beta_v, gamma_v, seasonal_periods, p, q)
        F = make_fmatrix(
            alpha,
            beta_v,
            small_phi,
            seasonal_periods,
            g.gamma_bold_matrix,
            ar_coefs,
            ma_coefs,
        )

        fitted_values_and_errors = calc_model(y, vec(x_nought), F, g.g, w)
        e = fitted_values_and_errors.e
        fitted_values = fitted_values_and_errors.y_hat
        variance = sum(abs2, e) / length(y)
    end


    likelihood = opt_result.minimum

    aic = likelihood + 2 * (length(param_vector) + size(x_nought, 1))


    model = (
        lambda = lambda,
        alpha = alpha,
        beta = beta_v,
        damping_parameter = small_phi,
        gamma_values = gamma_v,
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
        y = y,
        biasadj = biasadj,
    )

    return model
end

function filter_specifics(
    y;
    box_cox::Bool,
    trend::Bool,
    damping::Bool,
    seasonal_periods,
    use_arma_errors::Bool,
    force_seasonality::Bool = false,
    init_box_cox = nothing,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
    biasadj::Bool = false,
    kwargs...,
)


    if !trend && damping

        return (aic = nothing,)
    end


    first_model = fit_specific_bats(
        y;
        use_box_cox = box_cox,
        use_beta = trend,
        use_damping = damping,
        seasonal_periods = seasonal_periods,
        init_box_cox = init_box_cox,
        bc_lower = bc_lower,
        bc_upper = bc_upper,
        biasadj = biasadj,
        kwargs...,
    )


    # Store the chosen seasonal configuration for ARMA model fitting
    best_seasonal_periods = seasonal_periods

    if seasonal_periods !== nothing && !force_seasonality
        non_seasonal_model = fit_specific_bats(
            y;
            use_box_cox = box_cox,
            use_beta = trend,
            use_damping = damping,
            seasonal_periods = nothing,
            init_box_cox = init_box_cox,
            bc_lower = bc_lower,
            bc_upper = bc_upper,
            biasadj = biasadj,
            kwargs...,
        )

        if _aic_val(first_model) > _aic_val(non_seasonal_model)
            best_seasonal_periods = nothing
            first_model = non_seasonal_model
        end
    end


    if use_arma_errors



        arma = auto_arima(collect(first_model.errors), 1; d = 0, kwargs...)

        p = arma.arma[1]
        q = arma.arma[2]

        if p != 0 || q != 0
            ar_coefs = p != 0 ? zeros(p) : nothing
            ma_coefs = q != 0 ? zeros(q) : nothing



            starting_params = first_model.parameters


            second_model = fit_specific_bats(
                y;
                use_box_cox = box_cox,
                use_beta = trend,
                use_damping = damping,
                seasonal_periods = best_seasonal_periods,
                starting_params = starting_params,
                ar_coefs = ar_coefs,
                ma_coefs = ma_coefs,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
                kwargs...,
            )

            if _aic_val(second_model) < _aic_val(first_model)
                return second_model
            else
                return first_model
            end
        else
            return first_model
        end
    else
        return first_model
    end
end

function fit_previous_bats_model(y::Vector{Float64}, old_model::BATSModel)
    # Handle constant model edge case
    if isempty(old_model.parameters)
        return create_constant_model(y)
    end

    # Extract frozen parameters
    paramz = unparameterise(old_model.parameters[:vect], old_model.parameters[:control])
    seasonal_periods = old_model.seasonal_periods
    p = isnothing(paramz.ar_coefs) ? 0 : length(paramz.ar_coefs)
    q = isnothing(paramz.ma_coefs) ? 0 : length(paramz.ma_coefs)

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
    w = make_wmatrix(paramz.small_phi, seasonal_periods, paramz.ar_coefs, paramz.ma_coefs)
    g_result = make_gmatrix(paramz.alpha, paramz.beta, paramz.gamma_v, seasonal_periods, p, q)
    F = make_fmatrix(
        paramz.alpha,
        paramz.beta,
        paramz.small_phi,
        seasonal_periods,
        g_result.gamma_bold_matrix,
        paramz.ar_coefs,
        paramz.ma_coefs,
    )

    # Seed states are already stored in Box-Cox transformed space
    x_nought = collect(Float64, vec(old_model.seed_states))

    # Run calc_model on new data
    result = calc_model(y_transformed, x_nought, F, g_result.g, w)

    # Compute variance and back-transform
    variance = sum(abs2, result.e) / length(y)
    fitted_values = !isnothing(paramz.lambda) ?
        inv_box_cox(result.y_hat; lambda=paramz.lambda, biasadj=old_model.biasadj, fvar=variance) : result.y_hat

    # Compute likelihood and AIC
    n = length(y)
    if !isnothing(paramz.lambda)
        likelihood = n * log(sum(abs2, result.e)) - 2 * (paramz.lambda - 1) * sum(log.(y))
    else
        likelihood = n * log(sum(abs2, result.e))
    end
    n_params = length(old_model.parameters[:vect]) + size(old_model.seed_states, 1)
    aic = likelihood + 2 * n_params

    method_label = bats_descriptor(paramz.lambda, paramz.ar_coefs, paramz.ma_coefs,
                                    paramz.small_phi, seasonal_periods)

    return BATSModel(
        paramz.lambda, paramz.alpha, paramz.beta, paramz.small_phi,
        paramz.gamma_v, paramz.ar_coefs, paramz.ma_coefs, seasonal_periods,
        collect(fitted_values), collect(result.e), result.x,
        old_model.seed_states, variance, aic, likelihood,
        0,  # optim_return_code (no optimization)
        y,  # caller swaps in orig_y
        Dict{Symbol,Any}(:vect => old_model.parameters[:vect],
                          :control => old_model.parameters[:control]),
        method_label,
        old_model.biasadj,
    )
end

function create_constant_model(y::Vector{Float64})
    n = length(y)
    y_mean = mean(y)

    return BATSModel(
        nothing,
        0.9999,
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
        bats_descriptor(nothing, nothing, nothing, nothing, nothing),
        false,
    )
end
