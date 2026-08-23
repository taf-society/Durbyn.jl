# ─── TBATS likelihood functions ───────────────────────────────────────────────
#
# Objective functions for the optimizer: calc_likelihood_tbats (with Box-Cox)
# and calc_likelihood_tbats_notransform (without Box-Cox).

function calc_likelihood_tbats(
    param_vector::Vector{Float64},
    opt_env::Dict{Symbol,Any};
    use_beta::Bool,
    use_small_phi::Bool,
    seasonal_periods::Union{Vector{<:Real},Nothing},
    k_vector::Union{Vector{Int},Nothing},
    p::Int = 0,
    q::Int = 0,
    tau::Int = 0,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
)
    control = TBATSParameterControl(
        true,
        use_beta,
        use_small_phi,
        isnothing(k_vector) ? 0 : 2 * length(k_vector),
        p,
        q
    )

    paramz = unparameterise_tbats(param_vector, control)
    box_cox_parameter = paramz.lambda
    alpha = paramz.alpha
    beta_v = paramz.beta
    small_phi = paramz.small_phi
    gamma_one_v = paramz.gamma_one_v
    gamma_two_v = paramz.gamma_two_v
    ar_coefs = paramz.ar_coefs
    ma_coefs = paramz.ma_coefs

    box_cox!(opt_env[:box_cox_buffer_x], vec(opt_env[:x_nought_untransformed]), 1; lambda=box_cox_parameter)
    x_nought = reshape(opt_env[:box_cox_buffer_x], :, 1)

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

    opt_env[:w_transpose] = w.w_transpose
    opt_env[:g] = reshape(g_result.g, :, 1)
    opt_env[:gamma_bold_matrix] = g_result.gamma_bold_matrix
    opt_env[:F] = F

    box_cox!(opt_env[:box_cox_buffer_y], vec(opt_env[:y]), 1; lambda=box_cox_parameter)
    n = size(opt_env[:y], 2)
    transformed_y = opt_env[:box_cox_buffer_y]

    w_t = opt_env[:w_transpose]
    g = opt_env[:g]
    y_hat = opt_env[:y_hat]
    e = opt_env[:e]
    x = opt_env[:x]
    Fx_buf = opt_env[:Fx_buffer]

    @inbounds for t = 1:n
        if t == 1
            y_hat[1, t] = dot(w_t, view(x_nought, :, 1))
            e[1, t] = transformed_y[t] - y_hat[1, t]
            mul!(Fx_buf, F, view(x_nought, :, 1))
            @. x[:, t] = Fx_buf + g * e[1, t]
        else
            y_hat[1, t] = dot(w_t, view(x, :, t-1))
            e[1, t] = transformed_y[t] - y_hat[1, t]
            mul!(Fx_buf, F, view(x, :, t-1))
            @. x[:, t] = Fx_buf + g * e[1, t]
        end
    end

    log_likelihood = n * log(sum(abs2, e)) - 2 * (box_cox_parameter - 1) * sum(log(yi) for yi in opt_env[:y])

    D = opt_env[:F] - opt_env[:g] * opt_env[:w_transpose]
    opt_env[:D] = D

    is_admissible = check_admissibility_tbats(
        D;
        box_cox = box_cox_parameter,
        small_phi = small_phi,
        ar_coefs = ar_coefs,
        ma_coefs = ma_coefs,
        tau = tau,
        bc_lower = bc_lower,
        bc_upper = bc_upper,
    )

    if is_admissible
        return log_likelihood
    else
        return 1e20
    end
end

function calc_likelihood_tbats_notransform(
    param_vector::Vector{Float64},
    opt_env::Dict{Symbol,Any},
    x_nought::AbstractMatrix;
    use_beta::Bool,
    use_small_phi::Bool,
    seasonal_periods::Union{Vector{<:Real},Nothing},
    k_vector::Union{Vector{Int},Nothing},
    p::Int = 0,
    q::Int = 0,
    tau::Int = 0,
)
    control = TBATSParameterControl(
        false,
        use_beta,
        use_small_phi,
        isnothing(k_vector) ? 0 : 2 * length(k_vector),
        p,
        q
    )

    paramz = unparameterise_tbats(param_vector, control)
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

    opt_env[:w_transpose] = w.w_transpose
    opt_env[:g] = reshape(g_result.g, :, 1)
    opt_env[:gamma_bold_matrix] = g_result.gamma_bold_matrix
    opt_env[:F] = F

    n = size(opt_env[:y], 2)

    w_t = opt_env[:w_transpose]
    g = opt_env[:g]
    y_hat = opt_env[:y_hat]
    e = opt_env[:e]
    x = opt_env[:x]
    y_data = opt_env[:y]
    Fx_buf = opt_env[:Fx_buffer]

    @inbounds for t = 1:n
        if t == 1
            y_hat[1, t] = dot(w_t, view(x_nought, :, 1))
            e[1, t] = y_data[1, t] - y_hat[1, t]
            mul!(Fx_buf, F, view(x_nought, :, 1))
            @. x[:, t] = Fx_buf + g * e[1, t]
        else
            y_hat[1, t] = dot(w_t, view(x, :, t-1))
            e[1, t] = y_data[1, t] - y_hat[1, t]
            mul!(Fx_buf, F, view(x, :, t-1))
            @. x[:, t] = Fx_buf + g * e[1, t]
        end
    end

    log_likelihood = n * log(sum(e .* e))

    D = opt_env[:F] - opt_env[:g] * opt_env[:w_transpose]
    opt_env[:D] = D

    is_admissible = check_admissibility_tbats(
        D;
        box_cox = nothing,
        small_phi = small_phi,
        ar_coefs = ar_coefs,
        ma_coefs = ma_coefs,
        tau = tau,
    )

    if is_admissible
        return log_likelihood
    else
        return 1e20
    end
end
