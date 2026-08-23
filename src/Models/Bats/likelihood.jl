# ─── BATS likelihood functions ────────────────────────────────────────────────
#
# Objective functions for the optimizer: calc_likelihood (with Box-Cox) and
# calc_likelihood2 (without Box-Cox). Both unpack parameters, rebuild matrices,
# run the state space recursion and return the penalized log-likelihood.

@inline function calc_likelihood(
    param_vector::Vector{Float64},
    opt_env::Dict{Symbol,Any};
    use_beta::Bool,
    use_small_phi::Bool,
    seasonal_periods::Union{Vector{Int},Nothing},
    p::Int = 0,
    q::Int = 0,
    tau::Int = 0,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
)

    idx = 1
    box_cox_parameter = param_vector[idx]
    idx += 1

    alpha = param_vector[idx]
    idx += 1

    if use_beta
        if use_small_phi
            small_phi = param_vector[idx]
            idx += 1
            beta_v = param_vector[idx]
            idx += 1
            gamma_start = 5
        else
            small_phi = 1.0
            beta_v = param_vector[idx]
            idx += 1
            gamma_start = 4
        end
    else
        small_phi = nothing
        beta_v = nothing
        gamma_start = 3
    end

    if seasonal_periods !== nothing
        n_gamma = length(seasonal_periods)
        gamma_vector = collect(param_vector[gamma_start:(gamma_start+n_gamma-1)])
        final_gamma_pos = gamma_start + n_gamma - 1
    else
        gamma_vector = nothing
        final_gamma_pos = gamma_start - 1
    end

    if p != 0
        ar_coefs = collect(param_vector[(final_gamma_pos+1):(final_gamma_pos+p)])
    else
        ar_coefs = nothing
    end

    if q != 0
        ma_coefs = collect(param_vector[(final_gamma_pos+p+1):end])
    else
        ma_coefs = nothing
    end

    box_cox!(opt_env[:x_nought_buffer], vec(opt_env[:x_nought_untransformed]), 1; lambda=box_cox_parameter)
    x_nought = reshape(opt_env[:x_nought_buffer], :, 1)

    w = make_wmatrix(small_phi, seasonal_periods, ar_coefs, ma_coefs)
    g = make_gmatrix(alpha, beta_v, gamma_vector, seasonal_periods, p, q)
    F = make_fmatrix(
        alpha,
        beta_v,
        small_phi,
        seasonal_periods,
        g.gamma_bold_matrix,
        ar_coefs,
        ma_coefs,
    )

    # Ensure w.w_transpose is a matrix (1, n_states)
    w_transpose_mat = w.w_transpose
    if ndims(w_transpose_mat) != 2
        w_transpose_mat = reshape(w_transpose_mat, 1, :)
    end

    opt_env[:w_transpose] = w_transpose_mat
    opt_env[:g] = reshape(g.g, :, 1)
    opt_env[:gamma_bold_matrix] = g.gamma_bold_matrix
    opt_env[:F] = F

    box_cox!(opt_env[:y_vec_buffer], vec(opt_env[:y]), 1; lambda=box_cox_parameter)
    n = size(opt_env[:y], 2)
    mat_transformed_y = reshape(opt_env[:y_vec_buffer], 1, n)

    calc_bats_faster(
        mat_transformed_y,
        opt_env[:y_hat],
        opt_env[:w_transpose],
        opt_env[:F],
        opt_env[:x],
        opt_env[:g],
        opt_env[:e],
        x_nought;
        seasonal_periods = seasonal_periods,
        beta_v = beta_v,
        tau = tau,
        p = p,
        q = q,
        Fx_buffer = opt_env[:Fx_buffer],
    )

    log_likelihood =
        n * log(sum(abs2, opt_env[:e])) -
        2 * (box_cox_parameter - 1) * sum(log.(opt_env[:y]))

    D = opt_env[:F] - opt_env[:g] * opt_env[:w_transpose]
    opt_env[:D] = D

    is_admissible = check_admissibility(
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

@inline function calc_likelihood2(
    param_vector::Vector{Float64},
    opt_env::Dict{Symbol,Any},
    x_nought::AbstractMatrix;
    use_beta::Bool,
    use_small_phi::Bool,
    seasonal_periods::Union{Vector{Int},Nothing},
    p::Int = 0,
    q::Int = 0,
    tau::Int = 0,
)

    idx = 1

    alpha = param_vector[idx]
    idx += 1

    if use_beta
        if use_small_phi
            small_phi = param_vector[idx]
            idx += 1
            beta_v = param_vector[idx]
            idx += 1
            gamma_start = 4
        else
            small_phi = 1.0
            beta_v = param_vector[idx]
            idx += 1
            gamma_start = 3
        end
    else
        small_phi = nothing
        beta_v = nothing
        gamma_start = 2
    end

    if seasonal_periods !== nothing
        n_gamma = length(seasonal_periods)
        gamma_vector = collect(param_vector[gamma_start:(gamma_start+n_gamma-1)])
        final_gamma_pos = gamma_start + n_gamma - 1
    else
        gamma_vector = nothing
        final_gamma_pos = gamma_start - 1
    end

    if p != 0
        ar_coefs = collect(param_vector[(final_gamma_pos+1):(final_gamma_pos+p)])
    else
        ar_coefs = nothing
    end

    if q != 0
        ma_coefs = collect(param_vector[(final_gamma_pos+p+1):end])
    else
        ma_coefs = nothing
    end

    w = make_wmatrix(small_phi, seasonal_periods, ar_coefs, ma_coefs)
    g = make_gmatrix(alpha, beta_v, gamma_vector, seasonal_periods, p, q)
    F = make_fmatrix(
        alpha,
        beta_v,
        small_phi,
        seasonal_periods,
        g.gamma_bold_matrix,
        ar_coefs,
        ma_coefs,
    )

    # Ensure w.w_transpose is a matrix (1, n_states)
    w_transpose_mat = w.w_transpose
    if ndims(w_transpose_mat) != 2
        w_transpose_mat = reshape(w_transpose_mat, 1, :)
    end

    opt_env[:w_transpose] = w_transpose_mat
    opt_env[:g] = reshape(g.g, :, 1)
    opt_env[:gamma_bold_matrix] = g.gamma_bold_matrix
    opt_env[:F] = F

    n = size(opt_env[:y], 2)

    calc_bats_faster(
        opt_env[:y],
        opt_env[:y_hat],
        opt_env[:w_transpose],
        opt_env[:F],
        opt_env[:x],
        opt_env[:g],
        opt_env[:e],
        x_nought;
        seasonal_periods = seasonal_periods,
        beta_v = beta_v,
        tau = tau,
        p = p,
        q = q,
        Fx_buffer = opt_env[:Fx_buffer],
    )

    log_likelihood = n * log(sum(abs2, opt_env[:e]))

    D = opt_env[:F] - opt_env[:g] * opt_env[:w_transpose]
    opt_env[:D] = D

    is_admissible = check_admissibility(
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
