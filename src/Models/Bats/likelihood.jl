# ─── BATS likelihood functions ────────────────────────────────────────────────

function _bats_update_matrices!(
    opt_env::Dict{Symbol,Any},
    alpha::Float64,
    beta_v::Union{Float64,Nothing},
    small_phi::Union{Float64,Nothing},
    gamma_vector::Union{AbstractVector{<:Real},Nothing},
    ar_coefs::Union{AbstractVector{<:Real},Nothing},
    ma_coefs::Union{AbstractVector{<:Real},Nothing},
    seasonal_periods::Union{Vector{Int},Nothing},
)
    layout = opt_env[:layout]::BATSMatrixLayout
    F = opt_env[:F]::Matrix{Float64}
    w_t = opt_env[:w_transpose]::Matrix{Float64}
    g = opt_env[:g]::Matrix{Float64}

    gamma_bold = opt_env[:gamma_bold_matrix]
    if gamma_bold !== nothing && gamma_vector !== nothing && seasonal_periods !== nothing
        update_bats_gamma_bold!(gamma_bold, seasonal_periods, gamma_vector)
    end

    update_bats_wg!(w_t, g, layout, alpha, beta_v, small_phi,
                    gamma_bold, ar_coefs, ma_coefs)
    update_bats_fmatrix!(F, layout, alpha, beta_v, small_phi,
                         gamma_bold, ar_coefs, ma_coefs)

    # D = F - g * w' (outer product), fused into the preallocated buffer.
    D = opt_env[:D]::Matrix{Float64}
    @. D = F - g * w_t
    return F, w_t, g, D
end

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

    F, w_t, g, D = _bats_update_matrices!(opt_env, alpha, beta_v, small_phi,
                                          gamma_vector, ar_coefs, ma_coefs,
                                          seasonal_periods)

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
    is_admissible || return 1e20

    box_cox!(opt_env[:x_nought_buffer], vec(opt_env[:x_nought_untransformed]), 1; lambda=box_cox_parameter)
    x_nought = reshape(opt_env[:x_nought_buffer], :, 1)

    box_cox!(opt_env[:y_vec_buffer], vec(opt_env[:y]), 1; lambda=box_cox_parameter)
    n = size(opt_env[:y], 2)
    mat_transformed_y = reshape(opt_env[:y_vec_buffer], 1, n)

    calc_bats_faster(
        mat_transformed_y,
        opt_env[:y_hat],
        w_t,
        F,
        opt_env[:x],
        g,
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
        2 * (box_cox_parameter - 1) * (opt_env[:sum_log_y]::Float64)

    return log_likelihood
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

    F, w_t, g, D = _bats_update_matrices!(opt_env, alpha, beta_v, small_phi,
                                          gamma_vector, ar_coefs, ma_coefs,
                                          seasonal_periods)

    is_admissible = check_admissibility(
        D;
        box_cox = nothing,
        small_phi = small_phi,
        ar_coefs = ar_coefs,
        ma_coefs = ma_coefs,
        tau = tau,
    )
    is_admissible || return 1e20

    n = size(opt_env[:y], 2)

    calc_bats_faster(
        opt_env[:y],
        opt_env[:y_hat],
        w_t,
        F,
        opt_env[:x],
        g,
        opt_env[:e],
        x_nought;
        seasonal_periods = seasonal_periods,
        beta_v = beta_v,
        tau = tau,
        p = p,
        q = q,
        Fx_buffer = opt_env[:Fx_buffer],
    )

    return n * log(sum(abs2, opt_env[:e]))
end
