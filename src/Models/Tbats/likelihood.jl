# ─── TBATS likelihood functions ───────────────────────────────────────────────
#
# Objective functions for the optimizer: calc_likelihood_tbats (with Box-Cox)
# and calc_likelihood_tbats_notransform (without Box-Cox).
#
# Per-evaluation work is kept minimal: the structural layout of w, g and F is
# fixed across evaluations, so only parameter-dependent entries are overwritten
# in preallocated buffers, and the admissibility check (which depends only on
# the parameters, not on the filtered states) runs BEFORE the O(n·dim²) state
# recursion so inadmissible candidate points skip it entirely. The returned
# objective values are identical to a full rebuild-then-filter evaluation.

function _tbats_update_matrices!(opt_env::Dict{Symbol,Any}, paramz, k_vector)
    layout = opt_env[:layout]::TBATSMatrixLayout
    F = opt_env[:F]::Matrix{Float64}
    w_t = opt_env[:w_transpose]::Matrix{Float64}
    g = opt_env[:g_vec]::Vector{Float64}

    gamma_bold = opt_env[:gamma_bold_matrix]
    if gamma_bold !== nothing && paramz.gamma_one_v !== nothing
        update_tbats_gamma_bold!(gamma_bold, k_vector, paramz.gamma_one_v, paramz.gamma_two_v)
    end

    update_tbats_wg!(w_t, g, layout, paramz.alpha, paramz.beta, paramz.small_phi,
                     gamma_bold, paramz.ar_coefs, paramz.ma_coefs)
    update_tbats_fmatrix!(F, layout, paramz.alpha, paramz.beta, paramz.small_phi,
                          gamma_bold, paramz.ar_coefs, paramz.ma_coefs)

    # D = F - g * w' (outer product), fused into the preallocated buffer.
    D = opt_env[:D]::Matrix{Float64}
    @. D = F - g * w_t
    return F, w_t, g, D
end

function _tbats_filter!(opt_env::Dict{Symbol,Any}, transformed_y, x_nought::Vector{Float64},
                        F::Matrix{Float64}, w_t::Matrix{Float64}, g::Vector{Float64})
    y_hat = opt_env[:y_hat]::Matrix{Float64}
    e = opt_env[:e]::Matrix{Float64}
    x = opt_env[:x]::Matrix{Float64}
    n = size(e, 2)

    @inbounds for t = 1:n
        xprev = t == 1 ? x_nought : view(x, :, t-1)
        y_hat[1, t] = dot(w_t, xprev)
        et = transformed_y[t] - y_hat[1, t]
        e[1, t] = et
        xt = view(x, :, t)
        mul!(xt, F, xprev)
        @. xt = xt + g * et
    end
    return e
end

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

    F, w_t, g, D = _tbats_update_matrices!(opt_env, paramz, k_vector)

    is_admissible = check_admissibility_tbats(
        D;
        box_cox = box_cox_parameter,
        small_phi = paramz.small_phi,
        ar_coefs = paramz.ar_coefs,
        ma_coefs = paramz.ma_coefs,
        tau = tau,
        bc_lower = bc_lower,
        bc_upper = bc_upper,
    )
    is_admissible || return 1e20

    box_cox!(opt_env[:box_cox_buffer_x], vec(opt_env[:x_nought_untransformed]), 1; lambda=box_cox_parameter)
    x_nought = opt_env[:box_cox_buffer_x]::Vector{Float64}

    box_cox!(opt_env[:box_cox_buffer_y], vec(opt_env[:y]), 1; lambda=box_cox_parameter)
    transformed_y = opt_env[:box_cox_buffer_y]::Vector{Float64}

    e = _tbats_filter!(opt_env, transformed_y, x_nought, F, w_t, g)

    n = size(e, 2)
    log_likelihood = n * log(sum(abs2, e)) -
                     2 * (box_cox_parameter - 1) * (opt_env[:sum_log_y]::Float64)

    return log_likelihood
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

    F, w_t, g, D = _tbats_update_matrices!(opt_env, paramz, k_vector)

    is_admissible = check_admissibility_tbats(
        D;
        box_cox = nothing,
        small_phi = paramz.small_phi,
        ar_coefs = paramz.ar_coefs,
        ma_coefs = paramz.ma_coefs,
        tau = tau,
    )
    is_admissible || return 1e20

    e = _tbats_filter!(opt_env, vec(opt_env[:y]), vec(x_nought), F, w_t, g)

    n = size(e, 2)
    return n * log(sum(e .* e))
end
