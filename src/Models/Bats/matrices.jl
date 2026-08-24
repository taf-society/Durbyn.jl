# ─── BATS state space matrix construction ─────────────────────────────────────
#
# Functions for building the state transition matrix F, measurement vector w,
# gain vector g, and initial state vector x₀ from model parameters.
# Also includes parameter packing/unpacking for the optimizer.

function make_xmatrix(
    l::Float64,
    b::Union{Float64,Nothing} = nothing,
    s_vector::Union{Vector{Float64},Nothing} = nothing,
    d_vector::Union{Vector{Float64},Nothing} = nothing,
    epsilon_vector::Union{Vector{Float64},Nothing} = nothing,
)
    x = [l]
    !isnothing(b) && push!(x, b)
    !isnothing(s_vector) && append!(x, s_vector)
    !isnothing(d_vector) && append!(x, d_vector)
    !isnothing(epsilon_vector) && append!(x, epsilon_vector)

    x_transpose = reshape(x, 1, :)
    x_col = reshape(x, :, 1)

    return (x = x_col, x_transpose = x_transpose)
end

function make_parscale_bats(control::ParameterControl)
    parscale = Float64[]

    if control.use_box_cox
        push!(parscale, 0.001)
        push!(parscale, 0.1)
    else
        push!(parscale, 0.1)
    end

    if control.use_beta
        control.use_damping && push!(parscale, 1e-2)
        push!(parscale, 1e-2)
    end

    control.length_gamma > 0 && append!(parscale, fill(1e-2, control.length_gamma))
    (control.p + control.q > 0) && append!(parscale, fill(1e-1, control.p + control.q))

    return parscale
end

function parameterise(
    alpha::Float64,
    beta_v::Union{Float64,Nothing},
    small_phi::Union{Float64,Nothing},
    gamma_v::Union{Vector{Float64},Nothing},
    lambda::Union{Float64,Nothing},
    ar_coefs::Union{Vector{Float64},Nothing},
    ma_coefs::Union{Vector{Float64},Nothing},
)

    param_vector = Float64[]

    use_box_cox = !isnothing(lambda)
    use_box_cox && push!(param_vector, lambda)
    push!(param_vector, alpha)

    if !isnothing(beta_v)
        use_beta = true
        if !isnothing(small_phi) && small_phi != 1.0
            push!(param_vector, small_phi)
            use_damping = true
        else
            use_damping = false
        end
        push!(param_vector, beta_v)
    else
        use_beta = false
        use_damping = false
    end

    length_gamma = isnothing(gamma_v) ? 0 : length(gamma_v)
    !isnothing(gamma_v) && append!(param_vector, gamma_v)

    p = isnothing(ar_coefs) ? 0 : length(ar_coefs)
    !isnothing(ar_coefs) && append!(param_vector, ar_coefs)

    q = isnothing(ma_coefs) ? 0 : length(ma_coefs)
    !isnothing(ma_coefs) && append!(param_vector, ma_coefs)

    control = ParameterControl(use_box_cox, use_beta, use_damping, length_gamma, p, q)
    return (vect = param_vector, control = control)
end

function unparameterise(param_vector::Vector{Float64}, control::ParameterControl)
    idx = 1

    lambda = nothing
    if control.use_box_cox
        lambda = param_vector[idx]
        idx += 1
    end

    alpha = param_vector[idx]
    idx += 1

    if control.use_beta
        if control.use_damping
            small_phi = param_vector[idx]
            idx += 1
        else
            small_phi = 1.0
        end
        beta = param_vector[idx]
        idx += 1
    else
        small_phi = nothing
        beta = nothing
    end

    if control.length_gamma > 0
        gamma_vector = param_vector[idx:(idx+control.length_gamma-1)]
        idx += control.length_gamma
    else
        gamma_vector = nothing
    end

    if control.p > 0
        ar_coefs = param_vector[idx:(idx+control.p-1)]
        idx += control.p
    else
        ar_coefs = nothing
    end

    if control.q > 0
        ma_coefs = param_vector[idx:end]
    else
        ma_coefs = nothing
    end

    return (
        lambda = lambda,
        alpha = alpha,
        beta = beta,
        small_phi = small_phi,
        gamma_v = gamma_vector,
        ar_coefs = ar_coefs,
        ma_coefs = ma_coefs,
    )
end

function build_seasonal_amatrix(seasonal_periods::Vector{Int})
    A = Matrix{Float64}(undef, 0, 0)

    for (i, m) in enumerate(seasonal_periods)
        a_row_one = zeros(1, m)
        a_row_one[end] = 1.0

        Ai =
            m > 1 ?
            vcat(a_row_one, hcat(Matrix{Float64}(I, m - 1, m - 1), zeros(m - 1, 1))) :
            a_row_one

        if i == 1
            A = Ai
        else
            old_rows, old_cols = size(A)
            A = vcat(A, zeros(size(Ai, 1), old_cols))
            A = hcat(A, zeros(size(A, 1), size(Ai, 2)))
            A[(old_rows+1):end, (old_cols+1):end] = Ai
        end
    end

    return A
end

@inline function make_fmatrix(
    alpha::Float64,
    beta::Union{Float64,Nothing},
    small_phi::Union{Float64,Nothing},
    seasonal_periods::Union{Vector{Int},Nothing},
    gamma_bold_matrix::Union{Matrix{Float64},Nothing},
    ar_coefs::Union{AbstractVector{<:Real},Nothing},
    ma_coefs::Union{AbstractVector{<:Real},Nothing},
)

    F = ones(1, 1)
    !isnothing(beta) && (F = hcat(F, small_phi))

    if !isnothing(seasonal_periods)
        tau = sum(seasonal_periods)
        F = hcat(F, zeros(1, tau))
    end

    p = isnothing(ar_coefs) ? 0 : length(ar_coefs)
    q = isnothing(ma_coefs) ? 0 : length(ma_coefs)

    p > 0 && (F = hcat(F, alpha .* ar_coefs'))
    q > 0 && (F = hcat(F, alpha .* ma_coefs'))

    if !isnothing(beta)
        beta_row = [0.0 small_phi]
        !isnothing(seasonal_periods) &&
            (beta_row = hcat(beta_row, zeros(1, sum(seasonal_periods))))
        p > 0 && (beta_row = hcat(beta_row, beta .* ar_coefs'))
        q > 0 && (beta_row = hcat(beta_row, beta .* ma_coefs'))
        F = vcat(F, beta_row)
    end

    if !isnothing(seasonal_periods)
        tau = sum(seasonal_periods)
        seasonal_rows = zeros(tau, 1)
        !isnothing(beta) && (seasonal_rows = hcat(seasonal_rows, zeros(tau, 1)))

        A = build_seasonal_amatrix(seasonal_periods)
        seasonal_rows = hcat(seasonal_rows, A)

        p > 0 && (seasonal_rows = hcat(seasonal_rows, gamma_bold_matrix' * reshape(ar_coefs, 1, :)))
        q > 0 && (seasonal_rows = hcat(seasonal_rows, gamma_bold_matrix' * reshape(ma_coefs, 1, :)))

        F = vcat(F, seasonal_rows)
    end

    if p > 0
        ar_rows = zeros(p, 1)
        !isnothing(beta) && (ar_rows = hcat(ar_rows, zeros(p, 1)))
        !isnothing(seasonal_periods) &&
            (ar_rows = hcat(ar_rows, zeros(p, sum(seasonal_periods))))

        ar_part =
            p > 1 ?
            vcat(ar_coefs', hcat(Matrix{Float64}(I, p - 1, p - 1), zeros(p - 1, 1))) :
            ar_coefs'
        ar_rows = hcat(ar_rows, ar_part)

        if q > 0
            ma_in_ar = zeros(p, q)
            ma_in_ar[1, :] = ma_coefs
            ar_rows = hcat(ar_rows, ma_in_ar)
        end

        F = vcat(F, ar_rows)
    end

    if q > 0
        ma_rows = zeros(q, 1)
        !isnothing(beta) && (ma_rows = hcat(ma_rows, zeros(q, 1)))
        !isnothing(seasonal_periods) &&
            (ma_rows = hcat(ma_rows, zeros(q, sum(seasonal_periods))))
        p > 0 && (ma_rows = hcat(ma_rows, zeros(q, p)))

        ma_part =
            q > 1 ?
            vcat(zeros(1, q), hcat(Matrix{Float64}(I, q - 1, q - 1), zeros(q - 1, 1))) :
            zeros(1, q)
        ma_rows = hcat(ma_rows, ma_part)

        F = vcat(F, ma_rows)
    end

    return F
end

@inline function make_wmatrix(
    small_phi::Union{Float64,Nothing},
    seasonal_periods::Union{Vector{Int},Nothing},
    ar_coefs::Union{AbstractVector{<:Real},Nothing},
    ma_coefs::Union{AbstractVector{<:Real},Nothing},
)
    w_size = 1
    !isnothing(small_phi) && (w_size += 1)
    !isnothing(seasonal_periods) && (w_size += sum(seasonal_periods))
    !isnothing(ar_coefs) && (w_size += length(ar_coefs))
    !isnothing(ma_coefs) && (w_size += length(ma_coefs))

    w_transpose = Vector{Float64}(undef, w_size)
    idx = 1
    w_transpose[idx] = 1.0
    idx += 1

    if !isnothing(small_phi)
        w_transpose[idx] = small_phi
        idx += 1
    end

    if !isnothing(seasonal_periods)
        for m in seasonal_periods
            for _ in 1:(m-1)
                w_transpose[idx] = 0.0
                idx += 1
            end
            w_transpose[idx] = 1.0
            idx += 1
        end
    end

    if !isnothing(ar_coefs)
        for val in ar_coefs
            w_transpose[idx] = val
            idx += 1
        end
    end

    if !isnothing(ma_coefs)
        for val in ma_coefs
            w_transpose[idx] = val
            idx += 1
        end
    end

    w_transpose_mat = reshape(w_transpose, 1, :)
    return (w_transpose = w_transpose_mat, w = w_transpose_mat')
end

@inline function make_gmatrix(
    alpha::Float64,
    beta::Union{Float64,Nothing},
    gamma_vector::Union{AbstractVector{<:Real},Nothing},
    seasonal_periods::Union{Vector{Int},Nothing},
    p::Int,
    q::Int,
)
    adjustBeta = !isnothing(beta)
    g_size = 1 + (adjustBeta ? 1 : 0)
    gammaLength = 0

    if !isnothing(gamma_vector) && !isnothing(seasonal_periods)
        gammaLength = sum(seasonal_periods)
        g_size += gammaLength
    end
    g_size += p + q

    g = Vector{Float64}(undef, g_size)
    idx = 1

    g[idx] = alpha
    idx += 1

    if adjustBeta
        g[idx] = beta
        idx += 1
    end

    gamma_bold_matrix = nothing

    if !isnothing(gamma_vector) && !isnothing(seasonal_periods)
        gamma_block_start = idx
        for (i, m) in enumerate(seasonal_periods)
            g[idx] = gamma_vector[i]
            idx += 1
            for _ in 1:(m-1)
                g[idx] = 0.0
                idx += 1
            end
        end
        # Create a copy, not a view, for compatibility
        gamma_bold_matrix = reshape(g[gamma_block_start:(idx-1)], 1, :)
    end

    if p > 0
        g[idx] = 1.0
        idx += 1
        for _ in 1:(p-1)
            g[idx] = 0.0
            idx += 1
        end
    end

    if q > 0
        g[idx] = 1.0
        idx += 1
        for _ in 1:(q-1)
            g[idx] = 0.0
            idx += 1
        end
    end

    return (g = g, gamma_bold_matrix = gamma_bold_matrix)
end

# ─── In-place per-evaluation matrix updates ──────────────────────────────────
#
# During optimization the structural layout of w, g, gamma_bold, and F is fixed
# by (use_beta, seasonal_periods, p, q); only parameter-dependent entries change
# between objective evaluations. These helpers overwrite exactly those entries
# in preallocated buffers, producing values identical to a full rebuild via
# make_wmatrix / make_gmatrix / make_fmatrix.

struct BATSMatrixLayout
    has_beta::Bool
    tau::Int
    p::Int
    q::Int
    col_seasonal::Int
    col_ar::Int
    col_ma::Int
    row_ar::Int
end

function BATSMatrixLayout(use_beta::Bool, tau::Int, p::Int, q::Int)
    n_beta = use_beta ? 1 : 0
    col_seasonal = 1 + n_beta + 1
    col_ar = 1 + n_beta + tau + 1
    col_ma = 1 + n_beta + tau + p + 1
    BATSMatrixLayout(use_beta, tau, p, q, col_seasonal, col_ar, col_ma, col_ar)
end

function update_bats_gamma_bold!(
    gamma_bold::Matrix{Float64},
    seasonal_periods::Vector{Int},
    gamma_vector::AbstractVector{<:Real},
)
    idx = 1
    @inbounds for (i, m) in enumerate(seasonal_periods)
        gamma_bold[idx] = gamma_vector[i]
        idx += 1
        for _ in 1:(m-1)
            gamma_bold[idx] = 0.0
            idx += 1
        end
    end
    return gamma_bold
end

function update_bats_wg!(
    w_transpose::Matrix{Float64},
    g::Matrix{Float64},
    layout::BATSMatrixLayout,
    alpha::Float64,
    beta::Union{Float64,Nothing},
    small_phi::Union{Float64,Nothing},
    gamma_bold::Union{Matrix{Float64},Nothing},
    ar_coefs::Union{AbstractVector{<:Real},Nothing},
    ma_coefs::Union{AbstractVector{<:Real},Nothing},
)
    idx = 2
    if small_phi !== nothing
        w_transpose[1, idx] = small_phi
        idx += 1
    end
    idx += layout.tau
    if ar_coefs !== nothing
        @inbounds for c in ar_coefs
            w_transpose[1, idx] = c
            idx += 1
        end
    end
    if ma_coefs !== nothing
        @inbounds for c in ma_coefs
            w_transpose[1, idx] = c
            idx += 1
        end
    end

    gi = 1
    g[gi] = alpha
    gi += 1
    if beta !== nothing
        g[gi] = beta
        gi += 1
    end
    if gamma_bold !== nothing
        @inbounds for v in vec(gamma_bold)
            g[gi] = v
            gi += 1
        end
    end
    return nothing
end

function update_bats_fmatrix!(
    F::Matrix{Float64},
    layout::BATSMatrixLayout,
    alpha::Float64,
    beta::Union{Float64,Nothing},
    small_phi::Union{Float64,Nothing},
    gamma_bold::Union{Matrix{Float64},Nothing},
    ar_coefs::Union{AbstractVector{<:Real},Nothing},
    ma_coefs::Union{AbstractVector{<:Real},Nothing},
)
    p = layout.p
    q = layout.q
    tau = layout.tau
    col_ar = layout.col_ar
    col_ma = layout.col_ma
    row_seasonal = layout.col_seasonal
    row_ar = layout.row_ar

    if layout.has_beta
        F[1, 2] = small_phi
        F[2, 2] = small_phi
    end

    if p > 0
        @inbounds for i in 1:p
            F[1, col_ar + i - 1] = alpha * ar_coefs[i]
        end
        if layout.has_beta
            @inbounds for i in 1:p
                F[2, col_ar + i - 1] = beta * ar_coefs[i]
            end
        end
    end
    if q > 0
        @inbounds for i in 1:q
            F[1, col_ma + i - 1] = alpha * ma_coefs[i]
        end
        if layout.has_beta
            @inbounds for i in 1:q
                F[2, col_ma + i - 1] = beta * ma_coefs[i]
            end
        end
    end

    if tau > 0 && gamma_bold !== nothing
        if p > 0
            @inbounds for j in 1:p, i in 1:tau
                F[row_seasonal + i - 1, col_ar + j - 1] = gamma_bold[i] * ar_coefs[j]
            end
        end
        if q > 0
            @inbounds for j in 1:q, i in 1:tau
                F[row_seasonal + i - 1, col_ma + j - 1] = gamma_bold[i] * ma_coefs[j]
            end
        end
    end

    if p > 0
        @inbounds for i in 1:p
            F[row_ar, col_ar + i - 1] = ar_coefs[i]
        end
        if q > 0
            @inbounds for i in 1:q
                F[row_ar, col_ma + i - 1] = ma_coefs[i]
            end
        end
    end

    return nothing
end
