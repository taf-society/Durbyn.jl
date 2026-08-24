# ─── TBATS state space matrix construction ────────────────────────────────────
#
# Functions for building the trigonometric seasonal matrices (C, S, A),
# state transition matrix F, measurement vector w, gain vector g, and
# initial state vector x₀. Also includes parameter packing/unpacking.

function make_ci_matrix(k::Int, m::Float64)
    C = zeros(k, k)
    for j = 1:k
        C[j, j] = cos(2 * π * j / m)
    end
    return C
end


function make_si_matrix(k::Int, m::Float64)
    S = zeros(k, k)
    for j = 1:k
        S[j, j] = sin(2 * π * j / m)
    end
    return S
end


function make_ai_matrix(C::Matrix{Float64}, S::Matrix{Float64}, k::Int)
    if k == 0
        return zeros(0, 0)
    end

    A = zeros(2k, 2k)
    A[1:k, 1:k] = C
    A[1:k, (k+1):(2k)] = S
    A[(k+1):(2k), 1:k] = -S
    A[(k+1):(2k), (k+1):(2k)] = C

    return A
end


function make_tbats_gamma_bold_matrix(k_vector::Vector{Int}, gamma_one::Vector{Float64}, gamma_two::Vector{Float64})
    tau = 2 * sum(k_vector)
    gamma_bold = zeros(1, tau)

    end_pos = 1
    for (i, k) in enumerate(k_vector)
        for j = end_pos:(end_pos + k - 1)
            gamma_bold[j] = gamma_one[i]
        end
        for j = (end_pos + k):(end_pos + 2k - 1)
            gamma_bold[j] = gamma_two[i]
        end
        end_pos += 2k
    end

    return gamma_bold
end


function make_tbats_fmatrix(
    alpha::Float64,
    beta::Union{Float64,Nothing},
    small_phi::Union{Float64,Nothing},
    seasonal_periods::Union{Vector{<:Real},Nothing},
    k_vector::Union{Vector{Int},Nothing},
    gamma_bold_matrix::Union{Matrix{Float64},Nothing},
    ar_coefs::Union{Vector{Float64},Nothing},
    ma_coefs::Union{Vector{Float64},Nothing},
)
    has_beta = !isnothing(beta)
    has_seasonal = !isnothing(seasonal_periods) && !isnothing(k_vector)
    tau = has_seasonal ? 2 * sum(k_vector) : 0
    p = isnothing(ar_coefs) ? 0 : length(ar_coefs)
    q = isnothing(ma_coefs) ? 0 : length(ma_coefs)

    n_beta = has_beta ? 1 : 0
    n_rows = 1 + n_beta + tau + p + q
    n_cols = n_rows

    F = zeros(n_rows, n_cols)

    col_level = 1
    col_beta = has_beta ? 2 : 0
    col_seasonal = 1 + n_beta + 1
    col_ar = 1 + n_beta + tau + 1
    col_ma = 1 + n_beta + tau + p + 1

    row_level = 1
    row_beta = has_beta ? 2 : 0
    row_seasonal = 1 + n_beta + 1
    row_ar = 1 + n_beta + tau + 1
    row_ma = 1 + n_beta + tau + p + 1

    F[row_level, col_level] = 1.0
    if has_beta
        F[row_level, col_beta] = small_phi
    end

    if p > 0
        for i in 1:p
            F[row_level, col_ar + i - 1] = alpha * ar_coefs[i]
        end
    end
    if q > 0
        for i in 1:q
            F[row_level, col_ma + i - 1] = alpha * ma_coefs[i]
        end
    end

    if has_beta
        F[row_beta, col_level] = 0.0
        F[row_beta, col_beta] = small_phi

        if p > 0
            for i in 1:p
                F[row_beta, col_ar + i - 1] = beta * ar_coefs[i]
            end
        end
        if q > 0
            for i in 1:q
                F[row_beta, col_ma + i - 1] = beta * ma_coefs[i]
            end
        end
    end

    if has_seasonal
        pos = 0
        for (m, k) in zip(seasonal_periods, k_vector)
            Ci = make_ci_matrix(k, Float64(m))
            Si = make_si_matrix(k, Float64(m))
            Ai = make_ai_matrix(Ci, Si, k)

            block_size = 2k
            r_start = row_seasonal + pos
            c_start = col_seasonal + pos
            F[r_start:(r_start+block_size-1), c_start:(c_start+block_size-1)] = Ai
            pos += block_size
        end

        if p > 0 && !isnothing(gamma_bold_matrix)
            B = gamma_bold_matrix' * reshape(ar_coefs, 1, :)
            for j in 1:p
                for i in 1:tau
                    F[row_seasonal + i - 1, col_ar + j - 1] = B[i, j]
                end
            end
        end
        if q > 0 && !isnothing(gamma_bold_matrix)

            C = gamma_bold_matrix' * reshape(ma_coefs, 1, :)
            for j in 1:q
                for i in 1:tau
                    F[row_seasonal + i - 1, col_ma + j - 1] = C[i, j]
                end
            end
        end
    end

    if p > 0
        for i in 1:p
            F[row_ar, col_ar + i - 1] = ar_coefs[i]
        end

        if q > 0
            for i in 1:q
                F[row_ar, col_ma + i - 1] = ma_coefs[i]
            end
        end

        for i in 2:p
            F[row_ar + i - 1, col_ar + i - 2] = 1.0
        end
    end


    if q > 0

        for i in 2:q
            F[row_ma + i - 1, col_ma + i - 2] = 1.0
        end
    end

    return F
end


function make_tbats_wmatrix(
    small_phi::Union{Float64,Nothing},
    k_vector::Union{Vector{Int},Nothing},
    ar_coefs::Union{Vector{Float64},Nothing},
    ma_coefs::Union{Vector{Float64},Nothing},
    tau::Int,
)
    n_phi = isnothing(small_phi) ? 0 : 1
    n_seasonal = tau
    n_ar = isnothing(ar_coefs) ? 0 : length(ar_coefs)
    n_ma = isnothing(ma_coefs) ? 0 : length(ma_coefs)
    total_size = 1 + n_phi + n_seasonal + n_ar + n_ma

    w_transpose = Vector{Float64}(undef, total_size)
    idx = 1

    w_transpose[idx] = 1.0
    idx += 1

    if !isnothing(small_phi)
        w_transpose[idx] = small_phi
        idx += 1
    end

    if !isnothing(k_vector) && tau > 0
        for k in k_vector
            for _ in 1:k
                w_transpose[idx] = 1.0
                idx += 1
            end
            for _ in 1:k
                w_transpose[idx] = 0.0
                idx += 1
            end
        end
    end

    if !isnothing(ar_coefs)
        for c in ar_coefs
            w_transpose[idx] = c
            idx += 1
        end
    end

    if !isnothing(ma_coefs)
        for c in ma_coefs
            w_transpose[idx] = c
            idx += 1
        end
    end

    w_transpose_mat = reshape(w_transpose, 1, :)
    return (w_transpose = w_transpose_mat, w = w_transpose_mat')
end


function make_tbats_gmatrix(
    alpha::Float64,
    beta::Union{Float64,Nothing},
    gamma_one::Union{Vector{Float64},Nothing},
    gamma_two::Union{Vector{Float64},Nothing},
    k_vector::Union{Vector{Int},Nothing},
    p::Int,
    q::Int,
)
    g = Float64[]
    push!(g, alpha)

    adjustBeta = !isnothing(beta)
    adjustBeta && push!(g, beta)

    gamma_bold_matrix = nothing

    if !isnothing(gamma_one) && !isnothing(gamma_two) && !isnothing(k_vector)
        gamma_bold_matrix = make_tbats_gamma_bold_matrix(k_vector, gamma_one, gamma_two)
        append!(g, vec(gamma_bold_matrix))
    end

    if p > 0
        push!(g, 1.0)
        append!(g, zeros(p - 1))
    end

    if q > 0
        push!(g, 1.0)
        append!(g, zeros(q - 1))
    end

    return (g = g, gamma_bold_matrix = gamma_bold_matrix)
end

function make_xmatrix_tbats(
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


function parameterise_tbats(
    alpha::Float64,
    beta_v::Union{Float64,Nothing},
    small_phi::Union{Float64,Nothing},
    gamma_one_v::Union{Vector{Float64},Nothing},
    gamma_two_v::Union{Vector{Float64},Nothing},
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

    length_gamma = 0
    if !isnothing(gamma_one_v) && !isnothing(gamma_two_v)
        append!(param_vector, gamma_one_v)
        append!(param_vector, gamma_two_v)
        length_gamma = length(gamma_one_v) + length(gamma_two_v)
    end

    p = isnothing(ar_coefs) ? 0 : length(ar_coefs)
    !isnothing(ar_coefs) && append!(param_vector, ar_coefs)

    q = isnothing(ma_coefs) ? 0 : length(ma_coefs)
    !isnothing(ma_coefs) && append!(param_vector, ma_coefs)

    control = TBATSParameterControl(use_box_cox, use_beta, use_damping, length_gamma, p, q)
    return (vect = param_vector, control = control)
end

function unparameterise_tbats(param_vector::Vector{Float64}, control::TBATSParameterControl)
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
        half_length = div(control.length_gamma, 2)
        gamma_one_v = param_vector[idx:(idx+half_length-1)]
        idx += half_length
        gamma_two_v = param_vector[idx:(idx+half_length-1)]
        idx += half_length
    else
        gamma_one_v = nothing
        gamma_two_v = nothing
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
        gamma_one_v = gamma_one_v,
        gamma_two_v = gamma_two_v,
        ar_coefs = ar_coefs,
        ma_coefs = ma_coefs,
    )
end

function make_parscale_tbats(control::TBATSParameterControl)
    parscale = Float64[]

    if control.use_box_cox
        push!(parscale, 0.001)
        push!(parscale, 0.01)
    else
        push!(parscale, 0.01)
    end

    if control.use_beta
        control.use_damping && push!(parscale, 1e-2)
        push!(parscale, 1e-2)
    end

    control.length_gamma > 0 && append!(parscale, fill(1e-5, control.length_gamma))
    (control.p + control.q > 0) && append!(parscale, fill(1e-1, control.p + control.q))

    return parscale
end

# ─── In-place per-evaluation matrix updates ──────────────────────────────────
#
# During optimization the structural layout of w, g, gamma_bold, and F is fixed
# by (use_beta, k_vector, p, q); only parameter-dependent entries change between
# objective evaluations. These helpers overwrite exactly those entries in
# preallocated buffers, producing values identical to a full rebuild via
# make_tbats_wmatrix / make_tbats_gmatrix / make_tbats_fmatrix.

struct TBATSMatrixLayout
    has_beta::Bool
    tau::Int
    p::Int
    q::Int
    col_seasonal::Int
    col_ar::Int
    col_ma::Int
    row_ar::Int
end

function TBATSMatrixLayout(use_beta::Bool, tau::Int, p::Int, q::Int)
    n_beta = use_beta ? 1 : 0
    col_seasonal = 1 + n_beta + 1
    col_ar = 1 + n_beta + tau + 1
    col_ma = 1 + n_beta + tau + p + 1
    TBATSMatrixLayout(use_beta, tau, p, q, col_seasonal, col_ar, col_ma, col_ar)
end

function update_tbats_gamma_bold!(
    gamma_bold::Matrix{Float64},
    k_vector::Vector{Int},
    gamma_one::Vector{Float64},
    gamma_two::Vector{Float64},
)
    end_pos = 1
    @inbounds for (i, k) in enumerate(k_vector)
        for j = end_pos:(end_pos + k - 1)
            gamma_bold[j] = gamma_one[i]
        end
        for j = (end_pos + k):(end_pos + 2k - 1)
            gamma_bold[j] = gamma_two[i]
        end
        end_pos += 2k
    end
    return gamma_bold
end

function update_tbats_wg!(
    w_transpose::Matrix{Float64},
    g::Vector{Float64},
    layout::TBATSMatrixLayout,
    alpha::Float64,
    beta::Union{Float64,Nothing},
    small_phi::Union{Float64,Nothing},
    gamma_bold::Union{Matrix{Float64},Nothing},
    ar_coefs::Union{Vector{Float64},Nothing},
    ma_coefs::Union{Vector{Float64},Nothing},
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

function update_tbats_fmatrix!(
    F::Matrix{Float64},
    layout::TBATSMatrixLayout,
    alpha::Float64,
    beta::Union{Float64,Nothing},
    small_phi::Union{Float64,Nothing},
    gamma_bold::Union{Matrix{Float64},Nothing},
    ar_coefs::Union{Vector{Float64},Nothing},
    ma_coefs::Union{Vector{Float64},Nothing},
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
