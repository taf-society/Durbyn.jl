"""
    BATSModel

Container for fitted BATS models (Box-Cox transformation, ARMA errors,
trend and seasonal components) following the definition in
De Livera, Hyndman & Snyder (2011) and the `forecast::bats` R
implementation.  Each field stores the estimated smoothing parameters,
state matrices, diagnostics and metadata required to regenerate forecasts
without re-fitting.

Fields capture Box-Cox lambda, level/trend/seasonal coefficients,
ARMA coefficients, the state matrices `x`/`seed_states`, innovation
variance, information criteria, the original series, optimizer metadata
and a descriptive `method` string such as `BATS(λ, {p,q}, φ, {m…})`.
"""
mutable struct BATSModel
    lambda::Union{Float64,Nothing}
    alpha::Float64
    beta::Union{Float64,Nothing}
    damping_parameter::Union{Float64,Nothing}
    gamma_values::Union{Vector{Float64},Nothing}
    ar_coefficients::Union{Vector{Float64},Nothing}
    ma_coefficients::Union{Vector{Float64},Nothing}
    seasonal_periods::Union{Vector{Int},Nothing}
    fitted_values::Vector{Float64}
    errors::Vector{Float64}
    x::Matrix{Float64}
    seed_states::AbstractArray{Float64}
    variance::Float64
    AIC::Float64
    likelihood::Float64
    optim_return_code::Int
    y::Vector{Float64}
    parameters::Dict{Symbol,Any}
    method::String
end

function bats_descriptor(
    lambda::Union{Float64,Nothing},
    ar_coefficients::Union{Vector{Float64},Nothing},
    ma_coefficients::Union{Vector{Float64},Nothing},
    damping_parameter::Union{Float64,Nothing},
    seasonal_periods::Union{Vector{Int},Nothing},
)
    lambda_str = isnothing(lambda) ? "1" : string(round(lambda, digits = 3))
    ar_count = isnothing(ar_coefficients) ? 0 : length(ar_coefficients)
    ma_count = isnothing(ma_coefficients) ? 0 : length(ma_coefficients)
    damping_str = isnothing(damping_parameter) ? "-" : string(round(damping_parameter, digits = 3))

    buffer = IOBuffer()
    print(buffer, "BATS(", lambda_str, ", {", ar_count, ",", ma_count, "}, ", damping_str, ", ")

    if isnothing(seasonal_periods) || isempty(seasonal_periods)
        print(buffer, "-)")
    else
        print(buffer, "{", join(seasonal_periods, ","), "})")
    end

    return String(take!(buffer))
end

bats_descriptor(model::BATSModel) = bats_descriptor(
    model.lambda,
    model.ar_coefficients,
    model.ma_coefficients,
    model.damping_parameter,
    model.seasonal_periods,
)

Base.string(model::BATSModel) = bats_descriptor(model)

struct ParameterControl
    use_box_cox::Bool
    use_beta::Bool
    use_damping::Bool
    length_gamma::Int
    p::Int
    q::Int
end

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

function calc_model(
    y::Vector{Float64},
    x_nought::Vector{Float64},
    F::Matrix{Float64},
    g::Vector{Float64},
    w::NamedTuple,
)

    n = length(y)
    dim = length(x_nought)

    x = zeros(dim, n)
    y_hat = zeros(n)
    e = zeros(n)

    y_hat[1] = (w.w_transpose*x_nought)[1]
    e[1] = y[1] - y_hat[1]
    x[:, 1] = F * x_nought + g * e[1]

    for t = 2:n
        y_hat[t] = (w.w_transpose*x[:, t-1])[1]
        e[t] = y[t] - y_hat[t]
        x[:, t] = F * x[:, t-1] + g * e[t]
    end

    return (y_hat = y_hat, e = e, x = x)
end

@inline function calc_bats_faster(
    y::AbstractMatrix,
    y_hat::AbstractMatrix,
    w_transpose::AbstractMatrix,
    F::AbstractMatrix,
    x::AbstractMatrix,
    g::AbstractMatrix,
    e::AbstractMatrix,
    x_nought::AbstractMatrix;
    seasonal_periods::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    beta_v::Union{Nothing,Any} = nothing,
    tau::Int = 0,
    p::Int = 0,
    q::Int = 0,
    Fx_buffer::Union{Nothing,AbstractVector} = nothing,
)

    adjBeta = (beta_v === nothing) ? 0 : 1
    lengthArma = p + q

    hasSeasonal = (seasonal_periods !== nothing)
    lengthSeasonal = hasSeasonal ? length(seasonal_periods) : 0

    nT = size(y, 2)
    nStates = size(x_nought, 1)
    Fcols = size(F, 2)

    if hasSeasonal
        @views y_hat[:, 1] = w_transpose[:, 1:(adjBeta+1)] * x_nought[1:(adjBeta+1), 1]

        previousS = 0
        for i = 1:lengthSeasonal
            sp = seasonal_periods[i]
            @views y_hat[1, 1] += x_nought[previousS+sp+adjBeta+1, 1]
            previousS += sp
        end
        if lengthArma > 0
            jstart = tau + adjBeta + 2
            @views y_hat[:, 1] .+=
                w_transpose[:, jstart:nStates] * x_nought[jstart:nStates, 1]
        end

        e[1, 1] = y[1, 1] - y_hat[1, 1]

        @views x[1:(adjBeta+1), 1] =
            F[1:(adjBeta+1), 1:(adjBeta+1)] * x_nought[1:(adjBeta+1), 1]

        if lengthArma > 0
            jstart = adjBeta + tau + 2
            @views x[1:(adjBeta+1), 1] .+=
                F[1:(adjBeta+1), jstart:Fcols] * x_nought[jstart:Fcols, 1]
        end

        previousS = 0
        for i = 1:lengthSeasonal
            sp = seasonal_periods[i]
            row_head = adjBeta + previousS + 2
            row_from = adjBeta + previousS + sp + 1
            x[row_head, 1] = x_nought[row_from, 1]

            if lengthArma > 0
                jstart = adjBeta + tau + 2
                @views x[row_head, 1] +=
                    dot(F[row_head, jstart:Fcols], x_nought[jstart:Fcols, 1])
            end

            rowDestStart = adjBeta + previousS + 3
            rowDestEnd = adjBeta + previousS + sp + 1
            rowSrcStart = adjBeta + previousS + 2
            rowSrcEnd = adjBeta + previousS + sp
            if rowDestStart <= rowDestEnd
                @views x[rowDestStart:rowDestEnd, 1] .= x_nought[rowSrcStart:rowSrcEnd, 1]
            end

            previousS += sp
        end

        if p > 0
            idx_p1 = adjBeta + tau + 2
            @views x[idx_p1, 1] = dot(F[idx_p1, idx_p1:Fcols], x_nought[idx_p1:Fcols, 1])

            if p > 1
                rowDestStart = adjBeta + tau + 3
                rowDestEnd = adjBeta + tau + p + 1
                rowSrcStart = adjBeta + tau + 2
                rowSrcEnd = adjBeta + tau + p
                if rowDestStart <= rowDestEnd
                    @views x[rowDestStart:rowDestEnd, 1] .=
                        x_nought[rowSrcStart:rowSrcEnd, 1]
                end
            end
        end

        if q > 0
            idx_q1 = adjBeta + tau + p + 2
            x[idx_q1, 1] = 0.0
            if q > 1
                rowDestStart = adjBeta + tau + p + 3
                rowDestEnd = adjBeta + tau + p + q + 1
                rowSrcStart = adjBeta + tau + p + 2
                rowSrcEnd = adjBeta + tau + p + q
                if rowDestStart <= rowDestEnd
                    @views x[rowDestStart:rowDestEnd, 1] .=
                        x_nought[rowSrcStart:rowSrcEnd, 1]
                end
            end
        end

        x[1, 1] += g[1, 1] * e[1, 1]

        if adjBeta == 1
            x[2, 1] += g[2, 1] * e[1, 1]
        end

        previousS = 0
        for i = 1:lengthSeasonal
            sp = seasonal_periods[i]
            rowS = adjBeta + previousS + 2
            x[rowS, 1] += g[rowS, 1] * e[1, 1]
            previousS += sp
        end

        if p > 0
            idx_p1 = adjBeta + tau + 2
            x[idx_p1, 1] += e[1, 1]
            if q > 0
                idx_q1 = adjBeta + tau + p + 2
                x[idx_q1, 1] += e[1, 1]
            end
        elseif q > 0
            idx_q1 = adjBeta + tau + 2
            x[idx_q1, 1] += e[1, 1]
        end

        @inbounds for t = 1:(nT-1)
            col_t = t + 1

            @views y_hat[:, col_t] .= w_transpose[:, 1:(adjBeta+1)] * x[1:(adjBeta+1), t]

            previousS = 0
            for i = 1:lengthSeasonal
                sp = seasonal_periods[i]
                @views y_hat[1, col_t] += x[previousS+sp+adjBeta+1, t]
                previousS += sp
            end

            if lengthArma > 0
                jstart = tau + adjBeta + 2
                @views y_hat[:, col_t] .+=
                    w_transpose[:, jstart:nStates] * x[jstart:nStates, t]
            end

            e[1, col_t] = y[1, col_t] - y_hat[1, col_t]

            @views x[1:(adjBeta+1), col_t] .=
                F[1:(adjBeta+1), 1:(adjBeta+1)] * x[1:(adjBeta+1), t]

            if lengthArma > 0
                jstart = adjBeta + tau + 2
                @views x[1:(adjBeta+1), col_t] .+=
                    F[1:(adjBeta+1), jstart:Fcols] * x[jstart:Fcols, t]
            end

            previousS = 0
            for i = 1:lengthSeasonal
                sp = seasonal_periods[i]

                row_head = adjBeta + previousS + 2
                row_from = adjBeta + previousS + sp + 1

                x[row_head, col_t] = x[row_from, t]

                if lengthArma > 0
                    jstart = adjBeta + tau + 2
                    @views x[row_head, col_t] +=
                        dot(F[row_head, jstart:Fcols], x[jstart:Fcols, t])
                end

                rowDestStart = adjBeta + previousS + 3
                rowDestEnd = adjBeta + previousS + sp + 1
                rowSrcStart = adjBeta + previousS + 2
                rowSrcEnd = adjBeta + previousS + sp
                if rowDestStart <= rowDestEnd
                    @views x[rowDestStart:rowDestEnd, col_t] .= x[rowSrcStart:rowSrcEnd, t]
                end

                previousS += sp
            end

            if p > 0
                idx_p1 = adjBeta + tau + 2
                @views x[idx_p1, col_t] = dot(F[idx_p1, idx_p1:Fcols], x[idx_p1:Fcols, t])

                if p > 1
                    rowDestStart = adjBeta + tau + 3
                    rowDestEnd = adjBeta + tau + p + 1
                    rowSrcStart = adjBeta + tau + 2
                    rowSrcEnd = adjBeta + tau + p
                    if rowDestStart <= rowDestEnd
                        @views x[rowDestStart:rowDestEnd, col_t] .=
                            x[rowSrcStart:rowSrcEnd, t]
                    end
                end
            end

            if q > 0
                idx_q1 = adjBeta + tau + p + 2
                x[idx_q1, col_t] = 0.0
                if q > 1
                    rowDestStart = adjBeta + tau + p + 3
                    rowDestEnd = adjBeta + tau + p + q + 1
                    rowSrcStart = adjBeta + tau + p + 2
                    rowSrcEnd = adjBeta + tau + p + q
                    if rowDestStart <= rowDestEnd
                        @views x[rowDestStart:rowDestEnd, col_t] .=
                            x[rowSrcStart:rowSrcEnd, t]
                    end
                end
            end

            x[1, col_t] += g[1, 1] * e[1, col_t]
            if adjBeta == 1
                x[2, col_t] += g[2, 1] * e[1, col_t]
            end

            previousS = 0
            for i = 1:lengthSeasonal
                sp = seasonal_periods[i]
                rowS = adjBeta + previousS + 2
                x[rowS, col_t] += g[rowS, 1] * e[1, col_t]
                previousS += sp
            end

            if p > 0
                idx_p1 = adjBeta + tau + 2
                x[idx_p1, col_t] += e[1, col_t]
                if q > 0
                    idx_q1 = adjBeta + tau + p + 2
                    x[idx_q1, col_t] += e[1, col_t]
                end
            elseif q > 0
                idx_q1 = adjBeta + tau + 2
                x[idx_q1, col_t] += e[1, col_t]
            end
        end

    else
        if Fx_buffer !== nothing
            y_hat[1, 1] = dot(view(w_transpose, 1, :), view(x_nought, :, 1))
            e[1, 1] = y[1, 1] - y_hat[1, 1]
            mul!(Fx_buffer, F, view(x_nought, :, 1))
            @. x[:, 1] = Fx_buffer + g * e[1, 1]

            @inbounds for t = 1:(nT-1)
                col_t = t + 1
                y_hat[1, col_t] = dot(view(w_transpose, 1, :), view(x, :, t))
                e[1, col_t] = y[1, col_t] - y_hat[1, col_t]
                mul!(Fx_buffer, F, view(x, :, t))
                @. x[:, col_t] = Fx_buffer + g * e[1, col_t]
            end
        else
            # Fallback path without buffer
            @views y_hat[:, 1] .= w_transpose * x_nought[:, 1]
            e[1, 1] = y[1, 1] - y_hat[1, 1]
            @views x[:, 1] .= F * x_nought[:, 1] .+ g .* e[1, 1]

            @inbounds for t = 1:(nT-1)
                col_t = t + 1
                @views y_hat[:, col_t] .= w_transpose * x[:, t]
                e[1, col_t] = y[1, col_t] - y_hat[1, col_t]
                @views x[:, col_t] .= F * x[:, t] .+ g .* e[1, col_t]
            end
        end
    end

    return nothing
end

function check_admissibility(
    D::AbstractMatrix{<:Real};
    box_cox::Union{Nothing,Float64} = nothing,
    small_phi::Union{Nothing,Float64} = nothing,
    ar_coefs::Union{Nothing,Vector{Float64}} = nothing,
    ma_coefs::Union{Nothing,Vector{Float64}} = nothing,
    tau::Int = 0,
    bc_lower::Float64 = 0.0,
    bc_upper::Float64 = 1.0,
)::Bool

    EPS = 1e-8
    RAD = 1.0 + 1e-2

    if box_cox !== nothing
        (box_cox <= bc_lower || box_cox >= bc_upper) && return false
    end

    if small_phi !== nothing
        (small_phi < 0.8 || small_phi > 1.0) && return false
    end

    if ar_coefs !== nothing
        p = 0
        @inbounds for i in eachindex(ar_coefs)
            if abs(ar_coefs[i]) > EPS
                p = i
            end
        end
        if p > 0
            coeffs = Vector{Float64}(undef, p + 1)
            coeffs[1] = 1.0
            @inbounds for i in 1:p
                coeffs[i + 1] = -ar_coefs[i]
            end
            rts = roots(Polynomial(coeffs))
            @inbounds for r in rts
                abs(r) < RAD && return false
            end
        end
    end

    if ma_coefs !== nothing
        q = 0
        @inbounds for i in eachindex(ma_coefs)
            if abs(ma_coefs[i]) > EPS
                q = i
            end
        end
        if q > 0
            coeffs = Vector{Float64}(undef, q + 1)
            coeffs[1] = 1.0
            @inbounds for i in 1:q
                coeffs[i + 1] = ma_coefs[i]
            end
            rts = roots(Polynomial(coeffs))
            @inbounds for r in rts
                abs(r) < RAD && return false
            end
        end
    end

    vals = eigvals(D)
    @inbounds for v in vals
        abs(v) >= RAD && return false
    end

    return true
end


function cutW(
    use_beta::Bool,
    w_tilda_transpose::AbstractMatrix{T},
    seasonal_periods::AbstractVector{Int},
    p::Int = 0,
    q::Int = 0,
) where {T}

    n_seasons = length(seasonal_periods)

    mask_vector = zeros(Int, n_seasons)


    i = n_seasons
    while i > 1
        for j = 1:(i-1)
            if seasonal_periods[i] % seasonal_periods[j] == 0
                mask_vector[j] = 1
            end
        end
        i -= 1
    end


    if n_seasons > 1
        for s = n_seasons:-1:2
            for j = (s-1):-1:1
                hcf = gcd(seasonal_periods[s], seasonal_periods[j])
                if hcf != 1
                    if mask_vector[s] != 1 && mask_vector[j] != 1
                        mask_vector[s] = -hcf
                    end
                end
            end
        end
    end


    n_cols = size(w_tilda_transpose, 2)

    cols = collect(1:n_cols)

    w_pos_counter = 1
    w_pos = 1
    if use_beta
        w_pos += 1
    end

    for s in seasonal_periods
        mv = mask_vector[w_pos_counter]

        if mv == 1

            first_del = w_pos + 1
            last_del = w_pos + s
            deleteat!(cols, first_del:last_del)

        elseif mv < 0

            w_pos += s
            first_del = w_pos + mv + 1
            last_del = w_pos
            deleteat!(cols, first_del:last_del)
            w_pos += mv

        else
            w_pos += s
            deleteat!(cols, w_pos)
            w_pos -= 1
        end

        w_pos_counter += 1
    end

    if p != 0 || q != 0
        total_cut = p + q
        if total_cut > 0
            start_cut = length(cols) - total_cut + 1
            deleteat!(cols, start_cut:length(cols))
        end
    end

    w_cut = w_tilda_transpose[:, cols]

    return (matrix = w_cut, mask_vector = mask_vector)
end

function calc_seasonal_seeds(
    use_beta::Bool,
    coefs::AbstractVector{T},
    seasonal_periods::AbstractVector{Int},
    mask_vector::AbstractVector{Int},
    p::Int = 0,
    q::Int = 0,
) where {T<:Real}

    x_pos_counter = 1
    sum_k = zero(T)

    if use_beta
        x_pos = 2
        new_x_nought = reshape(coefs[1:2], :, 1)
    else
        x_pos = 1
        new_x_nought = reshape(coefs[1:1], :, 1)
    end

    x_pos_counter = 1

    for s in seasonal_periods
        mv = mask_vector[x_pos_counter]

        if mv == 1
            season = zeros(T, s, 1)
            new_x_nought = vcat(new_x_nought, season)

        elseif mv < 0

            last_idx = x_pos + s + mv
            extract = coefs[(x_pos+1):last_idx]

            k = sum(extract)
            sum_k += k / s

            current_periodicity = extract .- (k / s)
            current_periodicity_mat = reshape(current_periodicity, :, 1)

            additional = fill(-k / s, -mv, 1)
            current_periodicity_mat = vcat(current_periodicity_mat, additional)

            new_x_nought = vcat(new_x_nought, current_periodicity_mat)

            x_pos = x_pos + s + mv

        else

            last_idx = x_pos + s - 1
            slice = coefs[(x_pos+1):last_idx]

            k = sum(slice)
            sum_k += k / s

            current_periodicity = vcat(slice .- (k / s), -k / s)
            current_periodicity_mat = reshape(current_periodicity, :, 1)

            new_x_nought = vcat(new_x_nought, current_periodicity_mat)

            x_pos = x_pos + s - 1
        end

        x_pos_counter += 1
    end

    if p != 0 || q != 0
        arma_len = p + q
        arma_seed_states = zeros(T, arma_len, 1)
        x_nought = vcat(new_x_nought, arma_seed_states)
    else
        x_nought = new_x_nought
    end

    return x_nought
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
    bc_lower::Float64 = 0.0,
    bc_upper::Float64 = 1.0,
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


function fitSpecificBATS(
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
    bc_lower::Float64 = 0.0,
    bc_upper::Float64 = 1.0,
    biasadj::Bool = false,
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

                lambda = box_cox_lambda(y, 1; lower = 0.0, upper = 1.5)
            end

            y_transformed, lambda = box_cox(y, 1; lambda=lambda)
        else
            lambda = nothing
        end

    else

        paramz = unparameterise(starting_params.vect, starting_params.control)

        lambda = paramz.lambda
        alpha = paramz.alpha
        beta_v = paramz.beta
        b = 0.0
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

        list_cut_w = cutW(use_beta, w_tilda_transpose, seasonal_periods, p, q)
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
        opt_result = optim(
            scaled_param0,
            objective_scaled;
            method = "Nelder-Mead",
            control = Dict("maxit" => maxit),
        )

        opt_par_scaled = opt_result.par
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
            opt_result = optim(
                scaled_param0,
                objective_scaled;
                method = "Nelder-Mead",
                control = Dict("maxit" => maxit),
            )
        else

            opt_result = optim(
                scaled_param0,
                objective_scaled;
                method = "BFGS",
            )
        end

        opt_par_scaled = opt_result.par
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


    likelihood = opt_result.value

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
        optim_return_code = opt_result.convergence,
        variance = variance,
        AIC = aic,
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

function filterSpecifics(
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

        return (AIC = Inf,)
    end


    first_model = fitSpecificBATS(
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
        non_seasonal_model = fitSpecificBATS(
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

        if first_model.AIC > non_seasonal_model.AIC
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


            second_model = fitSpecificBATS(
                y;
                use_box_cox = box_cox,
                use_beta = trend,
                use_damping = damping,
                seasonal_periods = best_seasonal_periods,
                ar_coefs = ar_coefs,
                ma_coefs = ma_coefs,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
                kwargs...,
            )

            if second_model.AIC < first_model.AIC
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

"""
    bats(y::AbstractVector{<:Real}, m::Union{Vector{Int},Nothing}=nothing;
         use_box_cox=nothing,
         use_trend=nothing,
         use_damped_trend=nothing,
         use_arma_errors=true,
         bc_lower=0.0,
         bc_upper=1.0,
         biasadj=false,
         model=nothing)

Fit a BATS model (Box-Cox transformation, ARMA errors, Trend and Seasonal
components) to the univariate series `y`, mirroring the behaviour of the
`forecast::bats` function described by De Livera, Hyndman & Snyder (2011).
When `model === nothing` the function automatically searches over Box-Cox,
trend and damping combinations (and optionally ARMA errors) selecting the
best model by AIC; if `model` is supplied the same structure is refit to `y`.

# Arguments
- `y`: real-valued vector representing the time series.
- `m`: optional vector of seasonal periods (ignored when all are 1); the
  `bats(y, m::Int)` method simply wraps this call with `[m]`.
- `use_box_cox`, `use_trend`, `use_damped_trend`: `Bool`, vector of `Bool`,
  or `nothing` to let the algorithm test both `true`/`false`.
- `use_arma_errors`: include ARMA(p, q) error structure via `auto_arima`.
- `bc_lower`, `bc_upper`: bounds for Box-Cox lambda search.
- `biasadj`: request bias-adjusted inverse Box-Cox transformation.
- `model`: result from a previous `bats` call; reuses its specification.
- `kwargs...`: forwarded to `auto_arima` when estimating ARMA errors.

# Returns
A [`BATSModel`](@ref) storing the fitted parameters, state vectors, original
series and a descriptive method label such as `BATS(λ, {p,q}, φ, {m…})`.

# References
- De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011). *Forecasting time
  series with complex seasonal patterns using exponential smoothing*.
  Journal of the American Statistical Association, 106(496), 1513‑1527.

# Examples
```julia
fit = bats(rand(120), 12)
fc = forecast(fit; h = 12)
```
"""
function bats(
    y::AbstractVector{<:Real},
    m::Union{Vector{Int},Nothing} = nothing;
    use_box_cox::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_damped_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_arma_errors::Bool = true,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
    biasadj::Bool = false,
    model = nothing,
)

    if ndims(y) != 1
        error("y should be a univariate time series (1D vector)")
    end

    orig_y = copy(y)
    orig_len = length(y)

    if m === nothing
        m = [1]
    end

    m = m[m.<length(y)]

    if isempty(m)
        m = [1]
    end

    m = unique(max.(m, 1))

    if all(m .== 1)
        m = nothing
    end

    y_contig = na_contiguous(y)
    if length(y_contig) != orig_len
        @warn "Missing values encountered. Using longest contiguous portion of time series"
    end
    y = y_contig

    if model !== nothing
        return fit_previous_bats_model(y, model)
    end

    if is_constant(y)
        @info "Series is constant. Returning trivial BATS model."
        m_const = create_constant_model(y)


        m_const.method = bats_descriptor(m_const)
        m_const.y = orig_y

        return m_const
    end

    if any(yi -> yi <= 0, y)
        use_box_cox = false
    end

    normalize_bool_vector(x) =
        x === nothing ? Bool[false, true] :
        x isa Bool ? Bool[x] :
        x isa AbstractVector{Bool} ? collect(x) :
        error("use_* arguments must be Bool, Vector{Bool}, or nothing")

    if use_box_cox === nothing
        use_box_cox = [false, true]
    end

    box_cox_values = normalize_bool_vector(use_box_cox)

    init_box_cox = nothing
    if any(box_cox_values)
        init_box_cox = box_cox_lambda(y, 1; lower = bc_lower, upper = bc_upper)
    end

    if use_trend === nothing
        use_trend = [false, true]
    elseif use_trend isa Bool && use_trend == false
        use_damped_trend = false
    end
    trend_values = normalize_bool_vector(use_trend)

    if use_damped_trend === nothing
        use_damped_trend = [false, true]
    end
    damping_values = normalize_bool_vector(use_damped_trend)

    y_num = Float64.(y)

    best_aic = Inf
    best_model = nothing

    model_count = 0
    for box_cox in box_cox_values
        for trend in trend_values
            for damping in damping_values
                model_count += 1
                current_model = try
                    filterSpecifics(
                        y_num,
                        box_cox = box_cox,
                        trend = trend,
                        damping = damping,
                        seasonal_periods = m,
                        use_arma_errors = use_arma_errors,
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                    )
                catch e
                    @warn "    Model failed: $e"
                    nothing
                end

                if current_model === nothing
                    continue
                end

                aic = getfield(current_model, :AIC)
                if aic < best_aic
                    best_aic = aic
                    best_model = current_model
                end
            end
        end
    end

    if best_model === nothing
        error("Unable to fit a model")
    end

    if hasproperty(best_model, :optim_return_code) &&
       getfield(best_model, :optim_return_code) != 0
        @warn "optim() did not converge."
    end

    method_label = bats_descriptor(
        best_model.lambda,
        best_model.ar_coefficients,
        best_model.ma_coefficients,
        best_model.damping_parameter,
        best_model.seasonal_periods,
    )


    result = BATSModel(
        best_model.lambda,
        best_model.alpha,
        best_model.beta,
        best_model.damping_parameter,
        best_model.gamma_values,
        best_model.ar_coefficients,
        best_model.ma_coefficients,
        best_model.seasonal_periods,
        best_model.fitted_values,
        best_model.errors,
        best_model.x,
        best_model.seed_states,
        best_model.variance,
        best_model.AIC,
        best_model.likelihood,
        best_model.optim_return_code,
        orig_y,
        Dict{Symbol,Any}(:vect => best_model.parameters.vect, :control => best_model.parameters.control),
        method_label,
    )

    return result
end

"""
    bats(y::AbstractVector{<:Real}, m::Int; kwargs...)

Convenience wrapper for the primary [`bats`](@ref) method when a single
seasonal period is supplied. Promotes `m` to a one-element vector and
forwards all keyword arguments so Box-Cox, trend/damping selection and
ARMA-error logic match the full interface.
"""
function bats(
    y::AbstractVector{<:Real},
    m::Int;
    use_box_cox::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_damped_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_arma_errors::Bool = true,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
    biasadj::Bool = false,
    model = nothing,
)
    return bats(
        y,
        [m],
        use_box_cox = use_box_cox,
        use_trend = use_trend,
        use_damped_trend = use_damped_trend,
        use_arma_errors = use_arma_errors,
        bc_lower = bc_lower,
        bc_upper = bc_upper,
        biasadj = biasadj,
        model = model,
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
        -Inf,
        -Inf,
        0,
        y,
        Dict{Symbol,Any}(),
        bats_descriptor(nothing, nothing, nothing, nothing, nothing),
    )
end

function forecast(
    model::BATSModel;
    h::Union{Int,Nothing} = nothing,
    level::Vector{Int} = [80, 95],
    fan::Bool = false,
    biasadj::Union{Bool,Nothing} = nothing,
)


    seasonal_periods = model.seasonal_periods
    ts_frequency =
        isnothing(seasonal_periods) || isempty(seasonal_periods) ? 1 :
        maximum(seasonal_periods)

    if h === nothing
        if isnothing(seasonal_periods) || isempty(seasonal_periods)
            h = (ts_frequency == 1) ? 10 : 2 * ts_frequency
        else
            h = 2 * maximum(seasonal_periods)
        end
    elseif h <= 0
        throw(ArgumentError("Forecast horizon out of bounds"))
    end

    if fan
        level = collect(51.0:3.0:99.0)
    else
        if minimum(level) > 0 && maximum(level) < 1

            level = 100.0 .* level
        elseif minimum(level) < 0 || maximum(level) > 99.99
            throw(ArgumentError("Confidence limit out of range"))
        end
    end
    n_levels = length(level)

    p = isnothing(model.ar_coefficients) ? 0 : length(model.ar_coefficients)
    q = isnothing(model.ma_coefficients) ? 0 : length(model.ma_coefficients)

    w = make_wmatrix(
        model.damping_parameter,
        model.seasonal_periods,
        model.ar_coefficients,
        model.ma_coefficients,
    )

    g_result = make_gmatrix(
        model.alpha,
        model.beta,
        model.gamma_values,
        model.seasonal_periods,
        p,
        q,
    )

    F = make_fmatrix(
        model.alpha,
        model.beta,
        model.damping_parameter,
        model.seasonal_periods,
        g_result.gamma_bold_matrix,
        model.ar_coefficients,
        model.ma_coefficients,
    )

    n_state = size(model.x, 1)
    x_states = zeros(Float64, n_state, h)
    y_forecast = zeros(Float64, h)

    x_last = model.x[:, end]


    y_forecast[1] = (w.w_transpose*x_last)[1]
    x_states[:, 1] = F * x_last


    if h > 1
        for t = 2:h
            x_states[:, t] = F * x_states[:, t-1]
            y_forecast[t] = (w.w_transpose*x_states[:, t-1])[1]
        end
    end

    variance_multiplier = ones(Float64, h)
    if h > 1
        f_running = Matrix{Float64}(I, n_state, n_state)
        for j = 1:(h-1)
            if j > 1
                f_running = f_running * F
            end

            c_j_vec = w.w_transpose * f_running * g_result.g
            c_j = c_j_vec[1]
            variance_multiplier[j+1] = variance_multiplier[j] + c_j^2
        end
    end

    variance = model.variance .* variance_multiplier
    stdev = sqrt.(variance)

    lower_bounds = Array{Float64}(undef, h, n_levels)
    upper_bounds = Array{Float64}(undef, h, n_levels)

    for (idx, lev) in enumerate(level)

        z = abs(quantile(Normal(), (100.0 - lev) / 200.0))
        marg_error = stdev .* z
        lower_bounds[:, idx] = y_forecast .- marg_error
        upper_bounds[:, idx] = y_forecast .+ marg_error
    end

    y_fc_out = copy(y_forecast)
    lb_out = copy(lower_bounds)
    ub_out = copy(upper_bounds)

    if !isnothing(model.lambda)
        λ = model.lambda

        if biasadj === nothing
            y_fc_out = inv_box_cox(y_forecast; lambda=λ, fvar=variance)
        else
            y_fc_out = inv_box_cox(y_forecast; lambda=λ, biasadj=biasadj, fvar=variance)
        end

        lb_out = inv_box_cox(lower_bounds; lambda=λ)
        ub_out = inv_box_cox(upper_bounds; lambda=λ)

        if λ < 1
            lb_out = max.(lb_out, 0.0)
        end
    end

    stored_method = hasproperty(model, :method) ? String(model.method) : ""
    method = isempty(stored_method) ? bats_descriptor(model) : stored_method
    x_series = getproperty(model, :y)
    fitted = getproperty(model, :fitted_values)
    residuals = getproperty(model, :errors)

    forecast_obj = Forecast(
        model,
        method,
        y_fc_out,
        level,
        x_series,
        ub_out,
        lb_out,
        fitted,
        residuals,
    )

    return forecast_obj
end

function fitted(model::BATSModel)
    return model.fitted_values
end

function Base.show(io::IO, model::BATSModel)
    println(io, model.method)
    println(io, "")
    println(io, "Parameters:")

    !isnothing(model.lambda) && println(io, "  Lambda:  ", round(model.lambda, digits = 4))
    println(io, "  Alpha:   ", round(model.alpha, digits = 4))

    if !isnothing(model.beta)
        println(io, "  Beta:    ", round(model.beta, digits = 4))
        damping = isnothing(model.damping_parameter) ? 1.0 : model.damping_parameter
        println(io, "  Damping: ", round(damping, digits = 4))
    end

    if !isnothing(model.gamma_values)
        println(
            io,
            "  Gamma:   ",
            join([round(g, digits = 4) for g in model.gamma_values], ", "),
        )
    end

    if !isnothing(model.ar_coefficients)
        println(
            io,
            "  AR:      ",
            join([round(φ, digits = 4) for φ in model.ar_coefficients], ", "),
        )
    end

    if !isnothing(model.ma_coefficients)
        println(
            io,
            "  MA:      ",
            join([round(θ, digits = 4) for θ in model.ma_coefficients], ", "),
        )
    end

    !isnothing(model.seasonal_periods) &&
        println(io, "  Seasonal periods: ", model.seasonal_periods)

    println(io, "")
    println(io, "Sigma:   ", round(sqrt(model.variance), digits = 4))
    println(io, "AIC:     ", round(model.AIC, digits = 2))
end
