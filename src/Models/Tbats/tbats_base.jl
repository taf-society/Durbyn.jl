"""
    TBATSModel

Container for a fitted TBATS model (Box-Cox transformation, ARMA errors,
trend, and multiple seasonal components via Fourier terms). Mirrors the
`forecast::tbats` object: fields store smoothing parameters, ARMA
coefficients, state matrices (`x`/`seed_states`), fitted values, errors,
likelihood, information criteria, and metadata needed to forecast without
refitting. The descriptor `TBATS(omega, {p,q}, phi, <m1,k1>,...,<mJ,kJ>)`
matches the R output, where `omega` is the Box-Cox lambda, `{p,q}` the ARMA
orders, `phi` the damping parameter, and `<m,k>` pairs define seasonal
periods and Fourier orders.

# References
- De Livera, A.M., Hyndman, R.J., & Snyder, R.D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. Journal of the American Statistical Association, 106(496), 1513-1527.
"""
mutable struct TBATSModel
    lambda::Union{Float64,Nothing}
    alpha::Float64
    beta::Union{Float64,Nothing}
    damping_parameter::Union{Float64,Nothing}
    gamma_one_values::Union{Vector{Float64},Nothing}
    gamma_two_values::Union{Vector{Float64},Nothing}
    ar_coefficients::Union{Vector{Float64},Nothing}
    ma_coefficients::Union{Vector{Float64},Nothing}
    seasonal_periods::Union{Vector{<:Real},Nothing}
    k_vector::Union{Vector{Int},Nothing}
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
    biasadj::Bool
end

function tbats_descriptor(
    lambda::Union{Float64,Nothing},
    ar_coefficients::Union{Vector{Float64},Nothing},
    ma_coefficients::Union{Vector{Float64},Nothing},
    damping_parameter::Union{Float64,Nothing},
    seasonal_periods::Union{Vector{<:Real},Nothing},
    k_vector::Union{Vector{Int},Nothing},
)
    lambda_str = isnothing(lambda) ? "1" : string(round(lambda, digits = 3))
    ar_count = isnothing(ar_coefficients) ? 0 : length(ar_coefficients)
    ma_count = isnothing(ma_coefficients) ? 0 : length(ma_coefficients)
    damping_str = isnothing(damping_parameter) ? "-" : string(round(damping_parameter, digits = 3))

    buffer = IOBuffer()
    print(buffer, "TBATS(", lambda_str, ", {", ar_count, ",", ma_count, "}, ", damping_str, ", ")

    if isnothing(seasonal_periods) || isempty(seasonal_periods)
        print(buffer, "{-})")
    else
        print(buffer, "{")
        for (i, (m, k)) in enumerate(zip(seasonal_periods, k_vector))
            print(buffer, "<", m, ",", k, ">")
            if i < length(seasonal_periods)
                print(buffer, ", ")
            end
        end
        print(buffer, "})")
    end

    return String(take!(buffer))
end

tbats_descriptor(model::TBATSModel) = tbats_descriptor(
    model.lambda,
    model.ar_coefficients,
    model.ma_coefficients,
    model.damping_parameter,
    model.seasonal_periods,
    model.k_vector,
)

Base.string(model::TBATSModel) = tbats_descriptor(model)

struct TBATSParameterControl
    use_box_cox::Bool
    use_beta::Bool
    use_damping::Bool
    length_gamma::Int
    p::Int
    q::Int
end

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

function check_admissibility_tbats(
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

function calc_model_tbats(
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


function fitSpecificTBATS(
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
    bc_lower::Float64 = 0.0,
    bc_upper::Float64 = 1.0,
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
            scaled_param0,
            objective_scaled;
            method = "Nelder-Mead",
            control = Dict("maxit" => maxit),
        )

        opt_par_scaled = opt_result.par
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
                scaled_param0,
                objective_scaled;
                method = "Nelder-Mead",
                control = Dict("maxit" => maxit),
            )
        else
            opt_result = optimize(
                scaled_param0,
                objective_scaled;
                method = "BFGS",
            )
        end

        opt_par_scaled = opt_result.par
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

    likelihood = opt_result.value
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
        optim_return_code = opt_result.convergence,
        variance = variance,
        AIC = aic,
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
    bc_lower::Float64 = 0.0,
    bc_upper::Float64 = 1.0,
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


function filterTBATSSpecifics(
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
            fitSpecificTBATS(
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
            @warn "fitSpecificTBATS in filterTBATSSpecifics failed: $e"
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
        @warn "auto_arima in filterTBATSSpecifics failed: $e"
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
        fitSpecificTBATS(
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
        @warn "fitSpecificTBATS with ARMA in filterTBATSSpecifics failed: $e"
        nothing
    end

    aic_first = first_model.AIC
    aic_second = second_model === nothing ? Inf : second_model.AIC

    if aic_second < aic_first
        return second_model
    else
        return first_model
    end
end

function fitPreviousTBATSModel(y::Vector{Float64}; model::TBATSModel)
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

"""
    tbats(y, m; use_box_cox=nothing, use_trend=nothing, use_damped_trend=nothing,
          use_arma_errors=true, bc_lower=0.0, bc_upper=1.0, biasadj=false,
          model=nothing)

Fit a TBATS model (Exponential smoothing state space with Box-Cox
transformation, ARMA errors, trend, and trigonometric seasonality) to a
univariate time series. This is a Julia port of `forecast::tbats`
(`De Livera, Hyndman & Snyder 2011`); it searches over Box-Cox, trend, and
damped-trend options, optimizes Fourier orders per seasonal period, and can
optionally refit a supplied TBATS/BATS model.

# Arguments
- `y`: Univariate series to model.
- `m`: Seasonal periods; pass `nothing` to infer nonseasonal.
- `use_box_cox`: Bool or vector of Bools; if `nothing`, both FALSE/TRUE are tried and chosen by AIC.
- `use_trend`: Bool or vector; if `nothing`, both are tried and chosen by AIC.
- `use_damped_trend`: Bool or vector; if `nothing`, both are tried and chosen by AIC (ignored when trend is FALSE).
- `use_arma_errors`: Whether to fit ARMA errors (orders selected by `auto_arima` on residuals).
- `bc_lower`/`bc_upper`: Bounds for Box-Cox lambda search.
- `biasadj`: Use bias-adjusted inverse Box-Cox for fitted/forecast means.
- `model`: Previous TBATS/BATS fit to refit without re-estimating parameters.
- `...`: Extra keywords forwarded to `auto_arima` when selecting ARMA(p,q).

# Returns
A `TBATSModel` (or `BATSModel` when no seasonality) storing parameters,
states, fitted values, residuals, variance, likelihood, AIC, and the
descriptor `TBATS(omega, {p,q}, phi, <m1,k1>,...,<mJ,kJ>)`.

# References
- De Livera, A.M., Hyndman, R.J., & Snyder, R.D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. Journal of the American Statistical Association, 106(496), 1513-1527.
"""
function tbats(
    y::AbstractVector{<:Real},
    m::Union{Vector{<:Real},Nothing} = nothing;
    use_box_cox::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_damped_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_arma_errors::Bool = true,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
    biasadj::Bool = false,
    model = nothing,
    k::Union{Vector{Int},Int,Nothing} = nothing,
    kwargs...,
)

    if ndims(y) != 1
        error("y should be a univariate time series (1D vector)")
    end

    orig_y = copy(y)
    orig_len = length(y)

    seasonal_periods = if m === nothing
        [1]
    else
        copy(m)
    end

    y_contig = na_contiguous(y)
    if length(y_contig) != orig_len
        @warn "Missing values encountered. Using longest contiguous portion of time series"
    end
    y = y_contig

    seasonal_periods = seasonal_periods[seasonal_periods .< length(y)]
    if isempty(seasonal_periods)
        seasonal_periods = [1]
    end
    seasonal_periods = unique(max.(seasonal_periods, 1))
    if all(seasonal_periods .== 1)
        seasonal_periods = nothing
    end

    if model !== nothing
        if model isa TBATSModel
            result = fitPreviousTBATSModel(collect(Float64, y); model = model)
            result.y = collect(Float64, orig_y)
            result.method = tbats_descriptor(result)
            return result
        elseif model isa BATSModel
            return bats(orig_y; model = model)
        else
            error("Unsupported model type for refit in tbats")
        end
    end

    if is_constant(y)
        m_const = create_constant_tbats_model(collect(Float64, y))
        m_const.y = collect(Float64, orig_y)
        m_const.method = tbats_descriptor(m_const)
        return m_const
    end

    if any(yi -> yi <= 0, y)
        if use_box_cox === true || (use_box_cox isa AbstractVector && any(use_box_cox))
            @warn "Series contains non-positive values. Box-Cox transformation disabled."
        end
        use_box_cox = false
    end

    normalize_bool_vector(x) =
        x === nothing ? Bool[false, true] :
        x isa Bool ? Bool[x] :
        x isa AbstractVector{Bool} ? collect(x) :
        error("use_* arguments must be Bool, Vector{Bool}, or nothing")

    box_cox_values = normalize_bool_vector(use_box_cox)
    if any(box_cox_values)
        bc_period = (seasonal_periods === nothing || isempty(seasonal_periods)) ? 1 : round(Int, first(seasonal_periods))
        init_box_cox = box_cox_lambda(y, bc_period; lower = bc_lower, upper = bc_upper)
    else
        init_box_cox = nothing
    end

    trend_values = begin
        if use_trend === nothing
            use_trend = [false, true]
        elseif use_trend isa Bool && use_trend == false
            use_damped_trend = false
        end
        normalize_bool_vector(use_trend)
    end

    damping_values = begin
        if use_damped_trend === nothing
            use_damped_trend = [false, true]
        end
        normalize_bool_vector(use_damped_trend)
    end

    model_params = Bool[any(box_cox_values), any(trend_values), any(damping_values)]

    y_num = Float64.(y)
    n = length(y_num)

    nonseasonal_model = bats(
        y_num;
        use_box_cox = use_box_cox,
        use_trend = use_trend,
        use_damped_trend = use_damped_trend,
        use_arma_errors = use_arma_errors,
        bc_lower = bc_lower,
        bc_upper = bc_upper,
        biasadj = biasadj,
    )

    if seasonal_periods === nothing
        nonseasonal_model.y = orig_y
        return nonseasonal_model
    else
        mask = seasonal_periods .== 1
        seasonal_periods = seasonal_periods[.!mask]
    end

    user_k = k
    if user_k !== nothing
        if user_k isa Int
            user_k = fill(user_k, length(seasonal_periods))
        end
        if length(user_k) != length(seasonal_periods)
            throw(ArgumentError("k must have the same length as seasonal_periods (got $(length(user_k)) vs $(length(seasonal_periods)))"))
        end
        if any(ki -> ki < 1, user_k)
            throw(ArgumentError("k values must be positive integers"))
        end
        for (i, period) in enumerate(seasonal_periods)
            max_k = floor(Int, (period - 1) / 2)
            if user_k[i] > max_k
                throw(ArgumentError("k[$(i)]=$(user_k[i]) exceeds max Fourier order $(max_k) for seasonal period $(period)"))
            end
        end
        k_vector = collect(Int, user_k)
    else
        k_vector = ones(Int, length(seasonal_periods))
    end

    function safe_fitSpecificTBATS(
        y_num;
        use_box_cox::Bool,
        use_beta::Bool,
        use_damping::Bool,
        seasonal_periods::Vector{<:Real},
        k_vector::Vector{Int},
        init_box_cox,
        bc_lower,
        bc_upper,
        biasadj,
    )
        try
            return fitSpecificTBATS(
                y_num;
                use_box_cox = use_box_cox,
                use_beta = use_beta,
                use_damping = use_damping,
                seasonal_periods = seasonal_periods,
                k_vector = k_vector,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
            )
        catch e
            @warn "fitSpecificTBATS failed: $e"
            return nothing
        end
    end

    best_model = safe_fitSpecificTBATS(
        y_num;
        use_box_cox = model_params[1],
        use_beta = model_params[2],
        use_damping = model_params[3],
        seasonal_periods = seasonal_periods,
        k_vector = k_vector,
        init_box_cox = init_box_cox,
        bc_lower = bc_lower,
        bc_upper = bc_upper,
        biasadj = biasadj,
    )

    best_aic = best_model === nothing ? Inf : getfield(best_model, :AIC)

    if user_k !== nothing
        # User provided k — skip k-search, go straight to model selection
        @goto k_search_done
    end

    for (i, period) in enumerate(seasonal_periods)
        if period == 2
            continue
        end

        max_k = floor(Int, (period - 1) / 2)

        if i != 1
            current_k = 2
            while current_k <= max_k
                if period % current_k != 0
                    current_k += 1
                    continue
                end
                latter = period / current_k
                if any(seasonal_periods[1:i-1] .% latter .== 0)
                    max_k = current_k - 1
                    break
                else
                    current_k += 1
                end
            end
        end

        if max_k == 1
            continue
        end

        if max_k <= 6
            k_vector[i] = max_k
            local_best_model = best_model
            local_best_aic = Inf

            while true
                new_model = safe_fitSpecificTBATS(
                    y_num;
                    use_box_cox = model_params[1],
                    use_beta = model_params[2],
                    use_damping = model_params[3],
                    seasonal_periods = seasonal_periods,
                    k_vector = k_vector,
                    init_box_cox = init_box_cox,
                    bc_lower = bc_lower,
                    bc_upper = bc_upper,
                    biasadj = biasadj,
                )

                new_aic = new_model === nothing ? Inf : getfield(new_model, :AIC)

                if new_aic > local_best_aic
                    k_vector[i] += 1
                    break
                else
                    if k_vector[i] == 1
                        local_best_model =
                            new_model === nothing ? local_best_model : new_model
                        local_best_aic = min(local_best_aic, new_aic)
                        break
                    end
                    k_vector[i] -= 1
                    local_best_model = new_model === nothing ? local_best_model : new_model
                    local_best_aic = min(local_best_aic, new_aic)
                end
            end

            if local_best_aic < best_aic
                best_aic = local_best_aic
                best_model = local_best_model
            end

        else
            step_up_k = copy(k_vector)
            step_down_k = copy(k_vector)
            step_up_k[i] = 7
            step_down_k[i] = 5
            k_vector[i] = 6

            up_model = safe_fitSpecificTBATS(
                y_num;
                use_box_cox = model_params[1],
                use_beta = model_params[2],
                use_damping = model_params[3],
                seasonal_periods = seasonal_periods,
                k_vector = step_up_k,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
            )
            level_model = safe_fitSpecificTBATS(
                y_num;
                use_box_cox = model_params[1],
                use_beta = model_params[2],
                use_damping = model_params[3],
                seasonal_periods = seasonal_periods,
                k_vector = k_vector,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
            )
            down_model = safe_fitSpecificTBATS(
                y_num;
                use_box_cox = model_params[1],
                use_beta = model_params[2],
                use_damping = model_params[3],
                seasonal_periods = seasonal_periods,
                k_vector = step_down_k,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
            )

            a_up = up_model === nothing ? Inf : getfield(up_model, :AIC)
            a_level = level_model === nothing ? Inf : getfield(level_model, :AIC)
            a_down = down_model === nothing ? Inf : getfield(down_model, :AIC)

            if a_down <= a_up && a_down <= a_level
                best_local = down_model
                best_local_aic = a_down
                k_vector[i] = 5

                while true
                    k_vector[i] -= 1
                    new_model = safe_fitSpecificTBATS(
                        y_num;
                        use_box_cox = model_params[1],
                        use_beta = model_params[2],
                        use_damping = model_params[3],
                        seasonal_periods = seasonal_periods,
                        k_vector = k_vector,
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                    )
                    new_aic = new_model === nothing ? Inf : getfield(new_model, :AIC)

                    if new_aic > best_local_aic
                        k_vector[i] += 1
                        break
                    else
                        best_local = new_model === nothing ? best_local : new_model
                        best_local_aic = min(best_local_aic, new_aic)
                    end
                    if k_vector[i] == 1
                        break
                    end
                end

                if best_local_aic < best_aic
                    best_aic = best_local_aic
                    best_model = best_local
                end

            elseif a_level <= a_up && a_level <= a_down
                if a_level < best_aic
                    best_aic = a_level
                    best_model = level_model
                end

            else
                best_local = up_model
                best_local_aic = a_up
                k_vector[i] = 7

                while true
                    k_vector[i] += 1
                    new_model = safe_fitSpecificTBATS(
                        y_num;
                        use_box_cox = model_params[1],
                        use_beta = model_params[2],
                        use_damping = model_params[3],
                        seasonal_periods = seasonal_periods,
                        k_vector = k_vector,
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                    )
                    new_aic = new_model === nothing ? Inf : getfield(new_model, :AIC)

                    if new_aic > best_local_aic
                        k_vector[i] -= 1
                        break
                    else
                        best_local = new_model === nothing ? best_local : new_model
                        best_local_aic = min(best_local_aic, new_aic)
                    end
                    if k_vector[i] == max_k
                        break
                    end
                end

                if best_local_aic < best_aic
                    best_aic = best_local_aic
                    best_model = best_local
                end
            end
        end
    end

    @label k_search_done

    aux_model = best_model

    if nonseasonal_model.AIC < (best_model === nothing ? Inf : best_model.AIC)
        best_model = nonseasonal_model
    end

    for box_cox in box_cox_values
        for trend in trend_values
            for damping in damping_values
                if !trend && damping
                    continue
                end

                if model_params == Bool[box_cox, trend, damping]
                    new_model = filterTBATSSpecifics(
                        y_num,
                        box_cox,
                        trend,
                        damping,
                        seasonal_periods,
                        k_vector,
                        use_arma_errors;
                        aux_model = aux_model,
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                        kwargs...,
                    )
                elseif trend || !damping
                    new_model = filterTBATSSpecifics(
                        y_num,
                        box_cox,
                        trend,
                        damping,
                        seasonal_periods,
                        k_vector,
                        use_arma_errors;
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                        kwargs...,
                    )
                else
                    continue
                end

                if new_model === nothing
                    continue
                end

                if best_model === nothing || new_model.AIC < best_model.AIC
                    best_model = new_model
                end
            end
        end
    end

    if hasproperty(best_model, :optim_return_code) &&
       getfield(best_model, :optim_return_code) != 0
        @warn "optimize() did not converge."
    end

    if best_model === nothing
        error("Failed to fit any TBATS/BATS model")
    end

    if best_model isa BATSModel
        return best_model
    end

    method_label = tbats_descriptor(
        best_model.lambda,
        best_model.ar_coefficients,
        best_model.ma_coefficients,
        best_model.damping_parameter,
        best_model.seasonal_periods,
        best_model.k_vector,
    )

    result = TBATSModel(
        best_model.lambda,
        best_model.alpha,
        best_model.beta,
        best_model.damping_parameter,
        best_model.gamma_one_values,
        best_model.gamma_two_values,
        best_model.ar_coefficients,
        best_model.ma_coefficients,
        best_model.seasonal_periods,
        best_model.k_vector,
        best_model.fitted_values,
        best_model.errors,
        best_model.x,
        best_model.seed_states,
        best_model.variance,
        best_model.AIC,
        best_model.likelihood,
        best_model.optim_return_code,
        orig_y,
        Dict{Symbol,Any}(
            :vect => best_model.parameters.vect,
            :control => best_model.parameters.control,
        ),
        method_label,
        biasadj,
    )

    return result
end



"""
    tbats(y::AbstractVector, m::Real; kwargs...)

Convenience wrapper for `tbats` when a single seasonal period is supplied as
a scalar. It forwards all keyword arguments to the primary `tbats` method.
"""
function tbats(
    y::AbstractVector{<:Real},
    m::Real;
    use_box_cox::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_damped_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_arma_errors::Bool = true,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
    biasadj::Bool = false,
    model = nothing,
    kwargs...,
)
    return tbats(
        y,
        [m];
        use_box_cox = use_box_cox,
        use_trend = use_trend,
        use_damped_trend = use_damped_trend,
        use_arma_errors = use_arma_errors,
        bc_lower = bc_lower,
        bc_upper = bc_upper,
        biasadj = biasadj,
        model = model,
        kwargs...,
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
        -Inf,
        -Inf,
        0,
        y,
        Dict{Symbol,Any}(),
        tbats_descriptor(nothing, nothing, nothing, nothing, nothing, nothing),
        false,
    )
end

function forecast(
    model::TBATSModel;
    h::Union{Int,Nothing} = nothing,
    level::AbstractVector{<:Real} = [80, 95],
    fan::Bool = false,
    biasadj::Union{Bool,Nothing} = nothing,
)
    seasonal_periods = model.seasonal_periods
    ts_frequency =
        isnothing(seasonal_periods) || isempty(seasonal_periods) ? 1 :
        maximum(seasonal_periods)

    if h === nothing
        if isnothing(seasonal_periods) || isempty(seasonal_periods)
            h = (ts_frequency == 1) ? 10 : round(Int, 2 * ts_frequency)
        else
            h = round(Int, 2 * maximum(seasonal_periods))
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
    tau = isnothing(model.k_vector) ? 0 : 2 * sum(model.k_vector)

    w = make_tbats_wmatrix(
        model.damping_parameter,
        model.k_vector,
        model.ar_coefficients,
        model.ma_coefficients,
        tau,
    )

    g_result = make_tbats_gmatrix(
        model.alpha,
        model.beta,
        model.gamma_one_values,
        model.gamma_two_values,
        model.k_vector,
        p,
        q,
    )

    F = make_tbats_fmatrix(
        model.alpha,
        model.beta,
        model.damping_parameter,
        model.seasonal_periods,
        model.k_vector,
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
        ba = biasadj === nothing ? model.biasadj : biasadj
        y_fc_out = inv_box_cox(y_forecast; lambda=λ, biasadj=ba, fvar=variance)

        lb_out = inv_box_cox(lower_bounds; lambda=λ)
        ub_out = inv_box_cox(upper_bounds; lambda=λ)

        if λ < 1
            lb_out = max.(lb_out, 0.0)
        end
    end

    stored_method = hasproperty(model, :method) ? String(model.method) : ""
    method = isempty(stored_method) ? tbats_descriptor(model) : stored_method
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

function fitted(model::TBATSModel)
    return model.fitted_values
end

function residuals(model::TBATSModel)
    return model.errors
end

function Base.show(io::IO, model::TBATSModel)
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

    if !isnothing(model.gamma_one_values)
        println(
            io,
            "  Gamma-1: ",
            join([round(g, digits = 4) for g in model.gamma_one_values], ", "),
        )
    end

    if !isnothing(model.gamma_two_values)
        println(
            io,
            "  Gamma-2: ",
            join([round(g, digits = 4) for g in model.gamma_two_values], ", "),
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

    if !isnothing(model.seasonal_periods) && !isnothing(model.k_vector)
        println(io, "  Seasonal periods: ", model.seasonal_periods)
        println(io, "  Fourier terms (k): ", model.k_vector)
    end

    println(io, "")
    println(io, "Sigma:   ", round(sqrt(model.variance), digits = 4))
    println(io, "AIC:     ", round(model.AIC, digits = 2))
end
