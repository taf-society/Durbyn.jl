"""
    TBATSModel

Container for fitted TBATS models (Trigonometric seasonality, Box-Cox transformation,
ARMA errors, Trend and seasonal components) following the definition in
De Livera, Hyndman & Snyder (2011) and the `forecast::tbats` R implementation.

The key difference from BATS is that TBATS uses Fourier terms (trigonometric
seasonal representation) instead of seasonal dummies, making it more efficient
for high-frequency seasonal data.

Fields store Box-Cox lambda, level/trend coefficients, Fourier smoothing parameters
(gamma.one.values and gamma.two.values for sine and cosine components), k.vector
(number of Fourier pairs for each seasonal period), ARMA coefficients, state matrices,
diagnostics and metadata.
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
    seasonal_periods::Union{Vector{Int},Nothing}
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
end

function tbats_descriptor(
    lambda::Union{Float64,Nothing},
    ar_coefficients::Union{Vector{Float64},Nothing},
    ma_coefficients::Union{Vector{Float64},Nothing},
    damping_parameter::Union{Float64,Nothing},
    seasonal_periods::Union{Vector{Int},Nothing},
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
    length_gamma::Int  # Total number of gamma parameters (2 * num_seasonal_periods)
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
    seasonal_periods::Union{Vector{Int},Nothing},
    k_vector::Union{Vector{Int},Nothing},
    gamma_bold_matrix::Union{Matrix{Float64},Nothing},
    ar_coefs::Union{Vector{Float64},Nothing},
    ma_coefs::Union{Vector{Float64},Nothing},
)
    F = ones(1, 1)
    !isnothing(beta) && (F = hcat(F, small_phi))

    if !isnothing(seasonal_periods) && !isnothing(k_vector)
        tau = 2 * sum(k_vector)
        F = hcat(F, zeros(1, tau))
    end

    p = isnothing(ar_coefs) ? 0 : length(ar_coefs)
    q = isnothing(ma_coefs) ? 0 : length(ma_coefs)

    p > 0 && (F = hcat(F, alpha .* ar_coefs'))
    q > 0 && (F = hcat(F, alpha .* ma_coefs'))

    if !isnothing(beta)
        beta_row = [0.0 small_phi]
        if !isnothing(seasonal_periods) && !isnothing(k_vector)
            tau = 2 * sum(k_vector)
            beta_row = hcat(beta_row, zeros(1, tau))
        end
        p > 0 && (beta_row = hcat(beta_row, beta .* ar_coefs'))
        q > 0 && (beta_row = hcat(beta_row, beta .* ma_coefs'))
        F = vcat(F, beta_row)
    end

    if !isnothing(seasonal_periods) && !isnothing(k_vector)
        tau = 2 * sum(k_vector)
        seasonal_rows = zeros(tau, 1)
        !isnothing(beta) && (seasonal_rows = hcat(seasonal_rows, zeros(tau, 1)))

        A = zeros(tau, tau)
        pos = 1
        for (i, (m, k)) in enumerate(zip(seasonal_periods, k_vector))
            if m == 2
                Ci = zeros(1, 1)
                Si = make_si_matrix(k, Float64(m))
                Ai = make_ai_matrix(Ci, Si, k)
            else
                Ci = make_ci_matrix(k, Float64(m))
                Si = make_si_matrix(k, Float64(m))
                Ai = make_ai_matrix(Ci, Si, k)
            end

            block_size = 2k
            A[pos:(pos+block_size-1), pos:(pos+block_size-1)] = Ai
            pos += block_size
        end

        seasonal_rows = hcat(seasonal_rows, A)

        p > 0 && (seasonal_rows = hcat(seasonal_rows, gamma_bold_matrix' * ar_coefs))
        q > 0 && (seasonal_rows = hcat(seasonal_rows, gamma_bold_matrix' * ma_coefs))

        F = vcat(F, seasonal_rows)
    end

    if p > 0
        ar_rows = zeros(p, 1)
        !isnothing(beta) && (ar_rows = hcat(ar_rows, zeros(p, 1)))
        if !isnothing(seasonal_periods) && !isnothing(k_vector)
            tau = 2 * sum(k_vector)
            ar_rows = hcat(ar_rows, zeros(p, tau))
        end

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
        if !isnothing(seasonal_periods) && !isnothing(k_vector)
            tau = 2 * sum(k_vector)
            ma_rows = hcat(ma_rows, zeros(q, tau))
        end
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


function make_tbats_wmatrix(
    small_phi::Union{Float64,Nothing},
    k_vector::Union{Vector{Int},Nothing},
    ar_coefs::Union{Vector{Float64},Nothing},
    ma_coefs::Union{Vector{Float64},Nothing},
    tau::Int,
)
    w_transpose = [1.0]
    !isnothing(small_phi) && push!(w_transpose, small_phi)

    if !isnothing(k_vector) && tau > 0
        for k in k_vector
            append!(w_transpose, ones(k))
            append!(w_transpose, zeros(k))
        end
    end

    !isnothing(ar_coefs) && append!(w_transpose, ar_coefs)
    !isnothing(ma_coefs) && append!(w_transpose, ma_coefs)

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
        idx = findall(x -> abs(x) > EPS, ar_coefs)
        if !isempty(idx)
            p = maximum(idx)
            coeffs = [1.0; -ar_coefs[1:p]]
            rts = roots(Polynomial(coeffs))
            minimum(abs.(rts)) < RAD && return false
        end
    end

    if ma_coefs !== nothing
        idx = findall(x -> abs(x) > EPS, ma_coefs)
        if !isempty(idx)
            q = maximum(idx)
            coeffs = [1.0; ma_coefs[1:q]]
            rts = roots(Polynomial(coeffs))
            minimum(abs.(rts)) < RAD && return false
        end
    end

    vals = eigvals(Matrix{Float64}(D))
    all(abs.(vals) .< RAD) || return false

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
    seasonal_periods::Union{Vector{Int},Nothing} = nothing,
    k_vector::Union{Vector{Int},Nothing} = nothing,
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
        k_vector = Int.(k_vector)
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
                lambda = box_cox_lambda(y, 1; lower = 0.0, upper = 1.5)
            end
            y_transformed, lambda = box_cox(y, 1; lambda=lambda)
        else
            lambda = nothing
        end
    else
        paramz = unparameterise_tbats(starting_params.vect, starting_params.control)
        lambda = paramz.lambda
        alpha = paramz.alpha
        beta_v = paramz.beta
        b = 0.0
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

    # Cut w for seed state estimation
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
            NelderMead(),
            Optim.Options(iterations = maxit),
        )

        opt_par_scaled = Optim.minimizer(opt_result)
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
                NelderMead(),
                Optim.Options(iterations = maxit),
            )
        else
            opt_result = optimize(objective_scaled, scaled_param0, BFGS())
        end

        opt_par_scaled = Optim.minimizer(opt_result)
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

    likelihood = Optim.minimum(opt_result)
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
        optim_return_code = Optim.converged(opt_result) ? 0 : 1,
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
    seasonal_periods::Union{Vector{Int},Nothing},
    k_vector::Union{Vector{Int},Nothing},
    p::Int = 0,
    q::Int = 0,
    tau::Int = 0,
    bc_lower::Float64 = 0.0,
    bc_upper::Float64 = 1.0,
)
    control = TBATSParameterControl(
        true,  # use_box_cox
        use_beta,
        use_small_phi,
        isnothing(k_vector) ? 0 : 2 * length(k_vector),  # length_gamma = 2 * num_seasonal_periods
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

    x_nought_vec, lambda = box_cox(vec(opt_env[:x_nought_untransformed]), 1; lambda=box_cox_parameter)
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

    opt_env[:w_transpose] = w.w_transpose
    opt_env[:g] = reshape(g_result.g, :, 1)
    opt_env[:gamma_bold_matrix] = g_result.gamma_bold_matrix
    opt_env[:F] = F

    y_transformed_vec, lambda = box_cox(vec(opt_env[:y]), 1; lambda=box_cox_parameter)
    n = size(opt_env[:y], 2)
    mat_transformed_y = reshape(y_transformed_vec, 1, n)

    # Simplified state space calculation
    for t = 1:n
        if t == 1
            opt_env[:y_hat][:, t] = opt_env[:w_transpose] * x_nought[:, 1]
            opt_env[:e][:, t] = mat_transformed_y[:, t] - opt_env[:y_hat][:, t]
            opt_env[:x][:, t] = F * x_nought[:, 1] + opt_env[:g] * opt_env[:e][1, t]
        else
            opt_env[:y_hat][:, t] = opt_env[:w_transpose] * opt_env[:x][:, t-1]
            opt_env[:e][:, t] = mat_transformed_y[:, t] - opt_env[:y_hat][:, t]
            opt_env[:x][:, t] = F * opt_env[:x][:, t-1] + opt_env[:g] * opt_env[:e][1, t]
        end
    end

    log_likelihood = n * log(sum(opt_env[:e] .^ 2)) - 2 * (box_cox_parameter - 1) * sum(log.(opt_env[:y]))

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
    seasonal_periods::Union{Vector{Int},Nothing},
    k_vector::Union{Vector{Int},Nothing},
    p::Int = 0,
    q::Int = 0,
    tau::Int = 0,
)
    control = TBATSParameterControl(
        false,  # use_box_cox
        use_beta,
        use_small_phi,
        isnothing(k_vector) ? 0 : 2 * length(k_vector),  # length_gamma = 2 * num_seasonal_periods
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

    # Simplified state space calculation
    for t = 1:n
        if t == 1
            opt_env[:y_hat][:, t] = opt_env[:w_transpose] * x_nought[:, 1]
            opt_env[:e][:, t] = opt_env[:y][:, t] - opt_env[:y_hat][:, t]
            opt_env[:x][:, t] = F * x_nought[:, 1] + opt_env[:g] * opt_env[:e][1, t]
        else
            opt_env[:y_hat][:, t] = opt_env[:w_transpose] * opt_env[:x][:, t-1]
            opt_env[:e][:, t] = opt_env[:y][:, t] - opt_env[:y_hat][:, t]
            opt_env[:x][:, t] = F * opt_env[:x][:, t-1] + opt_env[:g] * opt_env[:e][1, t]
        end
    end

    log_likelihood = n * log(sum(opt_env[:e] .* opt_env[:e]))

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

"""
    tbats(y::AbstractVector{<:Real}, m::Union{Vector{Int},Nothing}=nothing; kwargs...)

Fit a TBATS model (Trigonometric seasonality, Box-Cox transformation, ARMA errors,
Trend and seasonal components) to the univariate series `y`, mirroring the behaviour
of the `forecast::tbats` R function.

TBATS uses Fourier terms to represent seasonality, making it more efficient than BATS
for high-frequency seasonal data.

# Arguments
- `y`: real-valued vector representing the time series.
- `m`: optional vector of seasonal periods.
- `use_box_cox`, `use_trend`, `use_damped_trend`: `Bool`, vector of `Bool`, or `nothing`.
- `use_arma_errors`: include ARMA(p, q) error structure via `auto_arima`.
- `bc_lower`, `bc_upper`: bounds for Box-Cox lambda search.
- `biasadj`: request bias-adjusted inverse Box-Cox transformation.
- `model`: result from a previous `tbats` call; reuses its specification.

# Returns
A [`TBATSModel`](@ref) storing the fitted parameters, state vectors, original
series and a descriptive method label such as `TBATS(λ, {p,q}, φ, {<m,k>…})`.

# Examples
```julia
fit = tbats(rand(120), [12])
fc = forecast(fit; h = 12)
```
"""
function tbats(
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

    if is_constant(y)
        @info "Series is constant. Returning trivial TBATS model."
        m_const = create_constant_tbats_model(y)
        m_const.method = tbats_descriptor(m_const)
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

    # For now, use simple k-vector selection (k=1 for all periods)
    # More sophisticated selection can be added later
    if m !== nothing
        k_vector = ones(Int, length(m))
        # Adjust k based on period size
        for (i, period) in enumerate(m)
            if period == 2
                k_vector[i] = 1
            else
                max_k = floor(Int, (period - 1) / 2)
                k_vector[i] = min(1, max_k)  # Start with k=1 for simplicity
            end
        end
    else
        k_vector = nothing
    end

    best_aic = Inf
    best_model = nothing

    for box_cox in box_cox_values
        for trend in trend_values
            for damping in damping_values
                if !trend && damping
                    continue
                end

                current_model = try
                    fitSpecificTBATS(
                        y_num;
                        use_box_cox = box_cox,
                        use_beta = trend,
                        use_damping = damping,
                        seasonal_periods = m,
                        k_vector = k_vector,
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                    )
                catch e
                    @warn "Model failed: $e"
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
        Dict{Symbol,Any}(:vect => best_model.parameters.vect, :control => best_model.parameters.control),
        method_label,
    )

    return result
end

function tbats(
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
    return tbats(
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
    )
end

function forecast(
    model::TBATSModel;
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
