"""
    BFGSOptions

Options for BFGS optimization.
"""
Base.@kwdef struct BFGSOptions
    abstol::Float64 = -Inf
    reltol::Float64 = sqrt(eps(Float64))
    gtol::Float64 = 0.0
    trace::Bool = false
    maxit::Int = 100
    report_interval::Int = 10
    c1::Float64 = 1e-4
    c2::Float64 = 0.9
    max_linesearch::Int = 20
end

struct _BFGSObjectiveNonFinite <: Exception
    value::Float64
end

@inline function _write_full_from_active!(
    x_full::Vector{Float64},
    x_base::Vector{Float64},
    active::Vector{Int},
    x_active::AbstractVector{<:Real},
)
    copyto!(x_full, x_base)
    @inbounds for k in eachindex(active)
        x_full[active[k]] = x_active[k]
    end
    return x_full
end

@inline function _inf_norm(x::AbstractVector{<:Real})
    norm_inf = 0.0
    @inbounds @simd for i in eachindex(x)
        norm_inf = max(norm_inf, abs(Float64(x[i])))
    end
    return norm_inf
end

@inline function _objective_or_inf!(f_active::Function, x_active::Vector{Float64})
    try
        return f_active(x_active)
    catch err
        if err isa _BFGSObjectiveNonFinite
            return Inf
        end
        rethrow()
    end
end

function _central_difference_gradient!(
    gradient_out::Vector{Float64},
    x_active::Vector{Float64},
    f_active::Function,
)
    step_scale = cbrt(eps(Float64))
    @inbounds for i in eachindex(x_active)
        x_i = x_active[i]
        step_i = step_scale * max(abs(x_i), 1.0)

        x_active[i] = x_i + step_i
        f_plus = _objective_or_inf!(f_active, x_active)

        x_active[i] = x_i - step_i
        f_minus = _objective_or_inf!(f_active, x_active)

        x_active[i] = x_i

        if !isfinite(f_plus) || !isfinite(f_minus)
            return false
        end
        gradient_out[i] = (f_plus - f_minus) / (2.0 * step_i)
    end
    return true
end

function _analytic_active_gradient!(
    gradient_out::Vector{Float64},
    g!::Function,
    x_active::Vector{Float64},
    x_full::Vector{Float64},
    x_base::Vector{Float64},
    active::Vector{Int},
    gradient_full::Vector{Float64},
)
    _write_full_from_active!(x_full, x_base, active, x_active)
    fill!(gradient_full, 0.0)
    g!(gradient_full, x_full)
    @inbounds for k in eachindex(active)
        grad_value = gradient_full[active[k]]
        isfinite(grad_value) || return false
        gradient_out[k] = grad_value
    end
    return true
end

@inline function _trial_point!(
    x_trial::Vector{Float64},
    x_current::Vector{Float64},
    direction::Vector{Float64},
    alpha::Float64,
)
    @inbounds @simd for i in eachindex(x_trial, x_current, direction)
        x_trial[i] = x_current[i] + alpha * direction[i]
    end
    return x_trial
end

@inline function _quadratic_interpolation(
    alpha_a::Float64,
    phi_a::Float64,
    dphi_a::Float64,
    alpha_b::Float64,
    phi_b::Float64,
)
    numerator = dphi_a * (alpha_a - alpha_b)^2
    denominator = 2.0 * (phi_a - phi_b - dphi_a * (alpha_a - alpha_b))
    abs(denominator) > eps(Float64) || return NaN
    return alpha_a - numerator / denominator
end

@inline function _cubic_interpolation(
    alpha_a::Float64,
    phi_a::Float64,
    dphi_a::Float64,
    alpha_b::Float64,
    phi_b::Float64,
    dphi_b::Float64,
)
    alpha_a != alpha_b || return NaN
    delta = alpha_a - alpha_b
    d1 = dphi_a + dphi_b - 3.0 * (phi_a - phi_b) / delta
    discriminant = d1 * d1 - dphi_a * dphi_b
    discriminant >= 0.0 || return NaN
    d2 = sign(alpha_b - alpha_a) * sqrt(discriminant)
    denominator = dphi_b - dphi_a + 2.0 * d2
    abs(denominator) > eps(Float64) || return NaN
    return alpha_b - (alpha_b - alpha_a) * (dphi_b + d2 - d1) / denominator
end

@inline function _interpolate_zoom(
    alpha_lo::Float64,
    alpha_hi::Float64,
    phi_lo::Float64,
    phi_hi::Float64,
    dphi_lo::Float64,
    dphi_hi::Float64,
)
    alpha_try = NaN
    if isfinite(dphi_lo) && isfinite(dphi_hi)
        alpha_try = _cubic_interpolation(alpha_lo, phi_lo, dphi_lo, alpha_hi, phi_hi, dphi_hi)
    end
    if !isfinite(alpha_try) && isfinite(dphi_lo)
        alpha_try = _quadratic_interpolation(alpha_lo, phi_lo, dphi_lo, alpha_hi, phi_hi)
    end
    return alpha_try
end

function _zoom_line_search!(
    alpha_lo::Float64,
    alpha_hi::Float64,
    phi_lo::Float64,
    phi_hi::Float64,
    dphi_lo::Float64,
    dphi_hi::Float64,
    x_current::Vector{Float64},
    f_current::Float64,
    dphi0::Float64,
    direction::Vector{Float64},
    f_active::Function,
    gradient_active!::Function,
    x_trial::Vector{Float64},
    g_trial::Vector{Float64},
    c1::Float64,
    c2::Float64,
    max_linesearch::Int,
)
    if alpha_hi < alpha_lo
        alpha_lo, alpha_hi = alpha_hi, alpha_lo
        phi_lo, phi_hi = phi_hi, phi_lo
        dphi_lo, dphi_hi = dphi_hi, dphi_lo
    end

    for _ in 1:max_linesearch
        alpha_j = _interpolate_zoom(alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo, dphi_hi)

        interval_width = alpha_hi - alpha_lo
        lower_guard = alpha_lo + 0.1 * interval_width
        upper_guard = alpha_hi - 0.1 * interval_width
        if !(isfinite(alpha_j) && alpha_j > lower_guard && alpha_j < upper_guard)
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
        end

        _trial_point!(x_trial, x_current, direction, alpha_j)
        phi_j = _objective_or_inf!(f_active, x_trial)

        if !isfinite(phi_j) || phi_j > f_current + c1 * alpha_j * dphi0 || phi_j >= phi_lo
            alpha_hi = alpha_j
            phi_hi = phi_j
            dphi_hi = NaN
        else
            grad_ok = gradient_active!(g_trial, x_trial)
            if !grad_ok
                alpha_hi = alpha_j
                phi_hi = phi_j
                dphi_hi = NaN
                continue
            end

            dphi_j = LinearAlgebra.dot(g_trial, direction)
            if abs(dphi_j) <= -c2 * dphi0
                return true, alpha_j, phi_j
            end

            if dphi_j * (alpha_hi - alpha_lo) >= 0.0
                alpha_hi = alpha_lo
                phi_hi = phi_lo
                dphi_hi = dphi_lo
            end
            alpha_lo = alpha_j
            phi_lo = phi_j
            dphi_lo = dphi_j
        end

        if abs(alpha_hi - alpha_lo) <= 1e-14 * max(1.0, abs(alpha_lo) + abs(alpha_hi))
            break
        end
    end

    return false, 0.0, Inf
end

function _armijo_backtracking!(
    x_current::Vector{Float64},
    f_current::Float64,
    dphi0::Float64,
    direction::Vector{Float64},
    f_active::Function,
    gradient_active!::Function,
    x_trial::Vector{Float64},
    g_trial::Vector{Float64},
    c1::Float64,
    max_linesearch::Int,
    initial_step::Float64,
)
    alpha = max(initial_step, 1e-12)
    for _ in 1:max_linesearch
        _trial_point!(x_trial, x_current, direction, alpha)
        phi_alpha = _objective_or_inf!(f_active, x_trial)
        if isfinite(phi_alpha) && phi_alpha <= f_current + c1 * alpha * dphi0
            if gradient_active!(g_trial, x_trial)
                return true, alpha, phi_alpha
            end
        end
        alpha *= 0.5
        alpha < 1e-16 && break
    end
    return false, 0.0, Inf
end

function _strong_wolfe_line_search!(
    x_current::Vector{Float64},
    f_current::Float64,
    g_current::Vector{Float64},
    direction::Vector{Float64},
    f_active::Function,
    gradient_active!::Function,
    x_trial::Vector{Float64},
    g_trial::Vector{Float64},
    c1::Float64,
    c2::Float64,
    initial_step::Float64,
    max_linesearch::Int,
)
    dphi0 = LinearAlgebra.dot(g_current, direction)
    dphi0 < 0.0 || return false, 0.0, f_current

    alpha_prev = 0.0
    phi_prev = f_current
    dphi_prev = dphi0
    alpha = max(initial_step, 1e-12)

    for i in 1:max_linesearch
        _trial_point!(x_trial, x_current, direction, alpha)
        phi_alpha = _objective_or_inf!(f_active, x_trial)

        if !isfinite(phi_alpha) || phi_alpha > f_current + c1 * alpha * dphi0 || (i > 1 && phi_alpha >= phi_prev)
            return _zoom_line_search!(
                alpha_prev, alpha, phi_prev, phi_alpha, dphi_prev, NaN,
                x_current, f_current, dphi0, direction, f_active, gradient_active!,
                x_trial, g_trial, c1, c2, max_linesearch,
            )
        end

        grad_ok = gradient_active!(g_trial, x_trial)
        if !grad_ok
            return _zoom_line_search!(
                alpha_prev, alpha, phi_prev, phi_alpha, dphi_prev, NaN,
                x_current, f_current, dphi0, direction, f_active, gradient_active!,
                x_trial, g_trial, c1, c2, max_linesearch,
            )
        end
        dphi_alpha = LinearAlgebra.dot(g_trial, direction)

        if abs(dphi_alpha) <= -c2 * dphi0
            return true, alpha, phi_alpha
        end
        if dphi_alpha >= 0.0
            return _zoom_line_search!(
                alpha, alpha_prev, phi_alpha, phi_prev, dphi_alpha, dphi_prev,
                x_current, f_current, dphi0, direction, f_active, gradient_active!,
                x_trial, g_trial, c1, c2, max_linesearch,
            )
        end

        alpha_prev = alpha
        phi_prev = phi_alpha
        dphi_prev = dphi_alpha
        alpha = min(2.0 * alpha, 64.0)
    end

    return _armijo_backtracking!(
        x_current, f_current, dphi0, direction, f_active, gradient_active!,
        x_trial, g_trial, c1, max_linesearch, initial_step,
    )
end

function _set_diagonal_scaled_identity!(H::Matrix{Float64}, scale::Float64)
    fill!(H, 0.0)
    n = size(H, 1)
    @inbounds @simd for i in 1:n
        H[i, i] = scale
    end
    return H
end

function _bfgs_inverse_hessian_update!(
    H::Matrix{Float64},
    step_vector::Vector{Float64},
    gradient_step::Vector{Float64},
    hessian_times_gradient_step::Vector{Float64},
)
    sy = LinearAlgebra.dot(step_vector, gradient_step)
    sy > 0.0 || return false

    LinearAlgebra.mul!(hessian_times_gradient_step, H, gradient_step)
    rho = inv(sy)
    yHy = LinearAlgebra.dot(gradient_step, hessian_times_gradient_step)
    ss_scale = (1.0 + rho * yHy) * rho

    n = length(step_vector)
    @inbounds for j in 1:n
        s_j = step_vector[j]
        hy_j = hessian_times_gradient_step[j]
        for i in 1:j
            updated_value = H[i, j] -
                            rho * (step_vector[i] * hy_j + hessian_times_gradient_step[i] * s_j) +
                            ss_scale * step_vector[i] * s_j
            H[i, j] = updated_value
            H[j, i] = updated_value
        end
    end
    return true
end

function _compose_full_solution(
    x_base::Vector{Float64},
    active::Vector{Int},
    x_active::Vector{Float64},
)
    x_solution = copy(x_base)
    @inbounds for k in eachindex(active)
        x_solution[active[k]] = x_active[k]
    end
    return x_solution
end

"""
    bfgs(f, g, x0; mask=trues(length(x0)), options=BFGSOptions()) -> NamedTuple

Direct BFGS implementation with strong Wolfe line search.
"""
function bfgs(
    f::Function,
    g::Union{Function,Nothing},
    x0::Vector{Float64};
    mask = trues(length(x0)),
    options::BFGSOptions = BFGSOptions(),
)
    abstol = options.abstol
    reltol = options.reltol
    gtol = options.gtol
    trace = options.trace
    maxit = options.maxit
    report_interval = options.report_interval
    c1 = options.c1
    c2 = options.c2
    max_linesearch = options.max_linesearch

    n_total = length(x0)
    length(mask) == n_total || throw(ArgumentError("mask length must equal x0 length"))
    report_interval > 0 || throw(ArgumentError("report_interval must be positive for BFGS"))
    0.0 < c1 < c2 < 1.0 || throw(ArgumentError("BFGS line search requires 0 < c1 < c2 < 1"))
    max_linesearch > 0 || throw(ArgumentError("max_linesearch must be positive"))

    x_start = copy(x0)
    active = findall(mask)
    n_active = length(active)
    x_active = x_start[active]

    x_objective = copy(x_start)
    x_gradient = copy(x_start)
    gradient_full = zeros(Float64, n_total)

    f_calls = Ref(0)
    g_calls = Ref(0)

    f_active = function (x_active_current::Vector{Float64})
        _write_full_from_active!(x_objective, x_start, active, x_active_current)
        f_value = Float64(f(x_objective))
        f_calls[] += 1
        isfinite(f_value) || throw(_BFGSObjectiveNonFinite(f_value))
        return f_value
    end

    f_current = try
        f_active(x_active)
    catch err
        if err isa _BFGSObjectiveNonFinite
            error("BFGS: initial objective value is not finite")
        end
        rethrow()
    end

    if maxit <= 0 || n_active == 0
        return (
            x_opt = _compose_full_solution(x_start, active, x_active),
            f_opt = f_current,
            n_iter = 0,
            fail = 0,
            fn_evals = f_calls[],
            gr_evals = 0,
        )
    end

    gradient_active! = function (gradient_out::Vector{Float64}, x_active_current::Vector{Float64})
        g_calls[] += 1
        if isnothing(g)
            return _central_difference_gradient!(gradient_out, x_active_current, f_active)
        end
        return _analytic_active_gradient!(
            gradient_out, g, x_active_current, x_gradient, x_start, active, gradient_full,
        )
    end

    gradient_current = zeros(Float64, n_active)
    if !gradient_active!(gradient_current, x_active)
        return (
            x_opt = _compose_full_solution(x_start, active, x_active),
            f_opt = f_current,
            n_iter = 0,
            fail = 1,
            fn_evals = f_calls[],
            gr_evals = g_calls[],
        )
    end

    hessian_inverse = Matrix{Float64}(undef, n_active, n_active)
    _set_diagonal_scaled_identity!(hessian_inverse, 1.0)

    direction = similar(gradient_current)
    x_trial = similar(x_active)
    gradient_trial = similar(gradient_current)
    step_vector = similar(gradient_current)
    gradient_step = similar(gradient_current)
    hessian_times_gradient_step = similar(gradient_current)

    iteration_count = 0
    converged = false

    for iteration in 1:maxit
        gradient_norm_inf = _inf_norm(gradient_current)
        if gradient_norm_inf <= gtol || f_current <= abstol
            converged = true
            break
        end

        LinearAlgebra.mul!(direction, hessian_inverse, gradient_current)
        @inbounds @simd for i in eachindex(direction)
            direction[i] = -direction[i]
        end

        if !(LinearAlgebra.dot(gradient_current, direction) < 0.0)
            _set_diagonal_scaled_identity!(hessian_inverse, 1.0)
            @inbounds @simd for i in eachindex(direction, gradient_current)
                direction[i] = -gradient_current[i]
            end
        end

        initial_step = iteration == 1 ? inv(max(gradient_norm_inf, eps(Float64))) : 1.0
        accepted, step_length, f_trial = _strong_wolfe_line_search!(
            x_active, f_current, gradient_current, direction,
            f_active, gradient_active!, x_trial, gradient_trial,
            c1, c2, initial_step, max_linesearch,
        )

        if !accepted
            break
        end

        @inbounds @simd for i in eachindex(step_vector, x_trial, x_active)
            step_vector[i] = x_trial[i] - x_active[i]
        end
        @inbounds @simd for i in eachindex(gradient_step, gradient_trial, gradient_current)
            gradient_step[i] = gradient_trial[i] - gradient_current[i]
        end

        sy = LinearAlgebra.dot(step_vector, gradient_step)
        if iteration == 1 && sy > 0.0
            yy = LinearAlgebra.dot(gradient_step, gradient_step)
            if yy > eps(Float64)
                _set_diagonal_scaled_identity!(hessian_inverse, sy / yy)
            end
        end
        sy > 0.0 && _bfgs_inverse_hessian_update!(
            hessian_inverse, step_vector, gradient_step, hessian_times_gradient_step,
        )

        relative_decrease = abs(f_trial - f_current) / max(1.0, abs(f_current))
        step_norm_inf = _inf_norm(step_vector)

        copyto!(x_active, x_trial)
        copyto!(gradient_current, gradient_trial)
        f_current = f_trial
        iteration_count = iteration

        if trace && (iteration == 1 || iteration % report_interval == 0)
            println(
                "iter=", iteration,
                " f=", f_current,
                " g_inf=", _inf_norm(gradient_current),
                " step=", step_length,
            )
        end

        if _inf_norm(gradient_current) <= gtol ||
           f_current <= abstol ||
           relative_decrease <= reltol ||
           step_norm_inf <= sqrt(eps(Float64))
            converged = true
            break
        end
    end

    if converged
        return (
            x_opt = _compose_full_solution(x_start, active, x_active),
            f_opt = f_current,
            n_iter = iteration_count,
            fail = 0,
            fn_evals = f_calls[],
            gr_evals = g_calls[],
        )
    else
        return (
            x_opt = _compose_full_solution(x_start, active, x_active),
            f_opt = f_current,
            n_iter = iteration_count,
            fail = 1,
            fn_evals = f_calls[],
            gr_evals = g_calls[],
        )
    end
end
