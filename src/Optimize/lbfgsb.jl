"""
    LBFGSBOptions

Options for direct L-BFGS-B optimization.
"""
struct LBFGSBOptions
    memory_size::Int
    ftol_factor::Float64
    pg_tol::Float64
    maxit::Int
    print_level::Int
end

LBFGSBOptions(;
    memory_size::Int = 10,
    ftol_factor::Float64 = 1e7,
    pg_tol::Float64 = 1e-5,
    maxit::Int = 1000,
    print_level::Int = 0,
) = LBFGSBOptions(memory_size, ftol_factor, pg_tol, maxit, print_level)

struct _LBFGSBNonFinite <: Exception
    what::Symbol
end

@inline function _project_bounds!(
    x::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
)
    @inbounds for i in eachindex(x, lower, upper)
        xi = x[i]
        if xi < lower[i]
            x[i] = lower[i]
        elseif xi > upper[i]
            x[i] = upper[i]
        end
    end
    return x
end

@inline function _dot_vec(x::Vector{Float64}, y::Vector{Float64})::Float64
    acc = 0.0
    @inbounds @simd for i in eachindex(x, y)
        acc += x[i] * y[i]
    end
    return acc
end

@inline function _dot_col(matrix::Matrix{Float64}, col::Int, vector::Vector{Float64})::Float64
    acc = 0.0
    @inbounds @simd for i in eachindex(vector)
        acc += matrix[i, col] * vector[i]
    end
    return acc
end

@inline function _axpy_col!(
    vector::Vector{Float64},
    alpha::Float64,
    matrix::Matrix{Float64},
    col::Int,
)
    @inbounds @simd for i in eachindex(vector)
        vector[i] += alpha * matrix[i, col]
    end
    return vector
end

@inline function _ring_index(head::Int, offset_from_head::Int, memory_size::Int)::Int
    idx = head - offset_from_head
    return idx > 0 ? idx : idx + memory_size
end

@inline function _safe_objective(objective, x::Vector{Float64})::Float64
    fx = Float64(objective(x))
    isfinite(fx) || throw(_LBFGSBNonFinite(:objective))
    return fx
end

@inline function _is_at_lower(xi::Float64, li::Float64)::Bool
    return isfinite(li) && xi <= li + 1e-12 * (1.0 + abs(li))
end

@inline function _is_at_upper(xi::Float64, ui::Float64)::Bool
    return isfinite(ui) && xi >= ui - 1e-12 * (1.0 + abs(ui))
end

@inline function _is_active_bound(xi::Float64, gi::Float64, li::Float64, ui::Float64)::Bool
    return (_is_at_lower(xi, li) && gi > 0.0) || (_is_at_upper(xi, ui) && gi < 0.0)
end

@inline function _projected_gradient_inf_norm(
    x::Vector{Float64},
    g::Vector{Float64},
    lower::Vector{Float64},
    upper::Vector{Float64},
)::Float64
    max_abs = 0.0
    @inbounds for i in eachindex(x, g, lower, upper)
        trial = x[i] - g[i]
        projected = if trial < lower[i]
            lower[i]
        elseif trial > upper[i]
            upper[i]
        else
            trial
        end
        pg = projected - x[i]
        abs_pg = abs(pg)
        if abs_pg > max_abs
            max_abs = abs_pg
        end
    end
    return max_abs
end

@inline function _steepest_projected_direction!(
    direction::Vector{Float64},
    x::Vector{Float64},
    g::Vector{Float64},
    lower::Vector{Float64},
    upper::Vector{Float64},
)
    @inbounds for i in eachindex(direction, x, g, lower, upper)
        direction[i] = _is_active_bound(x[i], g[i], lower[i], upper[i]) ? 0.0 : -g[i]
    end
    return direction
end

@inline function _project_direction!(
    direction::Vector{Float64},
    x::Vector{Float64},
    g::Vector{Float64},
    lower::Vector{Float64},
    upper::Vector{Float64},
)
    @inbounds for i in eachindex(direction, x, g, lower, upper)
        if _is_active_bound(x[i], g[i], lower[i], upper[i]) ||
           (_is_at_lower(x[i], lower[i]) && direction[i] < 0.0) ||
           (_is_at_upper(x[i], upper[i]) && direction[i] > 0.0)
            direction[i] = 0.0
        end
    end
    return direction
end

function _compute_gradient!(
    gradient_out::Vector{Float64},
    objective,
    gradient::Union{Function,Nothing},
    x::Vector{Float64},
    lower::Vector{Float64},
    upper::Vector{Float64},
    f_x::Float64,
    fn_evals::Base.RefValue{Int},
    gr_evals::Base.RefValue{Int},
    x_work::Vector{Float64},
)
    if isnothing(gradient)
        step_rel = sqrt(eps(Float64))
        copyto!(x_work, x)
        @inbounds for i in eachindex(x, gradient_out, lower, upper)
            xi = x[i]
            h = step_rel * (1.0 + abs(xi))
            if xi + h <= upper[i]
                x_work[i] = xi + h
                f_plus = Float64(objective(x_work))
                fn_evals[] += 1
                isfinite(f_plus) || throw(_LBFGSBNonFinite(:objective))
                gradient_out[i] = (f_plus - f_x) / h
            elseif xi - h >= lower[i]
                x_work[i] = xi - h
                f_minus = Float64(objective(x_work))
                fn_evals[] += 1
                isfinite(f_minus) || throw(_LBFGSBNonFinite(:objective))
                gradient_out[i] = (f_x - f_minus) / h
            else
                gradient_out[i] = 0.0
            end
            x_work[i] = xi
        end
        gr_evals[] += 1
        return gradient_out
    end

    gradient_value = gradient(x)
    length(gradient_value) == length(x) || throw(ArgumentError("gradient must have length $(length(x))"))
    @inbounds for i in eachindex(gradient_out)
        gi = Float64(gradient_value[i])
        isfinite(gi) || throw(_LBFGSBNonFinite(:gradient))
        gradient_out[i] = gi
    end
    gr_evals[] += 1
    return gradient_out
end

@inline function _max_feasible_step(
    x::Vector{Float64},
    direction::Vector{Float64},
    lower::Vector{Float64},
    upper::Vector{Float64},
)::Float64
    alpha_max = Inf
    @inbounds for i in eachindex(x, direction, lower, upper)
        di = direction[i]
        if di > 0.0 && isfinite(upper[i])
            alpha_i = (upper[i] - x[i]) / di
            if alpha_i < alpha_max
                alpha_max = alpha_i
            end
        elseif di < 0.0 && isfinite(lower[i])
            alpha_i = (lower[i] - x[i]) / di
            if alpha_i < alpha_max
                alpha_max = alpha_i
            end
        end
    end
    return alpha_max
end

function _lbfgs_two_loop_direction!(
    direction::Vector{Float64},
    gradient::Vector{Float64},
    s_history::Matrix{Float64},
    y_history::Matrix{Float64},
    rho_history::Vector{Float64},
    history_size::Int,
    history_head::Int,
    initial_hessian_scale::Float64,
    q_work::Vector{Float64},
    r_work::Vector{Float64},
    alpha_work::Vector{Float64},
)
    copyto!(q_work, gradient)
    n_hist = history_size
    memory_size = size(s_history, 2)

    @inbounds for offset in 0:(n_hist - 1)
        idx = _ring_index(history_head, offset, memory_size)
        alpha_j = rho_history[idx] * _dot_col(s_history, idx, q_work)
        alpha_work[idx] = alpha_j
        _axpy_col!(q_work, -alpha_j, y_history, idx)
    end

    @inbounds @simd for i in eachindex(r_work, q_work)
        r_work[i] = initial_hessian_scale * q_work[i]
    end

    @inbounds for offset in (n_hist - 1):-1:0
        idx = _ring_index(history_head, offset, memory_size)
        beta_j = rho_history[idx] * _dot_col(y_history, idx, r_work)
        coeff = alpha_work[idx] - beta_j
        _axpy_col!(r_work, coeff, s_history, idx)
    end

    @inbounds @simd for i in eachindex(direction, r_work)
        direction[i] = -r_work[i]
    end
    return direction
end

function _line_search!(
    objective,
    gradient::Union{Function,Nothing},
    x::Vector{Float64},
    f_x::Float64,
    g_x::Vector{Float64},
    direction::Vector{Float64},
    lower::Vector{Float64},
    upper::Vector{Float64},
    x_trial::Vector{Float64},
    g_trial::Vector{Float64},
    enforce_curvature::Bool,
    fn_evals::Base.RefValue{Int},
    gr_evals::Base.RefValue{Int},
    x_work::Vector{Float64},
)
    c1 = 1e-4
    c2 = 0.9
    max_backtracks = 30
    min_step = 1e-16

    gtd = _dot_vec(g_x, direction)
    if !(gtd < 0.0)
        return false, 0.0, f_x
    end

    alpha_max = _max_feasible_step(x, direction, lower, upper)
    if isfinite(alpha_max)
        alpha_max = max(alpha_max, 0.0)
    end
    alpha = min(1.0, alpha_max)
    if !(alpha > 0.0)
        return false, 0.0, f_x
    end

    f_trial = f_x
    @inbounds for _ in 1:max_backtracks
        @simd for i in eachindex(x_trial, x, direction)
            x_trial[i] = x[i] + alpha * direction[i]
        end
        _project_bounds!(x_trial, lower, upper)

        f_trial = _safe_objective(objective, x_trial)
        fn_evals[] += 1

        if f_trial <= f_x + c1 * alpha * gtd
            _compute_gradient!(
                g_trial,
                objective,
                gradient,
                x_trial,
                lower,
                upper,
                f_trial,
                fn_evals,
                gr_evals,
                x_work,
            )
            if !enforce_curvature || abs(_dot_vec(g_trial, direction)) <= c2 * abs(gtd) || alpha <= 1e-8
                return true, alpha, f_trial
            end
        end

        alpha *= 0.5
        if alpha < min_step
            break
        end
    end

    return false, alpha, f_trial
end

raw"""
    lbfgsb(f, g, x0; mask=trues(length(x0)), lower=nothing, upper=nothing, options=LBFGSBOptions())

Direct equation-based L-BFGS-B implementation for bound-constrained minimization:

```math
\min f(x)\quad\text{s.t.}\quad l \le x \le u
```

Uses projected-gradient convergence, limited-memory inverse-Hessian recursion, and
bound-feasible Wolfe line search.
"""
function lbfgsb(
    f::Function,
    g::Union{Function,Nothing},
    x0::Vector{Float64};
    mask = trues(length(x0)),
    lower::Union{Nothing,Vector{Float64}} = nothing,
    upper::Union{Nothing,Vector{Float64}} = nothing,
    options::LBFGSBOptions = LBFGSBOptions(),
)
    n = length(x0)
    length(mask) == n || throw(ArgumentError("mask length must equal x0 length"))

    if n == 0
        f0 = Float64(f(x0))
        return (
            x_opt = copy(x0),
            f_opt = f0,
            n_iter = 0,
            fail = 0,
            fn_evals = 1,
            gr_evals = 0,
            message = "NOTHING TO DO",
        )
    end

    lower_bound = isnothing(lower) ? fill(-Inf, n) : copy(lower)
    upper_bound = isnothing(upper) ? fill(Inf, n) : copy(upper)
    length(lower_bound) == n || throw(ArgumentError("lower must have length $n"))
    length(upper_bound) == n || throw(ArgumentError("upper must have length $n"))

    @inbounds for i in 1:n
        if !mask[i]
            lower_bound[i] = x0[i]
            upper_bound[i] = x0[i]
        end
        if isnan(lower_bound[i])
            lower_bound[i] = -Inf
        end
        if isnan(upper_bound[i])
            upper_bound[i] = Inf
        end
    end

    @inbounds for i in 1:n
        if lower_bound[i] > upper_bound[i]
            return (
                x_opt = copy(x0),
                f_opt = Inf,
                n_iter = 0,
                fail = 52,
                fn_evals = 0,
                gr_evals = 0,
                message = "Error: no feasible solution (lower > upper)",
            )
        end
    end

    m = max(options.memory_size, 1)
    pg_tolerance = options.pg_tol > 0.0 ? options.pg_tol : 1e-5
    max_iterations = max(options.maxit, 0)
    has_finite_bounds = any(isfinite, lower_bound) || any(isfinite, upper_bound)
    enforce_curvature = !has_finite_bounds

    x = copy(x0)
    _project_bounds!(x, lower_bound, upper_bound)

    f_calls = Ref(0)
    g_calls = Ref(0)

    gradient = zeros(Float64, n)
    direction = similar(gradient)
    x_trial = similar(x)
    g_trial = similar(gradient)
    step_change = similar(x)
    grad_change = similar(gradient)
    x_work = copy(x)
    q_work = similar(gradient)
    r_work = similar(gradient)
    alpha_work = zeros(Float64, m)

    s_history = Matrix{Float64}(undef, n, m)
    y_history = Matrix{Float64}(undef, n, m)
    rho_history = zeros(Float64, m)
    history_size = 0
    history_head = 0
    initial_hessian_scale = 1.0

    fx = 0.0
    try
        fx = _safe_objective(f, x)
        f_calls[] += 1
        _compute_gradient!(
            gradient,
            f,
            g,
            x,
            lower_bound,
            upper_bound,
            fx,
            f_calls,
            g_calls,
            x_work,
        )

        if max_iterations == 0
            return (
                x_opt = copy(x),
                f_opt = fx,
                n_iter = 0,
                fail = 1,
                fn_evals = f_calls[],
                gr_evals = g_calls[],
                message = "maximum iterations reached",
            )
        end

        for iteration in 1:max_iterations
            pg_norm = _projected_gradient_inf_norm(x, gradient, lower_bound, upper_bound)
            if pg_norm <= pg_tolerance
                return (
                    x_opt = copy(x),
                    f_opt = fx,
                    n_iter = iteration - 1,
                    fail = 0,
                    fn_evals = f_calls[],
                    gr_evals = g_calls[],
                    message = "Converged",
                )
            end

            if history_size > 0
                _lbfgs_two_loop_direction!(
                    direction,
                    gradient,
                    s_history,
                    y_history,
                    rho_history,
                    history_size,
                    history_head,
                    initial_hessian_scale,
                    q_work,
                    r_work,
                    alpha_work,
                )
                _project_direction!(direction, x, gradient, lower_bound, upper_bound)
                if !(_dot_vec(gradient, direction) < 0.0)
                    _steepest_projected_direction!(direction, x, gradient, lower_bound, upper_bound)
                end
            else
                _steepest_projected_direction!(direction, x, gradient, lower_bound, upper_bound)
            end

            direction_norm2 = _dot_vec(direction, direction)
            if direction_norm2 < 1e-32
                return (
                    x_opt = copy(x),
                    f_opt = fx,
                    n_iter = iteration - 1,
                    fail = 0,
                    fn_evals = f_calls[],
                    gr_evals = g_calls[],
                    message = "Converged (no feasible descent direction)",
                )
            end

            accepted, step_size, fx_trial = _line_search!(
                f,
                g,
                x,
                fx,
                gradient,
                direction,
                lower_bound,
                upper_bound,
                x_trial,
                g_trial,
                enforce_curvature,
                f_calls,
                g_calls,
                x_work,
            )

            if !accepted
                return (
                    x_opt = copy(x),
                    f_opt = fx,
                    n_iter = iteration,
                    fail = 52,
                    fn_evals = f_calls[],
                    gr_evals = g_calls[],
                    message = "abnormal termination: line search failed",
                )
            end

            @inbounds @simd for i in eachindex(step_change, x_trial, x)
                step_change[i] = x_trial[i] - x[i]
            end
            @inbounds @simd for i in eachindex(grad_change, g_trial, gradient)
                grad_change[i] = g_trial[i] - gradient[i]
            end

            sy = _dot_vec(step_change, grad_change)
            yy = _dot_vec(grad_change, grad_change)

            copyto!(x, x_trial)
            copyto!(gradient, g_trial)
            fx = fx_trial

            if sy > eps(Float64) * yy && yy > 0.0
                history_head = history_head == m ? 1 : (history_head + 1)
                @inbounds @simd for i in eachindex(step_change)
                    s_history[i, history_head] = step_change[i]
                    y_history[i, history_head] = grad_change[i]
                end
                rho_history[history_head] = 1.0 / sy
                if history_size < m
                    history_size += 1
                end
                initial_hessian_scale = sy / yy
            end

            if options.print_level > 0 && (iteration % options.print_level == 0 || iteration == 1)
                println(
                    "iter=", iteration,
                    " step=", step_size,
                    " f=", fx,
                    " pg_inf=", _projected_gradient_inf_norm(x, gradient, lower_bound, upper_bound),
                )
            end
        end

        return (
            x_opt = copy(x),
            f_opt = fx,
            n_iter = max_iterations,
            fail = 1,
            fn_evals = f_calls[],
            gr_evals = g_calls[],
            message = "maximum iterations reached",
        )
    catch err
        if err isa _LBFGSBNonFinite
            msg = err.what === :gradient ?
                  "Error: gradient contains non-finite values" :
                  "Error: objective function returned non-finite value"
            return (
                x_opt = copy(x),
                f_opt = fx,
                n_iter = 0,
                fail = 52,
                fn_evals = f_calls[],
                gr_evals = g_calls[],
                message = msg,
            )
        end
        rethrow()
    end
end
