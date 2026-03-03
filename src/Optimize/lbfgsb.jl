import LinearAlgebra
using LinearAlgebra: dot, norm, ldiv!, UpperTriangular

"""
    LBFGSBOptions

Options for the L-BFGS-B limited-memory quasi-Newton optimizer with box constraints.

# Fields
- `memory_size::Int` — Number of correction pairs to store (default: 10)
- `ftol_factor::Float64` — Function tolerance factor; convergence when
  `(f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) ≤ ftol_factor * eps(Float64)` (default: 1e7)
- `pg_tol::Float64` — Projected gradient tolerance (default: 1e-5)
- `maxit::Int` — Maximum iterations (default: 1000)
- `print_level::Int` — Verbosity level; 0=silent (default: 0)

# References

- Byrd, R. H., Lu, P., Nocedal, J. & Zhu, C. (1995). A limited memory algorithm for
  bound constrained optimization. *SIAM J. Scientific Computing*, 16(5), 1190–1208.
"""
struct LBFGSBOptions
    memory_size::Int
    ftol_factor::Float64
    pg_tol::Float64
    maxit::Int
    print_level::Int
end

LBFGSBOptions(; memory_size::Int=10, ftol_factor::Float64=1e7, pg_tol::Float64=1e-5, maxit::Int=1000, print_level::Int=0) =
    LBFGSBOptions(memory_size, ftol_factor, pg_tol, maxit, print_level)


"""
    BoundType

Classification of variable bound constraints.
Byrd et al. (1995), Section 2.
"""
@enum BoundType::Int32 begin
    UNBOUNDED   = 0
    LOWER_ONLY  = 1
    BOTH_BOUNDS = 2
    UPPER_ONLY  = 3
end

@inline _has_lower(b::BoundType) = (b == LOWER_ONLY || b == BOTH_BOUNDS)
@inline _has_upper(b::BoundType) = (b == BOTH_BOUNDS || b == UPPER_ONLY)
@inline _is_bounded(b::BoundType) = (b != UNBOUNDED)

"""
    VarStatus

Classification of variable activity in L-BFGS-B.
Byrd et al. (1995), Section 3.
"""
@enum VarStatus::Int begin
    VAR_ZERO_GRADIENT = -3
    VAR_UNBOUNDED     = -1
    VAR_FREE          = 0
    VAR_LOWER_ACTIVE  = 1
    VAR_UPPER_ACTIVE  = 2
    VAR_FIXED         = 3
end

@inline _is_free(s::VarStatus) = (s == VAR_FREE || s == VAR_UNBOUNDED || s == VAR_ZERO_GRADIENT)
@inline _is_active(s::VarStatus) = (s == VAR_LOWER_ACTIVE || s == VAR_UPPER_ACTIVE || s == VAR_FIXED)


"""
    SingularFactorizationError

Thrown when a triangular system or Cholesky factorization encounters a
singular (zero-diagonal) element. Caught in the main L-BFGS-B loop to
trigger memory refresh.
"""
struct SingularFactorizationError <: Exception end

"""Advance a circular buffer pointer: wraps `ptr` around ring of size `m`."""
@inline _next_ring(ptr::Int, m::Int) = ptr % m + 1


"""
    LBFGSBWorkspace

Pre-allocated mutable workspace holding all arrays and state for the L-BFGS-B
algorithm.
"""
mutable struct LBFGSBWorkspace
    # Problem dimensions
    n::Int
    m::Int

    # n-length vectors
    cauchy_point::Vector{Float64}      # Generalized Cauchy Point
    reduced_grad::Vector{Float64}      # reduced gradient
    search_dir::Vector{Float64}        # search/step direction
    breakpoint_times::Vector{Float64}  # breakpoint times for GCP
    proj_grad::Vector{Float64}         # projected gradient

    # n-length index/status arrays
    free_vars::Vector{Int}             # free variable indices
    sort_order::Vector{Int}            # breakpoint sort order
    bound_types::Vector{BoundType}     # bound classification per variable
    var_status::Vector{VarStatus}      # active/free status per variable

    # Correction pair storage (n × m)
    step_corrections::Matrix{Float64}  # step correction vectors Sₖ
    grad_corrections::Matrix{Float64}  # gradient correction vectors Yₖ

    # m × m matrices (compact L-BFGS representation)
    sy_products::Matrix{Float64}       # Sᵀ Y products
    ss_products::Matrix{Float64}       # Sᵀ S products
    compact_form::Matrix{Float64}      # W^T compact form factor

    # 2m × 2m matrices
    reduced_hessian::Matrix{Float64}   # reduced Hessian for subspace min
    hessian_parts::Matrix{Float64}     # assembly workspace

    # 2m-length work vectors
    p_work::Vector{Float64}            # compact rep solve output
    c_work::Vector{Float64}            # GCP cumulative update
    breakpoint_work::Vector{Float64}   # GCP work buffer
    v_work::Vector{Float64}            # GCP temporary
    subspace_work::Vector{Float64}     # subspace minimization work
    solve_result::Vector{Float64}      # compact solve output
    solve_input::Vector{Float64}       # compact solve input

    # L-BFGS state scalars
    theta::Float64                     # Scaling factor θ
    n_corrections::Int                 # current number of stored corrections
    ring_head::Int                     # circular buffer head pointer
    tail_index::Int
    update_count::Int
    did_update::Bool

    # Active set state
    n_free::Int
    n_entering::Int
    leaving_start::Int
    has_bounds::Bool
    boxed::Bool
end

"""
    LBFGSBWorkspace(n, m, lb, ub)

Construct and initialize an L-BFGS-B workspace for a problem with `n` variables
and `m` correction pairs, given lower bounds `lb` and upper bounds `ub`.
"""
function LBFGSBWorkspace(n::Int, m::Int, lb::Vector{Float64}, ub::Vector{Float64})
    bound_types = _classify_bound_types(lb, ub)
    ws = LBFGSBWorkspace(
        n, m,
        Vector{Float64}(undef, n),   # cauchy_point
        Vector{Float64}(undef, n),   # reduced_grad
        Vector{Float64}(undef, n),   # search_dir
        Vector{Float64}(undef, n),   # breakpoint_times
        similar(lb),                  # proj_grad
        Vector{Int}(undef, n),       # free_vars
        Vector{Int}(undef, n),       # sort_order
        bound_types,                  # bound_types
        Vector{VarStatus}(undef, n), # var_status
        zeros(Float64, n, m),        # step_corrections
        zeros(Float64, n, m),        # grad_corrections
        zeros(Float64, m, m),        # sy_products
        zeros(Float64, m, m),        # ss_products
        zeros(Float64, m, m),        # compact_form
        zeros(Float64, 2m, 2m),      # reduced_hessian
        zeros(Float64, 2m, 2m),      # hessian_parts
        Vector{Float64}(undef, 2m),  # p_work
        Vector{Float64}(undef, 2m),  # c_work
        Vector{Float64}(undef, 2m),  # breakpoint_work
        Vector{Float64}(undef, 2m),  # v_work
        Vector{Float64}(undef, 2m),  # subspace_work
        Vector{Float64}(undef, 2m),  # solve_result
        Vector{Float64}(undef, 2m),  # solve_input
        1.0,   # theta
        0,     # n_corrections
        1,     # ring_head
        0,     # tail_index
        0,     # update_count
        false, # did_update
        n,     # n_free
        0,     # n_entering
        n + 1, # leaving_start
        false, # has_bounds
        false, # boxed
    )
    return ws
end

"""
    _reset_memory!(ws::LBFGSBWorkspace)

Reset L-BFGS memory after encountering singularity or non-positive definiteness.
"""
@inline function _reset_memory!(ws::LBFGSBWorkspace)
    ws.n_corrections = 0
    ws.ring_head = 1
    ws.theta = 1.0
    ws.update_count = 0
    ws.did_update = false
    return nothing
end


"""
    _classify_bound_types(lb, ub) -> Vector{BoundType}

Classify each variable's bound type.
Byrd et al. (1995), Section 2.
"""
function _classify_bound_types(lb::Vector{Float64}, ub::Vector{Float64})
    n = length(lb)
    bounds = Vector{BoundType}(undef, n)
    @inbounds for i in 1:n
        if isfinite(lb[i]) && isfinite(ub[i])
            bounds[i] = BOTH_BOUNDS
        elseif isfinite(lb[i])
            bounds[i] = LOWER_ONLY
        elseif isfinite(ub[i])
            bounds[i] = UPPER_ONLY
        else
            bounds[i] = UNBOUNDED
        end
    end
    return bounds
end

@inline function _project!(y::AbstractVector{Float64}, lb::Vector{Float64}, ub::Vector{Float64})
    @inbounds for i in eachindex(y)
        yi = y[i]
        if yi < lb[i]
            y[i] = lb[i]
        elseif yi > ub[i]
            y[i] = ub[i]
        end
    end
    return y
end


"""
    _proj_grad!(pg, x, grad, lb, ub, bounds)

Compute the projected gradient and return its infinity norm.
The projected gradient measures first-order optimality for box-constrained problems.

Byrd et al. (1995), Eq. (2.1).
"""
function _proj_grad!(pg::Vector{Float64}, x::Vector{Float64}, grad::Vector{Float64},
    lb::Vector{Float64}, ub::Vector{Float64}, bounds::Vector{BoundType})
    @inbounds for i in eachindex(x)
        gi = grad[i]
        bi = bounds[i]
        if _is_bounded(bi)
            if gi < 0.0
                if _has_upper(bi)
                    gi = max(x[i] - ub[i], gi)
                end
            else
                if _has_lower(bi)
                    gi = min(x[i] - lb[i], gi)
                end
            end
        end
        pg[i] = gi
    end

    return norm(pg, Inf)
end


function _cholesky!(A::AbstractMatrix{Float64}, n::Int)
    R = view(A, 1:n, 1:n)
    _, info = LinearAlgebra.LAPACK.potrf!('U', R)
    info > 0 && throw(SingularFactorizationError())
    return nothing
end

@inline function _check_diag_nonzero!(A::AbstractMatrix, n::Int)
    @inbounds for j in 1:n
        A[j, j] == 0.0 && throw(SingularFactorizationError())
    end
end

"""
    _compact_representation_solve!(p, ws, v)

Solve the compact L-BFGS representation system using the triangular matrices
`sy_products` (correction pairs) and `compact_form` (compact form) from workspace.

Throws `SingularFactorizationError` if the triangular system is singular.

Byrd et al. (1995), Theorem 2.3.
"""
function _compact_representation_solve!(p::AbstractVector{Float64}, ws::LBFGSBWorkspace,
    v::AbstractVector{Float64})
    ncorr = ws.n_corrections
    if ncorr == 0
        return nothing
    end
    sy_prod = ws.sy_products
    compact = ws.compact_form
    p[ncorr+1] = v[ncorr+1]
    @inbounds for i in 2:ncorr
        p[ncorr+i] = v[ncorr+i] + sum(sy_prod[i, k] * v[k] / sy_prod[k, k] for k in 1:i-1)
    end
    _check_diag_nonzero!(compact, ncorr)
    ldiv!(UpperTriangular(view(compact, 1:ncorr, 1:ncorr))', view(p, ncorr+1:2*ncorr))
    @inbounds for i in 1:ncorr
        p[i] = v[i] / sqrt(sy_prod[i, i])
    end
    ldiv!(UpperTriangular(view(compact, 1:ncorr, 1:ncorr)), view(p, ncorr+1:2*ncorr))
    @inbounds for i in 1:ncorr
        p[i] = -p[i] / sqrt(sy_prod[i, i])
    end
    @inbounds for i in 1:ncorr
        p[i] += sum(sy_prod[k, i] * p[ncorr+k] / sy_prod[i, i] for k in i+1:ncorr; init=0.0)
    end
    return nothing
end

"""
    _initialize_active_set!(x, lb, ub, bounds, var_status)

Project `x` onto the feasible region and classify each variable as free or active.

Returns `(was_projected, has_bounds, boxed)` where:
- `was_projected`: whether any variable was projected
- `has_bounds`: whether any bounds exist
- `boxed`: whether all variables have both bounds

Byrd et al. (1995), Section 3.
"""
function _initialize_active_set!(x::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64},
    bounds::Vector{BoundType}, var_status::Vector{VarStatus})
    n = length(x)
    was_projected = false
    has_bounds = false
    boxed = true
    @inbounds for i in 1:n
        if _is_bounded(bounds[i])
            if _has_lower(bounds[i]) && x[i] <= lb[i]
                if x[i] < lb[i]
                    was_projected = true
                    x[i] = lb[i]
                end
            elseif _has_upper(bounds[i]) && x[i] >= ub[i]
                if x[i] > ub[i]
                    was_projected = true
                    x[i] = ub[i]
                end
            end
        end
    end
    @inbounds for i in 1:n
        if bounds[i] != BOTH_BOUNDS
            boxed = false
        end
        if bounds[i] == UNBOUNDED
            var_status[i] = VAR_UNBOUNDED
        else
            has_bounds = true
            if bounds[i] == BOTH_BOUNDS && ub[i] - lb[i] <= 0.0
                var_status[i] = VAR_FIXED
            else
                var_status[i] = VAR_FREE
            end
        end
    end
    return was_projected, has_bounds, boxed
end

"""
    _generalized_cauchy_point!(ws, x, lb, ub, grad, proj_grad_norm)

Compute the Generalized Cauchy Point (GCP) by piecewise-linear search along the
projected steepest descent path.

Throws `SingularFactorizationError` if the compact representation solve fails.
Returns `n_intervals`.

Byrd et al. (1995), Algorithm CP (Section 4).
"""
function _generalized_cauchy_point!(ws::LBFGSBWorkspace, x::Vector{Float64},
    lb::Vector{Float64}, ub::Vector{Float64},
    grad::Vector{Float64}, proj_grad_norm::Float64)

    n = ws.n
    m = ws.m
    ncorr = ws.n_corrections
    ring_head = ws.ring_head
    theta = ws.theta
    bounds = ws.bound_types
    var_status = ws.var_status
    sort_order = ws.sort_order
    breakpoint_times = ws.breakpoint_times
    search_dir = ws.search_dir
    cauchy_point = ws.cauchy_point
    p_work = ws.p_work
    c_work = ws.c_work
    breakpoint_work = ws.breakpoint_work
    v_work = ws.v_work
    grad_corr = ws.grad_corrections
    step_corr = ws.step_corrections

    if proj_grad_norm <= 0.0
        copyto!(cauchy_point, x)
        return 0
    end
    all_bounded_active = true
    n_unbounded_free = 0
    n_breakpoints = 0
    n_corrections_2x = 2 * ncorr
    quad_term = 0.0
    n_intervals = 1

    fill!(view(p_work, 1:n_corrections_2x), 0.0)

    dist_to_lower = 0.0
    dist_to_upper = 0.0
    @inbounds for i in 1:n
        neg_grad_i = -grad[i]
        if var_status[i] != VAR_FIXED && var_status[i] != VAR_UNBOUNDED
            if _has_lower(bounds[i])
                dist_to_lower = x[i] - lb[i]
            end
            if _has_upper(bounds[i])
                dist_to_upper = ub[i] - x[i]
            end
            at_lower_bound = _has_lower(bounds[i]) && dist_to_lower <= 0.0
            at_upper_bound = _has_upper(bounds[i]) && dist_to_upper <= 0.0
            var_status[i] = VAR_FREE
            if at_lower_bound
                if neg_grad_i <= 0.0
                    var_status[i] = VAR_LOWER_ACTIVE
                end
            elseif at_upper_bound
                if neg_grad_i >= 0.0
                    var_status[i] = VAR_UPPER_ACTIVE
                end
            else
                if abs(neg_grad_i) <= 0.0
                    var_status[i] = VAR_ZERO_GRADIENT
                end
            end
        end
        ring_ptr = ring_head
        if var_status[i] != VAR_FREE && var_status[i] != VAR_UNBOUNDED
            search_dir[i] = 0.0
        else
            search_dir[i] = neg_grad_i
            quad_term -= neg_grad_i * neg_grad_i
            for j in 1:ncorr
                p_work[j] += grad_corr[i, ring_ptr] * neg_grad_i
                p_work[ncorr+j] += step_corr[i, ring_ptr] * neg_grad_i
                ring_ptr = _next_ring(ring_ptr, m)
            end
            if _has_lower(bounds[i]) && bounds[i] != UNBOUNDED && neg_grad_i < 0.0
                n_breakpoints += 1
                sort_order[n_breakpoints] = i
                breakpoint_times[n_breakpoints] = dist_to_lower / (-neg_grad_i)
            elseif _has_upper(bounds[i]) && neg_grad_i > 0.0
                n_breakpoints += 1
                sort_order[n_breakpoints] = i
                breakpoint_times[n_breakpoints] = dist_to_upper / neg_grad_i
            else
                n_unbounded_free += 1
                sort_order[n + 1 - n_unbounded_free] = i
                if abs(neg_grad_i) > 0.0
                    all_bounded_active = false
                end
            end
        end
    end

    if theta != 1.0
        view(p_work, ncorr+1:n_corrections_2x) .*= theta
    end
    copyto!(cauchy_point, x)
    if n_breakpoints == 0 && n_unbounded_free == 0
        return 0
    end
    fill!(view(c_work, 1:n_corrections_2x), 0.0)

    deriv_sum = -theta * quad_term
    deriv_sum_init = deriv_sum
    if ncorr > 0
        _compact_representation_solve!(v_work, ws, p_work)
        deriv_sum -= dot(view(v_work, 1:n_corrections_2x), view(p_work, 1:n_corrections_2x))
    end
    step_to_min = -quad_term / deriv_sum
    time_sum = 0.0

    skip_xcp_update = false

    if n_breakpoints > 0
        # Sort breakpoints ascending by time
        if n_breakpoints > 1
            bp_perm = sortperm(view(breakpoint_times, 1:n_breakpoints))
            sorted_t = breakpoint_times[bp_perm]
            sorted_o = sort_order[bp_perm]
            for i in 1:n_breakpoints
                breakpoint_times[i] = sorted_t[i]
                sort_order[i] = sorted_o[i]
            end
        end
        n_remaining = n_breakpoints
        bp_idx = 1
        t_break = 0.0

        while true
            t_prev = t_break
            t_break = breakpoint_times[bp_idx]
            breakpoint_var = sort_order[bp_idx]
            bp_idx += 1
            dt_break = t_break - t_prev
            if step_to_min < dt_break
                break
            end
            time_sum += dt_break
            n_remaining -= 1
            d_at_break = search_dir[breakpoint_var]
            search_dir[breakpoint_var] = 0.0
            if d_at_break > 0.0
                z_at_break = ub[breakpoint_var] - x[breakpoint_var]
                cauchy_point[breakpoint_var] = ub[breakpoint_var]
                var_status[breakpoint_var] = VAR_UPPER_ACTIVE
            else
                z_at_break = lb[breakpoint_var] - x[breakpoint_var]
                cauchy_point[breakpoint_var] = lb[breakpoint_var]
                var_status[breakpoint_var] = VAR_LOWER_ACTIVE
            end
            if n_remaining == 0 && n_breakpoints == n
                step_to_min = dt_break
                skip_xcp_update = true
                break
            end
            n_intervals += 1
            d_at_break_sq = d_at_break * d_at_break
            quad_term += dt_break * deriv_sum + d_at_break_sq - theta * d_at_break * z_at_break
            deriv_sum -= theta * d_at_break_sq
            if ncorr > 0
                view(c_work, 1:n_corrections_2x) .+= dt_break .* view(p_work, 1:n_corrections_2x)
                ring_ptr = ring_head
                @inbounds for j in 1:ncorr
                    breakpoint_work[j] = grad_corr[breakpoint_var, ring_ptr]
                    breakpoint_work[ncorr+j] = theta * step_corr[breakpoint_var, ring_ptr]
                    ring_ptr = _next_ring(ring_ptr, m)
                end
                _compact_representation_solve!(v_work, ws, breakpoint_work)
                vw = view(v_work, 1:n_corrections_2x)
                v_dot_c = dot(view(c_work, 1:n_corrections_2x), vw)
                v_dot_p = dot(view(p_work, 1:n_corrections_2x), vw)
                v_dot_bp = dot(view(breakpoint_work, 1:n_corrections_2x), vw)
                view(p_work, 1:n_corrections_2x) .-= d_at_break .* view(breakpoint_work, 1:n_corrections_2x)
                quad_term += d_at_break * v_dot_c
                deriv_sum += 2.0 * d_at_break * v_dot_p - d_at_break_sq * v_dot_bp
            end
            deriv_sum = max(deriv_sum, eps(Float64) * deriv_sum_init)
            if n_remaining > 0
                step_to_min = -quad_term / deriv_sum
                continue
            elseif all_bounded_active
                quad_term = 0.0; deriv_sum = 0.0; step_to_min = 0.0
            else
                step_to_min = -quad_term / deriv_sum
            end
            break
        end
    end

    if !skip_xcp_update
        step_to_min = max(step_to_min, 0.0)
        time_sum += step_to_min
        view(cauchy_point, 1:n) .+= time_sum .* view(search_dir, 1:n)
    end

    if ncorr > 0
        view(c_work, 1:n_corrections_2x) .+= step_to_min .* view(p_work, 1:n_corrections_2x)
    end
    return n_intervals
end

"""
    _partition_free_active!(ws, iter)

Partition variables into free and active sets based on the current `var_status` classification.
Track which variables entered or left the free set since last iteration.
Updates `ws.n_free`, `ws.n_entering`, `ws.leaving_start` in place.

Returns `needs_reassembly::Bool` indicating whether the reduced Hessian needs reassembly.

Byrd et al. (1995), Section 5.
"""
function _partition_free_active!(ws::LBFGSBWorkspace, iter::Int)
    n = ws.n
    n_free_prev = ws.n_free
    free_vars = ws.free_vars
    var_status = ws.var_status
    sort_order = ws.sort_order

    n_entering = 0
    leaving_start = n + 1
    if iter > 0 && ws.has_bounds
        for i in 1:n_free_prev
            k = free_vars[i]
            if _is_active(var_status[k])
                leaving_start -= 1
                sort_order[leaving_start] = k
            end
        end
        for i in n_free_prev+1:n
            k = free_vars[i]
            if _is_free(var_status[k])
                n_entering += 1
                sort_order[n_entering] = k
            end
        end
    end
    needs_reassembly = (leaving_start <= n) || (n_entering > 0) || ws.did_update
    n_free = 0
    n_active = 0
    @inbounds for i in 1:n
        if _is_free(var_status[i])
            n_free += 1
            free_vars[n_free] = i
        else
            n_active += 1
            free_vars[n + 1 - n_active] = i
        end
    end
    ws.n_free = n_free
    ws.n_entering = n_entering
    ws.leaving_start = leaving_start
    return needs_reassembly
end


"""
    _assemble_reduced_hessian!(ws)

Assemble the reduced Hessian WN for the free variables from the L-BFGS
correction pairs stored in the workspace.

Throws `SingularFactorizationError` on non-positive definiteness.

Byrd et al. (1995), Section 5, Equations (5.4)–(5.11).
"""
function _assemble_reduced_hessian!(ws::LBFGSBWorkspace)
    n = ws.n
    m = ws.m
    n_free = ws.n_free
    n_entering = ws.n_entering
    leaving_start = ws.leaving_start
    free_vars = ws.free_vars
    sort_order = ws.sort_order
    ncorr = ws.n_corrections
    ring_head = ws.ring_head
    theta = ws.theta
    red_hess = ws.reduced_hessian
    parts = ws.hessian_parts
    step_corr = ws.step_corrections
    grad_corr = ws.grad_corrections
    sy_prod = ws.sy_products

    if ws.did_update
        if ws.update_count > m
            for row_y in 1:m-1
                row_s = m + row_y
                for i in 1:m-row_y
                    parts[row_y+i-1, row_y] = parts[row_y+i, row_y+1]
                    parts[row_s+i-1, row_s] = parts[row_s+i, row_s+1]
                end
                for i in 1:m-1
                    parts[m+i, row_y] = parts[m+i+1, row_y+1]
                end
            end
        end
        col_ptr = ws.tail_index
        row_ptr = ring_head
        for col_y in 1:ncorr
            col_s = m + col_y
            yy_sum = 0.0; ss_sum = 0.0; sy_sum = 0.0
            for k in 1:n_free
                var_idx = free_vars[k]
                yy_sum += grad_corr[var_idx, col_ptr] * grad_corr[var_idx, row_ptr]
            end
            for k in n_free+1:n
                var_idx = free_vars[k]
                ss_sum += step_corr[var_idx, col_ptr] * step_corr[var_idx, row_ptr]
                sy_sum += step_corr[var_idx, col_ptr] * grad_corr[var_idx, row_ptr]
            end
            parts[ncorr, col_y] = yy_sum
            parts[m + ncorr, col_s] = ss_sum
            parts[m + ncorr, col_y] = sy_sum
            row_ptr = _next_ring(row_ptr, m)
        end
        row_ptr = ws.tail_index
        col_ptr = ring_head
        for i in 1:ncorr
            sy_sum = 0.0
            for k in 1:n_free
                var_idx = free_vars[k]
                sy_sum += step_corr[var_idx, col_ptr] * grad_corr[var_idx, row_ptr]
            end
            col_ptr = _next_ring(col_ptr, m)
            parts[m + i, ncorr] = sy_sum
        end
        cols_to_update = ncorr - 1
    else
        cols_to_update = ncorr
    end

    col_ptr = ring_head
    for row_y in 1:cols_to_update
        row_s = m + row_y
        row_ptr = ring_head
        for col_y in 1:row_y
            col_s = m + col_y
            enter_yy = 0.0; enter_ss = 0.0; leave_yy = 0.0; leave_ss = 0.0
            for k in 1:n_entering
                var_idx = sort_order[k]
                enter_yy += grad_corr[var_idx, col_ptr] * grad_corr[var_idx, row_ptr]
                enter_ss += step_corr[var_idx, col_ptr] * step_corr[var_idx, row_ptr]
            end
            for k in leaving_start:n
                var_idx = sort_order[k]
                leave_yy += grad_corr[var_idx, col_ptr] * grad_corr[var_idx, row_ptr]
                leave_ss += step_corr[var_idx, col_ptr] * step_corr[var_idx, row_ptr]
            end
            parts[row_y, col_y] += enter_yy - leave_yy
            parts[row_s, col_s] += -enter_ss + leave_ss
            row_ptr = _next_ring(row_ptr, m)
        end
        col_ptr = _next_ring(col_ptr, m)
    end
    col_ptr = ring_head
    for row_s in m+1:m+cols_to_update
        row_ptr = ring_head
        for col_y in 1:cols_to_update
            enter_sy = 0.0; leave_sy = 0.0
            for k in 1:n_entering
                var_idx = sort_order[k]
                enter_sy += step_corr[var_idx, col_ptr] * grad_corr[var_idx, row_ptr]
            end
            for k in leaving_start:n
                var_idx = sort_order[k]
                leave_sy += step_corr[var_idx, col_ptr] * grad_corr[var_idx, row_ptr]
            end
            if row_s <= col_y + m
                parts[row_s, col_y] += enter_sy - leave_sy
            else
                parts[row_s, col_y] += -enter_sy + leave_sy
            end
            row_ptr = _next_ring(row_ptr, m)
        end
        col_ptr = _next_ring(col_ptr, m)
    end

    for row_y in 1:ncorr
        s_block_row = ncorr + row_y
        parts_s_row = m + row_y
        for col_y in 1:row_y
            s_block_col = ncorr + col_y
            parts_s_col = m + col_y
            red_hess[col_y, row_y] = parts[row_y, col_y] / theta
            red_hess[s_block_col, s_block_row] = parts[parts_s_row, parts_s_col] * theta
        end
        for col_y in 1:row_y-1
            red_hess[col_y, s_block_row] = -parts[parts_s_row, col_y]
        end
        for col_y in row_y:ncorr
            red_hess[col_y, s_block_row] = parts[parts_s_row, col_y]
        end
        red_hess[row_y, row_y] += sy_prod[row_y, row_y]
    end

    _cholesky!(red_hess, ncorr)
    n_corrections_2x = 2 * ncorr
    _check_diag_nonzero!(red_hess, ncorr)
    ldiv!(UpperTriangular(view(red_hess, 1:ncorr, 1:ncorr))', view(red_hess, 1:ncorr, ncorr+1:n_corrections_2x))
    LinearAlgebra.BLAS.syrk!('U', 'T', 1.0,
        view(red_hess, 1:ncorr, ncorr+1:n_corrections_2x),
        1.0, view(red_hess, ncorr+1:n_corrections_2x, ncorr+1:n_corrections_2x))
    _cholesky!(view(red_hess, ncorr+1:n_corrections_2x, ncorr+1:n_corrections_2x), ncorr)
    return nothing
end

"""
    _factorize_wt_matrix!(ws)

Form and factorize the upper triangular matrix Wθ used in the compact
L-BFGS representation.

Throws `SingularFactorizationError` on non-positive definiteness.

Byrd et al. (1995), Eq. (3.2).
"""
function _factorize_wt_matrix!(ws::LBFGSBWorkspace)
    ncorr = ws.n_corrections
    theta = ws.theta
    compact = ws.compact_form
    sy_prod = ws.sy_products
    ss_prod = ws.ss_products
    for j in 1:ncorr
        compact[1, j] = theta * ss_prod[1, j]
    end
    for i in 2:ncorr
        for j in i:ncorr
            compact[i, j] = sum(sy_prod[i, k] * sy_prod[j, k] / sy_prod[k, k] for k in 1:i-1) +
                            theta * ss_prod[i, j]
        end
    end
    _cholesky!(compact, ncorr)
    return nothing
end

"""
    _reduced_gradient!(ws, x, grad)

Compute the reduced gradient for the free variables, using the L-BFGS compact
representation to account for curvature information.

Throws `SingularFactorizationError` if the compact representation solve fails.

Byrd et al. (1995), Section 5.
"""
function _reduced_gradient!(ws::LBFGSBWorkspace, x::Vector{Float64}, grad::Vector{Float64})
    n = ws.n
    m = ws.m
    ncorr = ws.n_corrections
    ring_head = ws.ring_head
    theta = ws.theta
    n_free = ws.n_free
    free_vars = ws.free_vars
    cauchy_point = ws.cauchy_point
    reduced_grad = ws.reduced_grad
    step_corr = ws.step_corrections
    grad_corr = ws.grad_corrections

    if !ws.has_bounds && ncorr > 0
        view(reduced_grad, 1:n) .= .-view(grad, 1:n)
    else
        for i in 1:n_free
            k = free_vars[i]
            reduced_grad[i] = -theta * (cauchy_point[k] - x[k]) - grad[k]
        end
        _compact_representation_solve!(ws.solve_result, ws, ws.solve_input)
        ring_ptr = ring_head
        for j in 1:ncorr
            step_i = ws.solve_result[j]
            bound_dist = theta * ws.solve_result[ncorr+j]
            for i in 1:n_free
                k = free_vars[i]
                reduced_grad[i] += grad_corr[k, ring_ptr] * step_i + step_corr[k, ring_ptr] * bound_dist
            end
            ring_ptr = _next_ring(ring_ptr, m)
        end
    end
    return nothing
end

"""
    _subspace_minimization!(ws, lb, ub)

Perform subspace minimization within the free variable space, subject to
bound constraints. This finds the minimizer of the quadratic model restricted
to the subspace of free variables identified at the Generalized Cauchy Point.

Throws `SingularFactorizationError` if the triangular solve fails.

Byrd et al. (1995), Section 5.
"""
function _subspace_minimization!(ws::LBFGSBWorkspace, lb::Vector{Float64}, ub::Vector{Float64})
    n_free = ws.n_free
    if n_free <= 0
        return nothing
    end
    m = ws.m
    ncorr = ws.n_corrections
    ring_head = ws.ring_head
    theta = ws.theta
    free_vars = ws.free_vars
    bounds = ws.bound_types
    cauchy_point = ws.cauchy_point
    search_dir = ws.search_dir
    step_corr = ws.step_corrections
    grad_corr = ws.grad_corrections
    subspace_work = ws.subspace_work
    red_hess = ws.reduced_hessian

    ring_ptr = ring_head
    @inbounds for i in 1:ncorr
        grad_corr_dot = 0.0; step_corr_dot = 0.0
        for j in 1:n_free
            k = free_vars[j]
            grad_corr_dot += grad_corr[k, ring_ptr] * search_dir[j]
            step_corr_dot += step_corr[k, ring_ptr] * search_dir[j]
        end
        subspace_work[i] = grad_corr_dot
        subspace_work[ncorr+i] = theta * step_corr_dot
        ring_ptr = _next_ring(ring_ptr, m)
    end
    n_corrections_2x = 2 * ncorr
    _check_diag_nonzero!(red_hess, n_corrections_2x)
    ldiv!(UpperTriangular(view(red_hess, 1:n_corrections_2x, 1:n_corrections_2x))', view(subspace_work, 1:n_corrections_2x))
    view(subspace_work, 1:ncorr) .*= -1
    ldiv!(UpperTriangular(view(red_hess, 1:n_corrections_2x, 1:n_corrections_2x)), view(subspace_work, 1:n_corrections_2x))
    ring_ptr = ring_head
    @inbounds for j in 1:ncorr
        j_s = ncorr + j
        for i in 1:n_free
            k = free_vars[i]
            search_dir[i] += (grad_corr[k, ring_ptr] * subspace_work[j] / theta + step_corr[k, ring_ptr] * subspace_work[j_s])
        end
        ring_ptr = _next_ring(ring_ptr, m)
    end
    view(search_dir, 1:n_free) ./= theta

    alpha = 1.0
    candidate_alpha = alpha
    bound_hit_idx = 0
    @inbounds for i in 1:n_free
        k = free_vars[i]
        dir_k = search_dir[i]
        if _is_bounded(bounds[k])
            if dir_k < 0.0 && _has_lower(bounds[k])
                bound_dist = lb[k] - cauchy_point[k]
                if bound_dist >= 0.0
                    candidate_alpha = 0.0
                elseif dir_k * alpha < bound_dist
                    candidate_alpha = bound_dist / dir_k
                end
            elseif dir_k > 0.0 && _has_upper(bounds[k])
                bound_dist = ub[k] - cauchy_point[k]
                if bound_dist <= 0.0
                    candidate_alpha = 0.0
                elseif dir_k * alpha > bound_dist
                    candidate_alpha = bound_dist / dir_k
                end
            end
            if candidate_alpha < alpha
                alpha = candidate_alpha
                bound_hit_idx = i
            end
        end
    end
    if alpha < 1.0
        dir_k = search_dir[bound_hit_idx]
        k = free_vars[bound_hit_idx]
        if dir_k > 0.0
            cauchy_point[k] = ub[k]
            search_dir[bound_hit_idx] = 0.0
        elseif dir_k < 0.0
            cauchy_point[k] = lb[k]
            search_dir[bound_hit_idx] = 0.0
        end
    end
    @inbounds for i in 1:n_free
        cauchy_point[free_vars[i]] += alpha * search_dir[i]
    end
    return nothing
end

"""
    _update_lbfgs_matrices!(ws, step, grad_diff, rr, dr, dtd)

Update the L-BFGS correction matrices with a new step/gradient difference pair.
Mutates workspace fields (`tail_index`, `n_corrections`, `ring_head`, `theta`) directly.

Byrd et al. (1995), Section 3.
"""
function _update_lbfgs_matrices!(ws::LBFGSBWorkspace,
    step::Vector{Float64}, grad_diff::Vector{Float64},
    grad_diff_sq::Float64, step_grad_dot::Float64, step_sq::Float64)

    m = ws.m

    if ws.update_count <= m
        ws.n_corrections = ws.update_count
        ws.tail_index = (ws.ring_head + ws.update_count - 2) % m + 1  # newest correction position
    else
        ws.tail_index = _next_ring(ws.tail_index, m)
        ws.ring_head = _next_ring(ws.ring_head, m)
    end
    ncorr = ws.n_corrections
    tail_idx = ws.tail_index
    ring_head = ws.ring_head
    step_corr = ws.step_corrections
    grad_corr = ws.grad_corrections
    sy_prod = ws.sy_products
    ss_prod = ws.ss_products

    step_corr[:, tail_idx] .= step
    grad_corr[:, tail_idx] .= grad_diff
    ws.theta = grad_diff_sq / step_grad_dot
    if ws.update_count > m
        for j in 1:ncorr-1
            for i in 1:j
                ss_prod[i, j] = ss_prod[i+1, j+1]
            end
            for i in 1:ncorr-j
                sy_prod[j+i-1, j] = sy_prod[j+i, j+1]
            end
        end
    end
    ring_ptr = ring_head
    for j in 1:ncorr-1
        sy_prod[ncorr, j] = dot(view(step_corr, :, tail_idx), view(grad_corr, :, ring_ptr))
        ss_prod[j, ncorr] = dot(view(step_corr, :, ring_ptr), view(step_corr, :, tail_idx))
        ring_ptr = _next_ring(ring_ptr, m)
    end
    ss_prod[ncorr, ncorr] = step_sq
    sy_prod[ncorr, ncorr] = step_grad_dot
    return nothing
end



"""
    lbfgsb(f, g, x0; mask=trues(length(x0)), lower=nothing, upper=nothing, options=LBFGSBOptions())

Minimize a function with box constraints using the L-BFGS-B algorithm.

L-BFGS-B is a limited-memory variant of BFGS that supports bound constraints on
variables. It stores only the last `memory_size` iterations of curvature information, making
it memory-efficient for large-scale problems. The algorithm uses a Generalized Cauchy
Point computation and subspace minimization within the free variable space.

# Arguments

- `f::Function`: Objective function, signature `f(x)` → scalar.
- `g::Function`: Gradient function, signature `g(x)` → gradient vector.
- `x0::Vector{Float64}`: Initial parameter vector.
- `mask`: Logical mask; frozen variables have bounds set to `lower[i]=upper[i]=x0[i]`.
- `lower`, `upper`: Optional bound vectors (`nothing` = unbounded).
- `options::LBFGSBOptions`: Algorithm parameters (`memory_size`, `ftol_factor`, `pg_tol`, `maxit`, `print_level`).

# Returns

Named tuple `(x_opt, f_opt, n_iter, fail, fn_evals, gr_evals, message)`.

# References

- Byrd, R. H., Lu, P., Nocedal, J. & Zhu, C. (1995). A limited memory algorithm for
  bound constrained optimization. *SIAM J. Scientific Computing*, 16(5), 1190–1208.
- Zhu, C., Byrd, R. H., Lu, P. & Nocedal, J. (1997). Algorithm 778: L-BFGS-B.
  *ACM Trans. Math. Software*, 23(4), 550–560.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Algorithms 3.5/3.6.
  Springer.
"""
function lbfgsb(f::Function, g::Function, x0::Vector{Float64};
    mask=trues(length(x0)),
    lower::Union{Nothing,Vector{Float64}}=nothing,
    upper::Union{Nothing,Vector{Float64}}=nothing,
    options::LBFGSBOptions=LBFGSBOptions())

    n = length(x0)
    m = options.memory_size
    if length(mask) != n
        throw(ArgumentError("mask length must equal x0 length"))
    end

    lb = lower === nothing ? fill(-Inf, n) : copy(lower)
    ub = upper === nothing ? fill(+Inf, n) : copy(upper)
    @inbounds for i in 1:n
        if !mask[i]
            lb[i] = x0[i]
            ub[i] = x0[i]
        end
    end

    @inbounds for i in 1:n
        if lb[i] > ub[i]
            return (x_opt=copy(x0), f_opt=Inf, n_iter=0,
                fail=52, fn_evals=0, gr_evals=0,
                message="Error: no feasible solution (lower > upper)")
        end
    end

    wk = LBFGSBWorkspace(n, m, lb, ub)

    x = copy(x0)
    _, wk.has_bounds, wk.boxed = _initialize_active_set!(x, lb, ub, wk.bound_types, wk.var_status)

    fx = f(x)
    fn_evals = 1
    if !isfinite(fx)
        return (x_opt=x, f_opt=fx, n_iter=0,
            fail=52, fn_evals=fn_evals, gr_evals=0,
            message="Error: objective function returned non-finite value")
    end

    grad = copy(g(x))
    gr_evals = 1

    if !all(isfinite, grad)
        return (x_opt=x, f_opt=fx, n_iter=0,
            fail=52, fn_evals=fn_evals, gr_evals=gr_evals,
            message="Error: gradient contains non-finite values")
    end

    proj_grad_norm = _proj_grad!(wk.proj_grad, x, grad, lb, ub, wk.bound_types)

    if options.print_level > 0
        println("At iterate     0  f= ", fx, "  |proj g|= ", proj_grad_norm)
    end
    if proj_grad_norm <= options.pg_tol
        return (x_opt=x, f_opt=fx, n_iter=0,
            fail=0, fn_evals=fn_evals, gr_evals=gr_evals,
            message="Converged: projected gradient norm below tolerance")
    end

    f_tol = options.ftol_factor * eps(Float64)
    fail = 1
    message = nothing
    iter = 0
    search_dir = wk.search_dir
    cauchy_point = wk.cauchy_point

    while true
        iter += 1

        needs_reassembly = false
        if !wk.has_bounds && wk.n_corrections > 0
            copyto!(cauchy_point, x)
            needs_reassembly = wk.did_update
        else
            try
                _generalized_cauchy_point!(wk, x, lb, ub, grad, proj_grad_norm)
            catch e
                e isa SingularFactorizationError || rethrow()
                if options.print_level > 0
                    println("Singular triangular system in GCP computation; refreshing memory.")
                end
                _reset_memory!(wk)
                continue
            end
            needs_reassembly = _partition_free_active!(wk, iter - 1)
        end

        if wk.n_free != 0 && wk.n_corrections != 0
            try
                if needs_reassembly
                    _assemble_reduced_hessian!(wk)
                end
                n_corrections_2x = 2 * wk.n_corrections
                view(wk.solve_input, 1:n_corrections_2x) .= view(wk.c_work, 1:n_corrections_2x)
                _reduced_gradient!(wk, x, grad)
                view(search_dir, 1:wk.n_free) .= view(wk.reduced_grad, 1:wk.n_free)
                _subspace_minimization!(wk, lb, ub)
            catch e
                e isa SingularFactorizationError || rethrow()
                if options.print_level > 0
                    println("Singular triangular system in subspace minimization; refreshing memory.")
                end
                _reset_memory!(wk)
                continue
            end
        end

        search_dir .= cauchy_point .- x

        dnorm_sq = dot(search_dir, search_dir)
        if dnorm_sq < 1e-32
            fail = 0
            message = "Converged: search direction is effectively zero"
            break
        end

        x_prev = copy(x)
        grad_prev = copy(grad)
        f_prev = fx

        max_step = if !wk.has_bounds
            1e10
        elseif iter == 1
            1.0
        else
            min(_max_feasible_step(x, search_dir, lb, ub), 1e10)
        end

        ok, _, fnew, gnew, xnew, fe, ge = _strong_wolfe_line_search!(x, fx, grad, search_dir, lb, ub,
            f, g; iter=iter, boxed=wk.boxed, max_step=max_step)
        fn_evals += fe
        gr_evals += ge

        if !ok
            x .= x_prev
            grad .= grad_prev
            fx = f_prev
            if wk.n_corrections == 0
                fail = 52
                message = "Error: line search failed to find acceptable step"
                break
            else
                if options.print_level > 0
                    println("Bad direction in line search; refreshing memory.")
                end
                _reset_memory!(wk)
                continue
            end
        end

        if !isfinite(fnew)
            x .= x_prev
            grad .= grad_prev
            fx = f_prev
            fail = 52
            message = "Error: objective function returned non-finite value"
            break
        end

        x .= xnew
        fx = fnew
        grad .= copy(gnew)

        if !all(isfinite, grad)
            fail = 52
            message = "Error: gradient contains non-finite values"
            break
        end

        proj_grad_norm = _proj_grad!(wk.proj_grad, x, grad, lb, ub, wk.bound_types)

        if options.print_level > 0
            println("At iterate ", iter, "  f= ", fx, "  |proj g|= ", proj_grad_norm)
        end

        if iter > options.maxit
            break
        end

        if proj_grad_norm <= options.pg_tol
            fail = 0
            message = "Converged: projected gradient norm below tolerance"
            break
        end
        max_f = max(abs(f_prev), abs(fx), 1.0)
        if f_prev - fx <= f_tol * max_f
            fail = 0
            message = "Converged: relative function reduction below tolerance"
            break
        end

        search_dir .= x .- x_prev
        wk.reduced_grad .= grad .- grad_prev
        grad_diff_sq = dot(wk.reduced_grad, wk.reduced_grad)
        step_grad_dot = dot(wk.reduced_grad, search_dir)
        step_sq = dot(search_dir, search_dir)
        descent_magnitude = -dot(grad_prev, search_dir)

        if step_grad_dot <= eps(Float64) * descent_magnitude
            wk.did_update = false
        else
            wk.did_update = true
            wk.update_count += 1
            _update_lbfgs_matrices!(wk, search_dir, wk.reduced_grad, grad_diff_sq, step_grad_dot, step_sq)
            try
                _factorize_wt_matrix!(wk)
            catch e
                e isa SingularFactorizationError || rethrow()
                if options.print_level > 0
                    println("Nonpositive definiteness in compact form factorization; refreshing memory.")
                end
                _reset_memory!(wk)
            end
        end
    end

    return (x_opt=x, f_opt=fx, n_iter=iter,
        fail=fail, fn_evals=fn_evals, gr_evals=gr_evals,
        message=message)
end
