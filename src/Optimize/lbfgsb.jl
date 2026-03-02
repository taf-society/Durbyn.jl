import LinearAlgebra
using LinearAlgebra: dot, norm

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
algorithm. Replaces the Fortran-style pattern of passing 20+ separate arrays
as positional arguments to every internal function.
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
    solve_result::Vector{Float64}      # compact solve output (replaces wa[1:2m])
    solve_input::Vector{Float64}       # compact solve input (replaces wa[2m+1:4m])

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
    nbd = Vector{BoundType}(undef, n)
    @inbounds for i in 1:n
        if isfinite(lb[i]) && isfinite(ub[i])
            nbd[i] = BOTH_BOUNDS
        elseif isfinite(lb[i])
            nbd[i] = LOWER_ONLY
        elseif isfinite(ub[i])
            nbd[i] = UPPER_ONLY
        else
            nbd[i] = UNBOUNDED
        end
    end
    return nbd
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
    _proj_grad!(pg, x, grad, lb, ub, nbd)

Compute the projected gradient and return its infinity norm.
The projected gradient measures first-order optimality for box-constrained problems.

Byrd et al. (1995), Eq. (2.1).
"""
function _proj_grad!(pg::Vector{Float64}, x::Vector{Float64}, grad::Vector{Float64},
    lb::Vector{Float64}, ub::Vector{Float64}, nbd::Vector{BoundType})
    @inbounds for i in eachindex(x)
        gi = grad[i]
        bi = nbd[i]
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

    proj_grad_norm = 0.0
    @inbounds for i in eachindex(pg)
        v = abs(pg[i])
        if isnan(v)
            return NaN
        end
        if v > proj_grad_norm
            proj_grad_norm = v
        end
    end
    return proj_grad_norm
end


function _cholesky!(A::AbstractMatrix{Float64}, n::Int)
    @inbounds for j in 1:n
        s = 0.0
        for k in 1:j-1
            t = A[k,j]
            for i in 1:k-1
                t -= A[i,k] * A[i,j]
            end
            t /= A[k,k]
            A[k,j] = t
            s += t * t
        end
        s = A[j,j] - s
        if s <= 0.0
            throw(SingularFactorizationError())
        end
        A[j,j] = sqrt(s)
    end
    return nothing
end

function _triangular_solve!(A::AbstractMatrix{Float64}, r::Int, n::Int, b::AbstractVector{Float64}, boff::Int)
    @inbounds for j in n:-1:1
        if A[r+j-1, r+j-1] == 0.0
            throw(SingularFactorizationError())
        end
        b[boff+j] /= A[r+j-1, r+j-1]
        for i in 1:j-1
            b[boff+i] -= A[r+i-1, r+j-1] * b[boff+j]
        end
    end
    return nothing
end

function _triangular_solve_t!(A::AbstractMatrix{Float64}, r::Int, n::Int, b::AbstractVector{Float64}, boff::Int)
    @inbounds for j in 1:n
        s = b[boff+j]
        for i in 1:j-1
            s -= A[r+i-1, r+j-1] * b[boff+i]
        end
        if A[r+j-1, r+j-1] == 0.0
            throw(SingularFactorizationError())
        end
        b[boff+j] = s / A[r+j-1, r+j-1]
    end
    return nothing
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
    col = ws.n_corrections
    if col == 0
        return nothing
    end
    sy = ws.sy_products
    wt = ws.compact_form
    p[col+1] = v[col+1]
    @inbounds for i in 2:col
        s = 0.0
        for k in 1:i-1
            s += sy[i, k] * v[k] / sy[k, k]
        end
        p[col+i] = v[col+i] + s
    end
    _triangular_solve_t!(wt, 1, col, p, col)
    @inbounds for i in 1:col
        p[i] = v[i] / sqrt(sy[i, i])
    end
    _triangular_solve!(wt, 1, col, p, col)
    @inbounds for i in 1:col
        p[i] = -p[i] / sqrt(sy[i, i])
    end
    @inbounds for i in 1:col
        s = 0.0
        for k in i+1:col
            s += sy[k, i] * p[col+k] / sy[i, i]
        end
        p[i] += s
    end
    return nothing
end

function _heap_sort!(t::Vector{Float64}, iorder::Vector{Int}, n::Int, needs_heapify::Bool)
    if needs_heapify
        @inbounds for k in 2:n
            val = t[k]
            key_in = iorder[k]
            i = k
            while i > 1
                j = div(i, 2)
                if val < t[j]
                    t[i] = t[j]
                    iorder[i] = iorder[j]
                    i = j
                else
                    break
                end
            end
            t[i] = val
            iorder[i] = key_in
        end
    end
    if n > 1
        out = t[1]
        key_out = iorder[1]
        val = t[n]
        key_in = iorder[n]
        i = 1
        @inbounds while true
            j = 2 * i
            if j > n - 1
                break
            end
            if t[j+1] < t[j]
                j += 1
            end
            if t[j] < val
                t[i] = t[j]
                iorder[i] = iorder[j]
                i = j
            else
                break
            end
        end
        t[i] = val
        iorder[i] = key_in
        t[n] = out
        iorder[n] = key_out
    end
end


"""
    _initialize_active_set!(x, lb, ub, nbd, var_status, n)

Project `x` onto the feasible region and classify each variable as free or active.

Returns `(was_projected, has_bounds, boxed)` where:
- `was_projected`: whether any variable was projected
- `has_bounds`: whether any bounds exist
- `boxed`: whether all variables have both bounds

Byrd et al. (1995), Section 3.
"""
function _initialize_active_set!(x::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64},
    nbd::Vector{BoundType}, var_status::Vector{VarStatus}, n::Int)
    was_projected = false
    has_bounds = false
    boxed = true
    @inbounds for i in 1:n
        if _is_bounded(nbd[i])
            if _has_lower(nbd[i]) && x[i] <= lb[i]
                if x[i] < lb[i]
                    was_projected = true
                    x[i] = lb[i]
                end
            elseif _has_upper(nbd[i]) && x[i] >= ub[i]
                if x[i] > ub[i]
                    was_projected = true
                    x[i] = ub[i]
                end
            end
        end
    end
    @inbounds for i in 1:n
        if nbd[i] != BOTH_BOUNDS
            boxed = false
        end
        if nbd[i] == UNBOUNDED
            var_status[i] = VAR_UNBOUNDED
        else
            has_bounds = true
            if nbd[i] == BOTH_BOUNDS && ub[i] - lb[i] <= 0.0
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
    col = ws.n_corrections
    ring_head = ws.ring_head
    theta = ws.theta
    nbd = ws.bound_types
    var_status = ws.var_status
    sort_order = ws.sort_order
    breakpoint_times = ws.breakpoint_times
    search_dir = ws.search_dir
    cauchy_point = ws.cauchy_point
    p = ws.p_work
    c = ws.c_work
    breakpoint_work = ws.breakpoint_work
    v = ws.v_work
    wy = ws.grad_corrections
    step_corr = ws.step_corrections

    if proj_grad_norm <= 0.0
        copyto!(cauchy_point, x)
        return 0
    end
    all_bounded_active = true
    n_unbounded_free = n + 1
    n_breakpoints = 0
    min_break_idx = 0
    min_break_time = 0.0
    n_corrections_2x = 2 * col
    quad_term = 0.0
    n_intervals = 1

    @inbounds for i in 1:n_corrections_2x
        p[i] = 0.0
    end

    dist_to_lower = 0.0
    dist_to_upper = 0.0
    @inbounds for i in 1:n
        neg_grad_i = -grad[i]
        if var_status[i] != VAR_FIXED && var_status[i] != VAR_UNBOUNDED
            if _has_lower(nbd[i])
                dist_to_lower = x[i] - lb[i]
            end
            if _has_upper(nbd[i])
                dist_to_upper = ub[i] - x[i]
            end
            xlower = _has_lower(nbd[i]) && dist_to_lower <= 0.0
            xupper = _has_upper(nbd[i]) && dist_to_upper <= 0.0
            var_status[i] = VAR_FREE
            if xlower
                if neg_grad_i <= 0.0
                    var_status[i] = VAR_LOWER_ACTIVE
                end
            elseif xupper
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
            for j in 1:col
                p[j] += wy[i, ring_ptr] * neg_grad_i
                p[col+j] += step_corr[i, ring_ptr] * neg_grad_i
                ring_ptr = _next_ring(ring_ptr, m)
            end
            if _has_lower(nbd[i]) && nbd[i] != UNBOUNDED && neg_grad_i < 0.0
                n_breakpoints += 1
                sort_order[n_breakpoints] = i
                breakpoint_times[n_breakpoints] = dist_to_lower / (-neg_grad_i)
                if n_breakpoints == 1 || breakpoint_times[n_breakpoints] < min_break_time
                    min_break_time = breakpoint_times[n_breakpoints]
                    min_break_idx = n_breakpoints
                end
            elseif _has_upper(nbd[i]) && neg_grad_i > 0.0
                n_breakpoints += 1
                sort_order[n_breakpoints] = i
                breakpoint_times[n_breakpoints] = dist_to_upper / neg_grad_i
                if n_breakpoints == 1 || breakpoint_times[n_breakpoints] < min_break_time
                    min_break_time = breakpoint_times[n_breakpoints]
                    min_break_idx = n_breakpoints
                end
            else
                n_unbounded_free -= 1
                sort_order[n_unbounded_free] = i
                if abs(neg_grad_i) > 0.0
                    all_bounded_active = false
                end
            end
        end
    end

    if theta != 1.0
        @inbounds for j in col+1:n_corrections_2x
            p[j] *= theta
        end
    end
    copyto!(cauchy_point, x)
    if n_breakpoints == 0 && n_unbounded_free == n + 1
        return 0
    end
    @inbounds for j in 1:n_corrections_2x
        c[j] = 0.0
    end

    deriv_sum = -theta * quad_term
    deriv_sum_org = deriv_sum
    if col > 0
        _compact_representation_solve!(v, ws, p)
        s = 0.0
        @inbounds for i in 1:n_corrections_2x
            s += v[i] * p[i]
        end
        deriv_sum -= s
    end
    dtm = -quad_term / deriv_sum
    tsum = 0.0

    skip_xcp_update = false

    if n_breakpoints > 0
        nleft = n_breakpoints
        cauchyiter = 1
        tj = 0.0

        while true
            tj0 = tj
            if cauchyiter == 1
                tj = min_break_time
                breakpoint_var = sort_order[min_break_idx]
            else
                if cauchyiter == 2
                    if min_break_idx != n_breakpoints
                        breakpoint_times[min_break_idx] = breakpoint_times[n_breakpoints]
                        sort_order[min_break_idx] = sort_order[n_breakpoints]
                    end
                end
                _heap_sort!(breakpoint_times, sort_order, nleft, cauchyiter == 2)
                tj = breakpoint_times[nleft]
                breakpoint_var = sort_order[nleft]
            end
            dt_break = tj - tj0
            if dtm < dt_break
                break
            end
            tsum += dt_break
            nleft -= 1
            cauchyiter += 1
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
            if nleft == 0 && n_breakpoints == n
                dtm = dt_break
                skip_xcp_update = true
                break
            end
            n_intervals += 1
            d_at_break_sq = d_at_break * d_at_break
            quad_term += dt_break * deriv_sum + d_at_break_sq - theta * d_at_break * z_at_break
            deriv_sum -= theta * d_at_break_sq
            if col > 0
                @inbounds for j in 1:n_corrections_2x
                    c[j] += dt_break * p[j]
                end
                ring_ptr = ring_head
                @inbounds for j in 1:col
                    breakpoint_work[j] = wy[breakpoint_var, ring_ptr]
                    breakpoint_work[col+j] = theta * step_corr[breakpoint_var, ring_ptr]
                    ring_ptr = _next_ring(ring_ptr, m)
                end
                _compact_representation_solve!(v, ws, breakpoint_work)
                wy_c = 0.0; wy_p = 0.0; ws_delta = 0.0
                @inbounds for j in 1:n_corrections_2x
                    wy_c += c[j] * v[j]
                    wy_p += p[j] * v[j]
                    ws_delta += breakpoint_work[j] * v[j]
                end
                @inbounds for j in 1:n_corrections_2x
                    p[j] -= d_at_break * breakpoint_work[j]
                end
                quad_term += d_at_break * wy_c
                deriv_sum += 2.0 * d_at_break * wy_p - d_at_break_sq * ws_delta
            end
            deriv_sum = max(deriv_sum, eps(Float64) * deriv_sum_org)
            if nleft > 0
                dtm = -quad_term / deriv_sum
                continue
            elseif all_bounded_active
                quad_term = 0.0; deriv_sum = 0.0; dtm = 0.0
            else
                dtm = -quad_term / deriv_sum
            end
            break
        end
    end

    if !skip_xcp_update
        dtm = max(dtm, 0.0)
        tsum += dtm
        @inbounds for i in 1:n
            cauchy_point[i] += tsum * search_dir[i]
        end
    end

    if col > 0
        @inbounds for j in 1:n_corrections_2x
            c[j] += dtm * p[j]
        end
    end
    return n_intervals
end

"""
    _partition_free_active!(ws, iter)

Partition variables into free and active sets based on the current `var_status` classification.
Track which variables entered or left the free set since last iteration.
Updates `ws.n_free`, `ws.n_entering`, `ws.leaving_start` in place.

Returns `wrk::Bool` indicating whether the reduced Hessian needs reassembly.

Byrd et al. (1995), Section 5.
"""
function _partition_free_active!(ws::LBFGSBWorkspace, iter::Int)
    n = ws.n
    nfree_prev = ws.n_free
    free_vars = ws.free_vars
    var_status = ws.var_status
    sort_order = ws.sort_order

    n_entering = 0
    leaving_start = n + 1
    if iter > 0 && ws.has_bounds
        for i in 1:nfree_prev
            k = free_vars[i]
            if _is_active(var_status[k])
                leaving_start -= 1
                sort_order[leaving_start] = k
            end
        end
        for i in nfree_prev+1:n
            k = free_vars[i]
            if _is_free(var_status[k])
                n_entering += 1
                sort_order[n_entering] = k
            end
        end
    end
    wrk = (leaving_start < n + 1) || (n_entering > 0) || ws.did_update
    n_free = 0
    iact = n + 1
    @inbounds for i in 1:n
        if _is_free(var_status[i])
            n_free += 1
            free_vars[n_free] = i
        else
            iact -= 1
            free_vars[iact] = i
        end
    end
    ws.n_free = n_free
    ws.n_entering = n_entering
    ws.leaving_start = leaving_start
    return wrk
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
    col = ws.n_corrections
    ring_head = ws.ring_head
    theta = ws.theta
    wn = ws.reduced_hessian
    hp = ws.hessian_parts
    step_corr = ws.step_corrections
    wy = ws.grad_corrections
    sy = ws.sy_products

    if ws.did_update
        if ws.update_count > m
            for row_y in 1:m-1
                row_s = m + row_y
                for i in 1:m-row_y
                    hp[row_y+i-1, row_y] = hp[row_y+i, row_y+1]
                    hp[row_s+i-1, row_s] = hp[row_s+i, row_s+1]
                end
                for i in 1:m-1
                    hp[m+i, row_y] = hp[m+i+1, row_y+1]
                end
            end
        end
        col_ptr = ((ring_head + col - 2) % m) + 1
        row_ptr = ring_head
        for col_y in 1:col
            col_s = m + col_y
            yy_sum = 0.0; ss_sum = 0.0; sy_sum = 0.0
            for k in 1:n_free
                var_idx = free_vars[k]
                yy_sum += wy[var_idx, col_ptr] * wy[var_idx, row_ptr]
            end
            for k in n_free+1:n
                var_idx = free_vars[k]
                ss_sum += step_corr[var_idx, col_ptr] * step_corr[var_idx, row_ptr]
                sy_sum += step_corr[var_idx, col_ptr] * wy[var_idx, row_ptr]
            end
            hp[col, col_y] = yy_sum
            hp[m + col, col_s] = ss_sum
            hp[m + col, col_y] = sy_sum
            row_ptr = _next_ring(row_ptr, m)
        end
        row_ptr = ((ring_head + col - 2) % m) + 1
        col_ptr = ring_head
        for i in 1:col
            sy_sum = 0.0
            for k in 1:n_free
                var_idx = free_vars[k]
                sy_sum += step_corr[var_idx, col_ptr] * wy[var_idx, row_ptr]
            end
            col_ptr = _next_ring(col_ptr, m)
            hp[m + i, col] = sy_sum
        end
        upcl = col - 1
    else
        upcl = col
    end

    col_ptr = ring_head
    for row_y in 1:upcl
        row_s = m + row_y
        row_ptr = ring_head
        for col_y in 1:row_y
            col_s = m + col_y
            enter_yy = 0.0; enter_ss = 0.0; leave_yy = 0.0; leave_ss = 0.0
            for k in 1:n_entering
                var_idx = sort_order[k]
                enter_yy += wy[var_idx, col_ptr] * wy[var_idx, row_ptr]
                enter_ss += step_corr[var_idx, col_ptr] * step_corr[var_idx, row_ptr]
            end
            for k in leaving_start:n
                var_idx = sort_order[k]
                leave_yy += wy[var_idx, col_ptr] * wy[var_idx, row_ptr]
                leave_ss += step_corr[var_idx, col_ptr] * step_corr[var_idx, row_ptr]
            end
            hp[row_y, col_y] += enter_yy - leave_yy
            hp[row_s, col_s] += -enter_ss + leave_ss
            row_ptr = _next_ring(row_ptr, m)
        end
        col_ptr = _next_ring(col_ptr, m)
    end
    col_ptr = ring_head
    for row_s in m+1:m+upcl
        row_ptr = ring_head
        for col_y in 1:upcl
            enter_sy = 0.0; leave_sy = 0.0
            for k in 1:n_entering
                var_idx = sort_order[k]
                enter_sy += step_corr[var_idx, col_ptr] * wy[var_idx, row_ptr]
            end
            for k in leaving_start:n
                var_idx = sort_order[k]
                leave_sy += step_corr[var_idx, col_ptr] * wy[var_idx, row_ptr]
            end
            if row_s <= col_y + m
                hp[row_s, col_y] += enter_sy - leave_sy
            else
                hp[row_s, col_y] += -enter_sy + leave_sy
            end
            row_ptr = _next_ring(row_ptr, m)
        end
        col_ptr = _next_ring(col_ptr, m)
    end

    for row_y in 1:col
        wn_s = col + row_y
        hp_s = m + row_y
        for col_y in 1:row_y
            wn_js = col + col_y
            hp_js = m + col_y
            wn[col_y, row_y] = hp[row_y, col_y] / theta
            wn[wn_js, wn_s] = hp[hp_s, hp_js] * theta
        end
        for col_y in 1:row_y-1
            wn[col_y, wn_s] = -hp[hp_s, col_y]
        end
        for col_y in row_y:col
            wn[col_y, wn_s] = hp[hp_s, col_y]
        end
        wn[row_y, row_y] += sy[row_y, row_y]
    end

    _cholesky!(wn, col)
    n_corrections_2x = 2 * col
    for js in col+1:n_corrections_2x
        _triangular_solve_t!(wn, 1, col, view(wn, :, js), 0)
    end
    for is_ in col+1:n_corrections_2x
        for js in is_:n_corrections_2x
            s = 0.0
            @inbounds for k in 1:col
                s += wn[k, is_] * wn[k, js]
            end
            wn[is_, js] += s
        end
    end
    _cholesky!(view(wn, col+1:n_corrections_2x, col+1:n_corrections_2x), col)
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
    col = ws.n_corrections
    theta = ws.theta
    wt = ws.compact_form
    sy = ws.sy_products
    ss = ws.ss_products
    for j in 1:col
        wt[1, j] = theta * ss[1, j]
    end
    for i in 2:col
        for j in i:col
            k1 = min(i, j) - 1
            accum = 0.0
            for k in 1:k1
                accum += sy[i, k] * sy[j, k] / sy[k, k]
            end
            wt[i, j] = accum + theta * ss[i, j]
        end
    end
    _cholesky!(wt, col)
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
    col = ws.n_corrections
    ring_head = ws.ring_head
    theta = ws.theta
    n_free = ws.n_free
    free_vars = ws.free_vars
    cauchy_point = ws.cauchy_point
    reduced_grad = ws.reduced_grad
    step_corr = ws.step_corrections
    wy = ws.grad_corrections

    if !ws.has_bounds && col > 0
        for i in 1:n
            reduced_grad[i] = -grad[i]
        end
    else
        for i in 1:n_free
            k = free_vars[i]
            reduced_grad[i] = -theta * (cauchy_point[k] - x[k]) - grad[k]
        end
        _compact_representation_solve!(ws.solve_result, ws, ws.solve_input)
        ring_ptr = ring_head
        for j in 1:col
            step_i = ws.solve_result[j]
            bound_dist = theta * ws.solve_result[col+j]
            for i in 1:n_free
                k = free_vars[i]
                reduced_grad[i] += wy[k, ring_ptr] * step_i + step_corr[k, ring_ptr] * bound_dist
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
    col = ws.n_corrections
    ring_head = ws.ring_head
    theta = ws.theta
    free_vars = ws.free_vars
    nbd = ws.bound_types
    cauchy_point = ws.cauchy_point
    search_dir = ws.search_dir
    step_corr = ws.step_corrections
    wy = ws.grad_corrections
    subspace_work = ws.subspace_work
    wn = ws.reduced_hessian

    ring_ptr = ring_head
    @inbounds for i in 1:col
        wy_sum = 0.0; ws_sum = 0.0
        for j in 1:n_free
            k = free_vars[j]
            wy_sum += wy[k, ring_ptr] * search_dir[j]
            ws_sum += step_corr[k, ring_ptr] * search_dir[j]
        end
        subspace_work[i] = wy_sum
        subspace_work[col+i] = theta * ws_sum
        ring_ptr = _next_ring(ring_ptr, m)
    end
    n_corrections_2x = 2 * col
    _triangular_solve_t!(wn, 1, n_corrections_2x, subspace_work, 0)
    @inbounds for i in 1:col
        subspace_work[i] = -subspace_work[i]
    end
    _triangular_solve!(wn, 1, n_corrections_2x, subspace_work, 0)
    ring_ptr = ring_head
    @inbounds for jy in 1:col
        js = col + jy
        for i in 1:n_free
            k = free_vars[i]
            search_dir[i] += (wy[k, ring_ptr] * subspace_work[jy] / theta + step_corr[k, ring_ptr] * subspace_work[js])
        end
        ring_ptr = _next_ring(ring_ptr, m)
    end
    @inbounds for i in 1:n_free
        search_dir[i] /= theta
    end

    alpha = 1.0
    candidate_alpha = alpha
    bound_hit_idx = 0
    @inbounds for i in 1:n_free
        k = free_vars[i]
        dir_k = search_dir[i]
        if _is_bounded(nbd[k])
            if dir_k < 0.0 && _has_lower(nbd[k])
                bound_dist = lb[k] - cauchy_point[k]
                if bound_dist >= 0.0
                    candidate_alpha = 0.0
                elseif dir_k * alpha < bound_dist
                    candidate_alpha = bound_dist / dir_k
                end
            elseif dir_k > 0.0 && _has_upper(nbd[k])
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
    rr::Float64, dr::Float64, dtd::Float64)

    n = ws.n
    m = ws.m

    if ws.update_count <= m
        ws.n_corrections = ws.update_count
        ws.tail_index = (ws.ring_head + ws.update_count - 2) % m + 1
    else
        ws.tail_index = _next_ring(ws.tail_index, m)
        ws.ring_head = _next_ring(ws.ring_head, m)
    end
    col = ws.n_corrections
    tail_idx = ws.tail_index
    ring_head = ws.ring_head
    step_corr = ws.step_corrections
    wy = ws.grad_corrections
    sy = ws.sy_products
    ss = ws.ss_products

    @inbounds for i in 1:n
        step_corr[i, tail_idx] = step[i]
        wy[i, tail_idx] = grad_diff[i]
    end
    ws.theta = rr / dr
    if ws.update_count > m
        for j in 1:col-1
            for i in 1:j
                ss[i, j] = ss[i+1, j+1]
            end
            for i in 1:col-j
                sy[j+i-1, j] = sy[j+i, j+1]
            end
        end
    end
    ring_ptr = ring_head
    for j in 1:col-1
        sy[col, j] = dot(view(step_corr, :, tail_idx), view(wy, :, ring_ptr))
        ss[j, col] = dot(view(step_corr, :, ring_ptr), view(step_corr, :, tail_idx))
        ring_ptr = _next_ring(ring_ptr, m)
    end
    ss[col, col] = dtd
    sy[col, col] = dr
    return nothing
end


"""
    _max_feasible_step(x, d, lb, ub)

Compute the maximum step length α such that `x + α*d` remains feasible
(within bounds `lb` and `ub`).

Nocedal & Wright (2006), Section 16.7.
"""
@inline function _max_feasible_step(x::Vector{Float64}, d::Vector{Float64},
    lb::Vector{Float64}, ub::Vector{Float64})
    alpha_max = Inf
    @inbounds for i in eachindex(x)
        di = d[i]
        if di < 0.0 && isfinite(lb[i])
            if x[i] <= lb[i]
                return 0.0
            end
            alpha_max = min(alpha_max, (lb[i] - x[i]) / di)
        elseif di > 0.0 && isfinite(ub[i])
            if x[i] >= ub[i]
                return 0.0
            end
            alpha_max = min(alpha_max, (ub[i] - x[i]) / di)
        end
    end
    return alpha_max
end


"""
    _strong_wolfe_line_search!(...)

Perform a line search satisfying the strong Wolfe conditions using
bracketing and zoom.

Nocedal & Wright (2006), Algorithm 3.5.
"""
function _strong_wolfe_line_search!(x::Vector{Float64}, fx::Float64, grad::Vector{Float64},
    d::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64},
    f::Function, g::Function;
    c1::Float64=1e-3, c2::Float64=0.9, iter::Int=1,
    xtol::Float64=0.1, boxed::Bool=false,
    max_evals::Int=40, max_step::Float64=Inf)

    n = length(x)
    phi0prime = dot(grad, d)
    if !(isfinite(phi0prime)) || phi0prime >= 0.0

        return false, 0.0, fx, grad, x, 0, 0
    end

    alpha_max = _max_feasible_step(x, d, lb, ub)
    if isfinite(max_step) && max_step < alpha_max
        alpha_max = max_step
    end
    if !(alpha_max > 0.0)
        return false, 0.0, fx, grad, x, 0, 0
    end

    Dnorm = sqrt(dot(d, d))
    alpha_init = (iter == 1 && !boxed) ? (1.0 / max(Dnorm, eps())) : 1.0
    alpha = isfinite(alpha_max) ? min(alpha_init, alpha_max) : alpha_init

    xtrial = similar(x)
    gtrial = similar(grad)

    fevals = 0
    gevals = 0

    f_prev = fx
    alpha_prev = 0.0
    f_best = fx
    alpha_best = 0.0
    g_best = copy(grad)

    eval_at = function (alphat)
        @inbounds @. xtrial = x + alphat * d
        ft = f(xtrial)
        fevals += 1
        gt = g(xtrial)
        gevals += 1
        return ft, gt
    end

    ft, gt = eval_at(alpha)
    phialpha = dot(gt, d)
    if ft <= fx + c1 * alpha * phi0prime
        f_best = ft
        alpha_best = alpha
        g_best .= gt
    end

    for k in 1:max_evals
        if (ft > fx + c1 * alpha * phi0prime) || (k > 1 && ft >= f_prev)

            ok, alphaz, fz, gz, ez_fe, ez_ge = _wolfe_zoom!(x, fx, grad, d, lb, ub, f, g,
                alpha_prev, alpha, f_prev, ft,
                phi0prime; c1=c1, c2=c2, xtol=xtol, max_evals=max_evals - (fevals + gevals))
            fevals += ez_fe
            gevals += ez_ge
            if ok
                @inbounds @. xtrial = x + alphaz * d
                return true, alphaz, fz, gz, xtrial, fevals, gevals
            else
                if alpha_best > 0.0
                    @inbounds @. xtrial = x + alpha_best * d
                    return true, alpha_best, f_best, g_best, xtrial, fevals, gevals
                end
                return false, 0.0, fx, grad, x, fevals, gevals
            end
        end
        if abs(phialpha) <= c2 * abs(phi0prime)
            @inbounds @. xtrial = x + alpha * d
            return true, alpha, ft, gt, xtrial, fevals, gevals
        end
        if phialpha >= 0.0

            ok, alphaz, fz, gz, ez_fe, ez_ge = _wolfe_zoom!(x, fx, grad, d, lb, ub, f, g,
                alpha, alpha_prev, ft, f_prev,
                phi0prime; c1=c1, c2=c2, xtol=xtol, max_evals=max_evals - (fevals + gevals))
            fevals += ez_fe
            gevals += ez_ge
            if ok
                @inbounds @. xtrial = x + alphaz * d
                return true, alphaz, fz, gz, xtrial, fevals, gevals
            else
                if alpha_best > 0.0
                    @inbounds @. xtrial = x + alpha_best * d
                    return true, alpha_best, f_best, g_best, xtrial, fevals, gevals
                end
                return false, 0.0, fx, grad, x, fevals, gevals
            end
        end

        alpha_prev, f_prev = alpha, ft
        alpha = min(2.0 * alpha, alpha_max)
        if alpha == alpha_prev
            if alpha_best > 0.0
                @inbounds @. xtrial = x + alpha_best * d
                return true, alpha_best, f_best, g_best, xtrial, fevals, gevals
            elseif ft < fx
                @inbounds @. xtrial = x + alpha * d
                return true, alpha, ft, gt, xtrial, fevals, gevals
            else
                return false, 0.0, fx, grad, x, fevals, gevals
            end
        end
        ft, gt = eval_at(alpha)
        phialpha = dot(gt, d)
        if ft <= fx + c1 * alpha * phi0prime
            f_best = ft
            alpha_best = alpha
            g_best .= gt
        end
    end

    if alpha_best > 0.0
        @inbounds @. xtrial = x + alpha_best * d
        return true, alpha_best, f_best, g_best, xtrial, fevals, gevals
    end
    return false, 0.0, fx, grad, x, fevals, gevals
end

"""
    _cubic_interpolation_min(a1, f1, dphi1, a2, f2, dphi2)

Find the minimizer of the cubic interpolant through two points with
function values and derivatives.

Nocedal & Wright (2006), Eq. (3.59).
"""
function _cubic_interpolation_min(a1, f1, dphi1, a2, f2, dphi2)
    d1 = dphi1 + dphi2 - 3.0 * (f2 - f1) / (a2 - a1)
    disc = d1 * d1 - dphi1 * dphi2
    disc < 0.0 && return NaN
    d2 = copysign(sqrt(disc), a2 - a1)
    return a2 - (a2 - a1) * (dphi2 + d2 - d1) / (dphi2 - dphi1 + 2.0 * d2)
end

"""
    _wolfe_zoom!(...)

Zoom phase of the strong Wolfe line search. Given a bracket [alpha_lo, alpha_hi]
known to contain an acceptable step length, narrow the bracket until the strong
Wolfe conditions are satisfied.

Nocedal & Wright (2006), Algorithm 3.6.
"""
function _wolfe_zoom!(x, fx, gx, d, lb, ub, f, g,
    alpha_lo, alpha_hi, f_lo, f_hi, phi0prime; c1=1e-3, c2=0.9, xtol=0.1, max_evals=30)
    fevals = 0
    gevals = 0
    xtrial = similar(x)
    gtrial = similar(gx)

    @inbounds @. xtrial = x + alpha_lo * d
    f_lo_eval = f_lo
    g_lo = g(xtrial)
    gevals += 1
    phi_lo = dot(g_lo, d)
    g_lo_saved = copy(g_lo)

    phi_hi = NaN

    eval_at = function (alphat)
        @inbounds @. xtrial = x + alphat * d
        ft = f(xtrial)
        fevals += 1
        gt = g(xtrial)
        gevals += 1
        return ft, gt
    end

    for _ in 1:max_evals
        bracket_max = max(alpha_lo, alpha_hi)
        if bracket_max > 0 && abs(alpha_hi - alpha_lo) <= xtol * bracket_max
            return true, alpha_lo, f_lo_eval, g_lo_saved, fevals, gevals
        end
        alpha_j = _cubic_interpolation_min(alpha_lo, f_lo_eval, phi_lo, alpha_hi, f_hi, phi_hi)
        a_min = min(alpha_lo, alpha_hi)
        a_max = max(alpha_lo, alpha_hi)
        if !isfinite(alpha_j) || alpha_j <= a_min || alpha_j >= a_max
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
        end

        fj, gj = eval_at(alpha_j)
        if (fj > fx + c1 * alpha_j * phi0prime) || (fj >= f_lo_eval)
            alpha_hi, f_hi = alpha_j, fj
            phi_hi = dot(gj, d)
        else
            phij = dot(gj, d)
            if abs(phij) <= c2 * abs(phi0prime)
                return true, alpha_j, fj, gj, fevals, gevals
            end
            if phij * (alpha_hi - alpha_lo) >= 0
                alpha_hi, f_hi = alpha_lo, f_lo_eval
                phi_hi = phi_lo
            end
            alpha_lo, f_lo_eval = alpha_j, fj
            phi_lo = phij
            g_lo_saved .= gj
        end
        if abs(alpha_hi - alpha_lo) <= 1e-16
            return true, alpha_lo, f_lo_eval, g_lo_saved, fevals, gevals
        end
    end
    return false, 0.0, fx, gx, fevals, gevals
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
    _, wk.has_bounds, wk.boxed = _initialize_active_set!(x, lb, ub, wk.bound_types, wk.var_status, n)

    fx = f(x)
    fn_evals = 1
    if !isfinite(fx)
        return (x_opt=x, f_opt=fx, n_iter=0,
            fail=52, fn_evals=fn_evals, gr_evals=0,
            message="Error: objective function returned non-finite value")
    end

    grad = copy(g(x))
    gr_evals = 1

    for i in eachindex(grad)
        if !isfinite(grad[i])
            return (x_opt=x, f_opt=fx, n_iter=0,
                fail=52, fn_evals=fn_evals, gr_evals=gr_evals,
                message="Error: gradient contains non-finite values")
        end
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

        wrk = false
        if !wk.has_bounds && wk.n_corrections > 0
            copyto!(cauchy_point, x)
            wrk = wk.did_update
        else
            try
                _generalized_cauchy_point!(wk, x, lb, ub, grad, proj_grad_norm)
            catch e
                e isa SingularFactorizationError || rethrow()
                if options.print_level > 0
                    println("Singular triangular system in cauchy; refreshing memory.")
                end
                _reset_memory!(wk)
                continue
            end
            wrk = _partition_free_active!(wk, iter - 1)
        end

        if wk.n_free != 0 && wk.n_corrections != 0
            try
                if wrk
                    _assemble_reduced_hessian!(wk)
                end
                @inbounds for i in 1:2*wk.n_corrections
                    wk.solve_input[i] = wk.c_work[i]
                end
                _reduced_gradient!(wk, x, grad)
                @inbounds for i in 1:wk.n_free
                    search_dir[i] = wk.reduced_grad[i]
                end
                _subspace_minimization!(wk, lb, ub)
            catch e
                e isa SingularFactorizationError || rethrow()
                if options.print_level > 0
                    println("Singular triangular system in subsm; refreshing memory.")
                end
                _reset_memory!(wk)
                continue
            end
        end

        @inbounds for i in 1:n
            search_dir[i] = cauchy_point[i] - x[i]
        end

        dnorm_sq = dot(search_dir, search_dir)
        if dnorm_sq < 1e-32
            fail = 0
            message = "Converged: search direction is effectively zero"
            break
        end

        x_prev = copy(x)
        grad_prev = copy(grad)
        fold = fx

        max_step = 1e10
        if wk.has_bounds
            nbd = wk.bound_types
            if iter == 1
                max_step = 1.0
            else
                @inbounds for i in 1:n
                    step_i = search_dir[i]
                    if _is_bounded(nbd[i])
                        if step_i < 0.0 && _has_lower(nbd[i])
                            bound_dist = lb[i] - x[i]
                            if bound_dist >= 0.0
                                max_step = 0.0
                            elseif step_i * max_step < bound_dist
                                max_step = bound_dist / step_i
                            end
                        elseif step_i > 0.0 && _has_upper(nbd[i])
                            bound_dist = ub[i] - x[i]
                            if bound_dist <= 0.0
                                max_step = 0.0
                            elseif step_i * max_step > bound_dist
                                max_step = bound_dist / step_i
                            end
                        end
                    end
                end
            end
        end

        ok, _, fnew, gnew, xnew, fe, ge = _strong_wolfe_line_search!(x, fx, grad, search_dir, lb, ub,
            f, g; iter=iter, boxed=wk.boxed, max_step=max_step)
        fn_evals += fe
        gr_evals += ge

        if !ok
            x .= x_prev
            grad .= grad_prev
            fx = fold
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
            fx = fold
            fail = 52
            message = "Error: objective function returned non-finite value"
            break
        end

        x .= xnew
        fx = fnew
        grad .= copy(gnew)

        has_nonfinite_grad = false
        for i in eachindex(grad)
            if !isfinite(grad[i])
                has_nonfinite_grad = true
                break
            end
        end
        if has_nonfinite_grad
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
        max_f = max(abs(fold), abs(fx), 1.0)
        if fold - fx <= f_tol * max_f
            fail = 0
            message = "Converged: relative function reduction below tolerance"
            break
        end

        @inbounds for i in 1:n
            search_dir[i] = x[i] - x_prev[i]
            wk.reduced_grad[i] = grad[i] - grad_prev[i]
        end
        rr = dot(wk.reduced_grad, wk.reduced_grad)
        dr = dot(wk.reduced_grad, search_dir)
        dtd = dot(search_dir, search_dir)
        descent_magnitude = -dot(grad_prev, search_dir)

        if dr <= eps(Float64) * descent_magnitude
            wk.did_update = false
        else
            wk.did_update = true
            wk.update_count += 1
            _update_lbfgs_matrices!(wk, search_dir, wk.reduced_grad, rr, dr, dtd)
            try
                _factorize_wt_matrix!(wk)
            catch e
                e isa SingularFactorizationError || rethrow()
                if options.print_level > 0
                    println("Nonpositive definiteness in formt; refreshing memory.")
                end
                _reset_memory!(wk)
            end
        end
    end

    return (x_opt=x, f_opt=fx, n_iter=iter,
        fail=fail, fn_evals=fn_evals, gr_evals=gr_evals,
        message=message)
end
