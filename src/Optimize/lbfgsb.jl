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
            return j
        end
        A[j,j] = sqrt(s)
    end
    return 0
end

function _triangular_solve!(A::AbstractMatrix{Float64}, r::Int, n::Int, b::AbstractVector{Float64}, boff::Int)
    @inbounds for j in n:-1:1
        if A[r+j-1, r+j-1] == 0.0
            return j
        end
        b[boff+j] /= A[r+j-1, r+j-1]
        for i in 1:j-1
            b[boff+i] -= A[r+i-1, r+j-1] * b[boff+j]
        end
    end
    return 0
end

function _triangular_solve_t!(A::AbstractMatrix{Float64}, r::Int, n::Int, b::AbstractVector{Float64}, boff::Int)
    @inbounds for j in 1:n
        s = b[boff+j]
        for i in 1:j-1
            s -= A[r+i-1, r+j-1] * b[boff+i]
        end
        if A[r+j-1, r+j-1] == 0.0
            return j
        end
        b[boff+j] = s / A[r+j-1, r+j-1]
    end
    return 0
end

"""
    _compact_representation_solve!(p, sy, wt, col, m, v)

Solve the compact L-BFGS representation system using the triangular matrices
`sy` (correction pairs) and `wt` (compact form).

Byrd et al. (1995), Theorem 2.3.
"""
function _compact_representation_solve!(p::AbstractVector{Float64}, sy::Matrix{Float64}, wt::Matrix{Float64},
    col::Int, m::Int, v::AbstractVector{Float64})
    if col == 0
        return 0
    end
    p[col+1] = v[col+1]
    @inbounds for i in 2:col
        s = 0.0
        for k in 1:i-1
            s += sy[i, k] * v[k] / sy[k, k]
        end
        p[col+i] = v[col+i] + s
    end
    info = _triangular_solve_t!(wt, 1, col, p, col)
    if info != 0
        return info
    end
    @inbounds for i in 1:col
        p[i] = v[i] / sqrt(sy[i, i])
    end

    info = _triangular_solve!(wt, 1, col, p, col)
    if info != 0
        return info
    end
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
    return 0
end

function _heap_sort!(t::Vector{Float64}, iorder::Vector{Int}, n::Int, iheap::Int)
    if iheap == 0
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
    _generalized_cauchy_point!(...)

Compute the Generalized Cauchy Point (GCP) by piecewise-linear search along the
projected steepest descent path.

Byrd et al. (1995), Algorithm CP (Section 4).
"""
function _generalized_cauchy_point!(n::Int, x::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64},
    nbd::Vector{BoundType}, grad::Vector{Float64}, iorder::Vector{Int}, var_status::Vector{VarStatus},
    t::Vector{Float64}, d::Vector{Float64}, xcp::Vector{Float64}, m::Int,
    wy::Matrix{Float64}, ws::Matrix{Float64}, sy::Matrix{Float64}, wt::Matrix{Float64},
    theta::Float64, col::Int, head::Int,
    p::Vector{Float64}, c::Vector{Float64}, breakpoint_work::Vector{Float64}, v::Vector{Float64},
    proj_grad_norm::Float64)

    if proj_grad_norm <= 0.0
        copyto!(xcp, x)
        return 0, 0
    end
    all_bounded_active = true
    nfree = n + 1
    nbreak = 0
    ibkmin = 0
    bkmin = 0.0
    col2 = 2 * col
    f1 = 0.0
    n_intervals = 1

    @inbounds for i in 1:col2
        p[i] = 0.0
    end

    tl = 0.0
    tu = 0.0
    @inbounds for i in 1:n
        neggi = -grad[i]
        if var_status[i] != VAR_FIXED && var_status[i] != VAR_UNBOUNDED
            if _has_lower(nbd[i])
                tl = x[i] - lb[i]
            end
            if _has_upper(nbd[i])
                tu = ub[i] - x[i]
            end
            xlower = _has_lower(nbd[i]) && tl <= 0.0
            xupper = _has_upper(nbd[i]) && tu <= 0.0
            var_status[i] = VAR_FREE
            if xlower
                if neggi <= 0.0
                    var_status[i] = VAR_LOWER_ACTIVE
                end
            elseif xupper
                if neggi >= 0.0
                    var_status[i] = VAR_UPPER_ACTIVE
                end
            else
                if abs(neggi) <= 0.0
                    var_status[i] = VAR_ZERO_GRADIENT
                end
            end
        end
        buf_ptr = head
        if var_status[i] != VAR_FREE && var_status[i] != VAR_UNBOUNDED
            d[i] = 0.0
        else
            d[i] = neggi
            f1 -= neggi * neggi
            for j in 1:col
                p[j] += wy[i, buf_ptr] * neggi
                p[col+j] += ws[i, buf_ptr] * neggi
                buf_ptr = buf_ptr % m + 1
            end
            if _has_lower(nbd[i]) && nbd[i] != UNBOUNDED && neggi < 0.0
                nbreak += 1
                iorder[nbreak] = i
                t[nbreak] = tl / (-neggi)
                if nbreak == 1 || t[nbreak] < bkmin
                    bkmin = t[nbreak]
                    ibkmin = nbreak
                end
            elseif _has_upper(nbd[i]) && neggi > 0.0
                nbreak += 1
                iorder[nbreak] = i
                t[nbreak] = tu / neggi
                if nbreak == 1 || t[nbreak] < bkmin
                    bkmin = t[nbreak]
                    ibkmin = nbreak
                end
            else
                nfree -= 1
                iorder[nfree] = i
                if abs(neggi) > 0.0
                    all_bounded_active = false
                end
            end
        end
    end

    if theta != 1.0
        @inbounds for j in col+1:col2
            p[j] *= theta
        end
    end
    copyto!(xcp, x)
    if nbreak == 0 && nfree == n + 1
        return 0, 0
    end
    @inbounds for j in 1:col2
        c[j] = 0.0
    end

    f2 = -theta * f1
    f2_org = f2
    if col > 0
        info = _compact_representation_solve!(v, sy, wt, col, m, p)
        if info != 0
            return n_intervals, info
        end
        s = 0.0
        @inbounds for i in 1:col2
            s += v[i] * p[i]
        end
        f2 -= s
    end
    dtm = -f1 / f2
    tsum = 0.0

    skip_xcp_update = false

    if nbreak > 0
        nleft = nbreak
        cauchyiter = 1
        tj = 0.0

        while true
            tj0 = tj
            if cauchyiter == 1
                tj = bkmin
                breakpoint_var = iorder[ibkmin]
            else
                if cauchyiter == 2
                    if ibkmin != nbreak
                        t[ibkmin] = t[nbreak]
                        iorder[ibkmin] = iorder[nbreak]
                    end
                end
                _heap_sort!(t, iorder, nleft, cauchyiter - 2)
                tj = t[nleft]
                breakpoint_var = iorder[nleft]
            end
            dt = tj - tj0
            if dtm < dt
                break
            end
            tsum += dt
            nleft -= 1
            cauchyiter += 1
            d_at_break = d[breakpoint_var]
            d[breakpoint_var] = 0.0
            if d_at_break > 0.0
                z_at_break = ub[breakpoint_var] - x[breakpoint_var]
                xcp[breakpoint_var] = ub[breakpoint_var]
                var_status[breakpoint_var] = VAR_UPPER_ACTIVE
            else
                z_at_break = lb[breakpoint_var] - x[breakpoint_var]
                xcp[breakpoint_var] = lb[breakpoint_var]
                var_status[breakpoint_var] = VAR_LOWER_ACTIVE
            end
            if nleft == 0 && nbreak == n
                dtm = dt
                skip_xcp_update = true
                break
            end
            n_intervals += 1
            d_at_break_sq = d_at_break * d_at_break
            f1 += dt * f2 + d_at_break_sq - theta * d_at_break * z_at_break
            f2 -= theta * d_at_break_sq
            if col > 0
                @inbounds for j in 1:col2
                    c[j] += dt * p[j]
                end
                buf_ptr = head
                @inbounds for j in 1:col
                    breakpoint_work[j] = wy[breakpoint_var, buf_ptr]
                    breakpoint_work[col+j] = theta * ws[breakpoint_var, buf_ptr]
                    buf_ptr = buf_ptr % m + 1
                end
                info = _compact_representation_solve!(v, sy, wt, col, m, breakpoint_work)
                if info != 0
                    return n_intervals, info
                end
                wmc = 0.0; wmp = 0.0; wmw = 0.0
                @inbounds for j in 1:col2
                    wmc += c[j] * v[j]
                    wmp += p[j] * v[j]
                    wmw += breakpoint_work[j] * v[j]
                end
                @inbounds for j in 1:col2
                    p[j] -= d_at_break * breakpoint_work[j]
                end
                f1 += d_at_break * wmc
                f2 += 2.0 * d_at_break * wmp - d_at_break_sq * wmw
            end
            f2 = max(f2, eps(Float64) * f2_org)
            if nleft > 0
                dtm = -f1 / f2
                continue
            elseif all_bounded_active
                f1 = 0.0; f2 = 0.0; dtm = 0.0
            else
                dtm = -f1 / f2
            end
            break
        end
    end

    if !skip_xcp_update
        dtm = max(dtm, 0.0)
        tsum += dtm
        @inbounds for i in 1:n
            xcp[i] += tsum * d[i]
        end
    end

    if col > 0
        @inbounds for j in 1:col2
            c[j] += dtm * p[j]
        end
    end
    return n_intervals, 0
end

"""
    _partition_free_active!(n, nfree_prev, indx, var_status, indx2, has_bounds, did_update, iter)

Partition variables into free and active sets based on the current `var_status` classification.
Track which variables entered or left the free set since last iteration.

Byrd et al. (1995), Section 5.
"""
function _partition_free_active!(n::Int, nfree_prev::Int, indx::Vector{Int}, var_status::Vector{VarStatus},
    indx2::Vector{Int}, has_bounds::Bool, did_update::Bool, iter::Int)
    nenter = 0
    ileave = n + 1
    if iter > 0 && has_bounds
        for i in 1:nfree_prev
            k = indx[i]
            if _is_active(var_status[k])
                ileave -= 1
                indx2[ileave] = k
            end
        end
        for i in nfree_prev+1:n
            k = indx[i]
            if _is_free(var_status[k])
                nenter += 1
                indx2[nenter] = k
            end
        end
    end
    wrk = (ileave < n + 1) || (nenter > 0) || did_update
    nfree = 0
    iact = n + 1
    @inbounds for i in 1:n
        if _is_free(var_status[i])
            nfree += 1
            indx[nfree] = i
        else
            iact -= 1
            indx[iact] = i
        end
    end
    return nfree, nenter, ileave, wrk
end


"""
    _assemble_reduced_hessian!(...)

Assemble the reduced Hessian WN for the free variables from the L-BFGS
correction pairs stored in `ws`, `wy`, `sy`, and `ss`.

Byrd et al. (1995), Section 5, Equations (5.4)–(5.11).
"""
function _assemble_reduced_hessian!(n::Int, nsub::Int, indx::Vector{Int}, nenter::Int, ileave::Int,
    indx2::Vector{Int}, update_count::Int, did_update::Bool,
    wn::Matrix{Float64}, hessian_parts::Matrix{Float64}, m::Int,
    ws::Matrix{Float64}, wy::Matrix{Float64}, sy::Matrix{Float64},
    theta::Float64, col::Int, head::Int)

    if did_update
        if update_count > m
            for jy in 1:m-1
                js = m + jy
                for i in 1:m-jy
                    hessian_parts[jy+i-1, jy] = hessian_parts[jy+i, jy+1]
                    hessian_parts[js+i-1, js] = hessian_parts[js+i, js+1]
                end
                for i in 1:m-1
                    hessian_parts[m+i, jy] = hessian_parts[m+i+1, jy+1]
                end
            end
        end
        iy = col
        is_ = m + col
        ipntr = (head + col - 2) % m + 1
        jpntr = head
        for jy in 1:col
            js = m + jy
            temp1 = 0.0; temp2 = 0.0; temp3 = 0.0
            for k in 1:nsub
                k1 = indx[k]
                temp1 += wy[k1, ipntr] * wy[k1, jpntr]
            end
            for k in nsub+1:n
                k1 = indx[k]
                temp2 += ws[k1, ipntr] * ws[k1, jpntr]
                temp3 += ws[k1, ipntr] * wy[k1, jpntr]
            end
            hessian_parts[iy, jy] = temp1
            hessian_parts[is_, js] = temp2
            hessian_parts[is_, jy] = temp3
            jpntr = jpntr % m + 1
        end
        jpntr = (head + col - 2) % m + 1
        ipntr = head
        for i in 1:col
            is_ = m + i
            temp3 = 0.0
            for k in 1:nsub
                k1 = indx[k]
                temp3 += ws[k1, ipntr] * wy[k1, jpntr]
            end
            ipntr = ipntr % m + 1
            hessian_parts[is_, col] = temp3
        end
        upcl = col - 1
    else
        upcl = col
    end

    ipntr = head
    for iy in 1:upcl
        is_ = m + iy
        jpntr = head
        for jy in 1:iy
            js = m + jy
            temp1 = 0.0; temp2 = 0.0; temp3 = 0.0; temp4 = 0.0
            for k in 1:nenter
                k1 = indx2[k]
                temp1 += wy[k1, ipntr] * wy[k1, jpntr]
                temp2 += ws[k1, ipntr] * ws[k1, jpntr]
            end
            for k in ileave:n
                k1 = indx2[k]
                temp3 += wy[k1, ipntr] * wy[k1, jpntr]
                temp4 += ws[k1, ipntr] * ws[k1, jpntr]
            end
            hessian_parts[iy, jy] += temp1 - temp3
            hessian_parts[is_, js] += -temp2 + temp4
            jpntr = jpntr % m + 1
        end
        ipntr = ipntr % m + 1
    end
    ipntr = head
    for is_ in m+1:m+upcl
        jpntr = head
        for jy in 1:upcl
            temp1 = 0.0; temp3 = 0.0
            for k in 1:nenter
                k1 = indx2[k]
                temp1 += ws[k1, ipntr] * wy[k1, jpntr]
            end
            for k in ileave:n
                k1 = indx2[k]
                temp3 += ws[k1, ipntr] * wy[k1, jpntr]
            end
            if is_ <= jy + m
                hessian_parts[is_, jy] += temp1 - temp3
            else
                hessian_parts[is_, jy] += -temp1 + temp3
            end
            jpntr = jpntr % m + 1
        end
        ipntr = ipntr % m + 1
    end

    m2 = 2 * m
    for iy in 1:col
        is_ = col + iy
        is1 = m + iy
        for jy in 1:iy
            js = col + jy
            js1 = m + jy
            wn[jy, iy] = hessian_parts[iy, jy] / theta
            wn[js, is_] = hessian_parts[is1, js1] * theta
        end
        for jy in 1:iy-1
            wn[jy, is_] = -hessian_parts[is1, jy]
        end
        for jy in iy:col
            wn[jy, is_] = hessian_parts[is1, jy]
        end
        wn[iy, iy] += sy[iy, iy]
    end

    info = _cholesky!(wn, col)
    if info != 0
        return -1
    end
    col2 = 2 * col
    for js in col+1:col2
        info = _triangular_solve_t!(wn, 1, col, view(wn, :, js), 0)
        if info != 0
            return -1
        end
    end
    for is_ in col+1:col2
        for js in is_:col2
            s = 0.0
            @inbounds for k in 1:col
                s += wn[k, is_] * wn[k, js]
            end
            wn[is_, js] += s
        end
    end
    info = _cholesky!(view(wn, col+1:col2, col+1:col2), col)
    if info != 0
        return -2
    end
    return 0
end

"""
    _factorize_wt_matrix!(m, wt, sy, ss, col, theta)

Form and factorize the upper triangular matrix Wθ used in the compact
L-BFGS representation.

Byrd et al. (1995), Eq. (3.2).
"""
function _factorize_wt_matrix!(m::Int, wt::Matrix{Float64}, sy::Matrix{Float64},
    ss::Matrix{Float64}, col::Int, theta::Float64)
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
    info = _cholesky!(wt, col)
    return info != 0 ? -3 : 0
end

"""
    _reduced_gradient!(...)

Compute the reduced gradient for the free variables, using the L-BFGS compact
representation to account for curvature information.

Byrd et al. (1995), Section 5.
"""
function _reduced_gradient!(n::Int, m::Int, x::Vector{Float64}, grad::Vector{Float64},
    ws::Matrix{Float64}, wy::Matrix{Float64}, sy::Matrix{Float64}, wt::Matrix{Float64},
    z::Vector{Float64}, r::Vector{Float64}, wa::Vector{Float64},
    indx::Vector{Int}, theta::Float64, col::Int, head::Int, nfree::Int, has_bounds::Bool)

    if !has_bounds && col > 0
        for i in 1:n
            r[i] = -grad[i]
        end
    else
        for i in 1:nfree
            k = indx[i]
            r[i] = -theta * (z[k] - x[k]) - grad[k]
        end
        info = _compact_representation_solve!(wa, sy, wt, col, m, view(wa, 2m+1:4m))
        if info != 0
            return -8
        end
        buf_ptr = head
        for j in 1:col
            a1 = wa[j]
            a2 = theta * wa[col+j]
            for i in 1:nfree
                k = indx[i]
                r[i] += wy[k, buf_ptr] * a1 + ws[k, buf_ptr] * a2
            end
            buf_ptr = buf_ptr % m + 1
        end
    end
    return 0
end

"""
    _subspace_minimization!(...)

Perform subspace minimization within the free variable space, subject to
bound constraints. This finds the minimizer of the quadratic model restricted
to the subspace of free variables identified at the Generalized Cauchy Point.

Byrd et al. (1995), Section 5.
"""
function _subspace_minimization!(n::Int, m::Int, nsub::Int, indx::Vector{Int},
    lb::Vector{Float64}, ub::Vector{Float64}, nbd::Vector{BoundType},
    z::Vector{Float64}, d::Vector{Float64},
    ws::Matrix{Float64}, wy::Matrix{Float64}, theta::Float64,
    col::Int, head::Int, wv::Vector{Float64}, wn::Matrix{Float64})

    if nsub <= 0
        return 0, 0
    end
    buf_ptr = head
    @inbounds for i in 1:col
        temp1 = 0.0; temp2 = 0.0
        for j in 1:nsub
            k = indx[j]
            temp1 += wy[k, buf_ptr] * d[j]
            temp2 += ws[k, buf_ptr] * d[j]
        end
        wv[i] = temp1
        wv[col+i] = theta * temp2
        buf_ptr = buf_ptr % m + 1
    end
    m2 = 2 * m
    col2 = 2 * col
    info = _triangular_solve_t!(wn, 1, col2, wv, 0)
    if info != 0
        return 0, info
    end
    @inbounds for i in 1:col
        wv[i] = -wv[i]
    end
    info = _triangular_solve!(wn, 1, col2, wv, 0)
    if info != 0
        return 0, info
    end
    buf_ptr = head
    @inbounds for jy in 1:col
        js = col + jy
        for i in 1:nsub
            k = indx[i]
            d[i] += (wy[k, buf_ptr] * wv[jy] / theta + ws[k, buf_ptr] * wv[js])
        end
        buf_ptr = buf_ptr % m + 1
    end
    @inbounds for i in 1:nsub
        d[i] /= theta
    end

    alpha = 1.0
    temp1 = alpha
    ibd = 0
    @inbounds for i in 1:nsub
        k = indx[i]
        dk = d[i]
        if _is_bounded(nbd[k])
            if dk < 0.0 && _has_lower(nbd[k])
                temp2 = lb[k] - z[k]
                if temp2 >= 0.0
                    temp1 = 0.0
                elseif dk * alpha < temp2
                    temp1 = temp2 / dk
                end
            elseif dk > 0.0 && _has_upper(nbd[k])
                temp2 = ub[k] - z[k]
                if temp2 <= 0.0
                    temp1 = 0.0
                elseif dk * alpha > temp2
                    temp1 = temp2 / dk
                end
            end
            if temp1 < alpha
                alpha = temp1
                ibd = i
            end
        end
    end
    if alpha < 1.0
        dk = d[ibd]
        k = indx[ibd]
        if dk > 0.0
            z[k] = ub[k]
            d[ibd] = 0.0
        elseif dk < 0.0
            z[k] = lb[k]
            d[ibd] = 0.0
        end
    end
    @inbounds for i in 1:nsub
        z[indx[i]] += alpha * d[i]
    end
    iword = alpha < 1.0 ? 1 : 0
    return iword, 0
end

"""
    _update_lbfgs_matrices!(...)

Update the L-BFGS correction matrices (ws, wy, sy, ss) with a new step/gradient
difference pair.

Byrd et al. (1995), Section 3.
"""
function _update_lbfgs_matrices!(n::Int, m::Int, ws::Matrix{Float64}, wy::Matrix{Float64},
    sy::Matrix{Float64}, ss::Matrix{Float64},
    d::Vector{Float64}, r::Vector{Float64},
    tail_index::Int, update_count::Int, col::Int, head::Int,
    rr::Float64, dr::Float64, stp::Float64, dtd::Float64)

    if update_count <= m
        col = update_count
        tail_index = (head + update_count - 2) % m + 1
    else
        tail_index = tail_index % m + 1
        head = head % m + 1
    end
    @inbounds for i in 1:n
        ws[i, tail_index] = d[i]
        wy[i, tail_index] = r[i]
    end
    theta = rr / dr
    if update_count > m
        for j in 1:col-1
            for i in 1:j
                ss[i, j] = ss[i+1, j+1]
            end
            for i in 1:col-j
                sy[j+i-1, j] = sy[j+i, j+1]
            end
        end
    end
    buf_ptr = head
    for j in 1:col-1
        sy[col, j] = dot(view(ws, :, tail_index), view(wy, :, buf_ptr))
        ss[j, col] = dot(view(ws, :, buf_ptr), view(ws, :, tail_index))
        buf_ptr = buf_ptr % m + 1
    end
    if stp == 1.0
        ss[col, col] = dtd
    else
        ss[col, col] = stp * stp * dtd
    end
    sy[col, col] = dr
    return tail_index, col, head, theta
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
    _reset_lbfgs_memory!()

Reset L-BFGS memory after encountering singularity or non-positive definiteness.
Returns the reset values as a tuple.
"""
@inline function _reset_lbfgs_memory()
    return 0, 1, 1.0, 0, false  # col, head, theta, update_count, did_update
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

    nbd = _classify_bound_types(lb, ub)

    ws = zeros(Float64, n, m)
    wy = zeros(Float64, n, m)
    sy = zeros(Float64, m, m)
    ss = zeros(Float64, m, m)
    wt = zeros(Float64, m, m)
    wn = zeros(Float64, 2m, 2m)
    hessian_parts = zeros(Float64, 2m, 2m)

    theta = 1.0
    col = 0
    head = 1
    tail_index = 0
    update_count = 0
    did_update = false

    z = Vector{Float64}(undef, n)
    r = Vector{Float64}(undef, n)
    d = Vector{Float64}(undef, n)
    t_bp = Vector{Float64}(undef, n)
    wa = Vector{Float64}(undef, 8m)
    p_work = Vector{Float64}(undef, 2m)
    c_work = Vector{Float64}(undef, 2m)
    breakpoint_work = Vector{Float64}(undef, 2m)
    v_work = Vector{Float64}(undef, 2m)
    wv = Vector{Float64}(undef, 2m)
    indx = Vector{Int}(undef, n)
    var_status = Vector{VarStatus}(undef, n)
    indx2 = Vector{Int}(undef, n)

    x = copy(x0)
    was_projected, has_bounds, boxed = _initialize_active_set!(x, lb, ub, nbd, var_status, n)

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

    proj_grad = similar(grad)
    proj_grad_norm = _proj_grad!(proj_grad, x, grad, lb, ub, nbd)

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
    iter = 0
    nfree = n
    nact = 0
    ileave = n + 1
    nenter = 0

    while true
        iter += 1

        wrk = false
        if !has_bounds && col > 0
            copyto!(z, x)
            wrk = did_update
        else
            n_intervals, info = _generalized_cauchy_point!(n, x, lb, ub, nbd, grad, indx2, var_status,
                t_bp, d, z, m, wy, ws, sy, wt, theta, col, head,
                p_work, c_work, breakpoint_work, v_work, proj_grad_norm)
            if info != 0
                if options.print_level > 0
                    println("Singular triangular system in cauchy; refreshing memory.")
                end
                col, head, theta, update_count, did_update = _reset_lbfgs_memory()
                continue
            end
            nfree, nenter, ileave, wrk = _partition_free_active!(n, nfree, indx, var_status, indx2,
                has_bounds, did_update, iter - 1)
            nact = n - nfree
        end

        if nfree != 0 && col != 0
            if wrk
                info = _assemble_reduced_hessian!(n, nfree, indx, nenter, ileave, indx2, update_count, did_update,
                    wn, hessian_parts, m, ws, wy, sy, theta, col, head)
                if info != 0
                    if options.print_level > 0
                        println("Nonpositive definiteness in formk; refreshing memory.")
                    end
                    col, head, theta, update_count, did_update = _reset_lbfgs_memory()
                    continue
                end
            end
            @inbounds for i in 1:2*col
                wa[2m+i] = c_work[i]
            end
            info = _reduced_gradient!(n, m, x, grad, ws, wy, sy, wt, z, r, wa,
                indx, theta, col, head, nfree, has_bounds)
            if info == 0
                @inbounds for i in 1:nfree
                    d[i] = r[i]
                end
                iword, info = _subspace_minimization!(n, m, nfree, indx, lb, ub, nbd, z, d,
                    ws, wy, theta, col, head, wv, wn)
            end
            if info != 0
                if options.print_level > 0
                    println("Singular triangular system in subsm; refreshing memory.")
                end
                col, head, theta, update_count, did_update = _reset_lbfgs_memory()
                continue
            end
        end

        @inbounds for i in 1:n
            d[i] = z[i] - x[i]
        end

        dnorm_sq = dot(d, d)
        if dnorm_sq < 1e-32
            fail = 0
            message = "Converged: search direction is effectively zero"
            break
        end
        dnorm = sqrt(dnorm_sq)

        x_prev = copy(x)
        grad_prev = copy(grad)
        fold = fx

        max_step = 1e10
        if has_bounds
            if iter == 1
                max_step = 1.0
            else
                @inbounds for i in 1:n
                    a1 = d[i]
                    if _is_bounded(nbd[i])
                        if a1 < 0.0 && _has_lower(nbd[i])
                            a2 = lb[i] - x[i]
                            if a2 >= 0.0
                                max_step = 0.0
                            elseif a1 * max_step < a2
                                max_step = a2 / a1
                            end
                        elseif a1 > 0.0 && _has_upper(nbd[i])
                            a2 = ub[i] - x[i]
                            if a2 <= 0.0
                                max_step = 0.0
                            elseif a1 * max_step > a2
                                max_step = a2 / a1
                            end
                        end
                    end
                end
            end
        end

        ok, stp, fnew, gnew, xnew, fe, ge = _strong_wolfe_line_search!(x, fx, grad, d, lb, ub,
            f, g; iter=iter, boxed=boxed, max_step=max_step)
        fn_evals += fe
        gr_evals += ge

        if !ok
            x .= x_prev
            grad .= grad_prev
            fx = fold
            if col == 0
                fail = 52
                message = "Error: line search failed to find acceptable step"
                break
            else
                if options.print_level > 0
                    println("Bad direction in line search; refreshing memory.")
                end
                col, head, theta, update_count, did_update = _reset_lbfgs_memory()
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

        proj_grad_norm = _proj_grad!(proj_grad, x, grad, lb, ub, nbd)

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
            d[i] = x[i] - x_prev[i]
            r[i] = grad[i] - grad_prev[i]
        end
        rr = dot(r, r)
        dr = dot(r, d)
        dtd = dot(d, d)
        descent_magnitude = -dot(grad_prev, d)

        if dr <= eps(Float64) * descent_magnitude
            did_update = false
        else
            did_update = true
            update_count += 1
            tail_index, col, head, theta = _update_lbfgs_matrices!(n, m, ws, wy, sy, ss, d, r,
                tail_index, update_count, col, head, rr, dr, 1.0, dtd)
            info = _factorize_wt_matrix!(m, wt, sy, ss, col, theta)
            if info != 0
                if options.print_level > 0
                    println("Nonpositive definiteness in formt; refreshing memory.")
                end
                col, head, theta, update_count, did_update = _reset_lbfgs_memory()
            end
        end
    end

    return (x_opt=x, f_opt=fx, n_iter=iter,
        fail=fail, fn_evals=fn_evals, gr_evals=gr_evals,
        message=message)
end
