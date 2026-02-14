import LinearAlgebra
using LinearAlgebra: dot, norm

struct LBFGSBOptions
    m::Int
    factr::Float64
    pgtol::Float64
    maxit::Int
    iprint::Int
end

LBFGSBOptions(; m::Int=10, factr::Float64=1e7, pgtol::Float64=1e-5, maxit::Int=1000, iprint::Int=0) =
    LBFGSBOptions(m, factr, pgtol, maxit, iprint)


function _nbd_from_bounds(l::Vector{Float64}, u::Vector{Float64})
    n = length(l)
    nbd = Vector{Int32}(undef, n)
    @inbounds for i in 1:n
        if isfinite(l[i]) && isfinite(u[i])
            nbd[i] = 2
        elseif isfinite(l[i])
            nbd[i] = 1
        elseif isfinite(u[i])
            nbd[i] = 3
        else
            nbd[i] = 0
        end
    end
    return nbd
end

@inline function _project!(y::AbstractVector{Float64}, l::Vector{Float64}, u::Vector{Float64})
    @inbounds for i in eachindex(y)
        yi = y[i]
        if yi < l[i]
            y[i] = l[i]
        elseif yi > u[i]
            y[i] = u[i]
        end
    end
    return y
end


function _proj_grad!(pg::Vector{Float64}, x::Vector{Float64}, g::Vector{Float64},
    l::Vector{Float64}, u::Vector{Float64}, nbd::Vector{Int32})
    @inbounds for i in eachindex(x)
        gi = g[i]
        bi = nbd[i]
        if bi != 0
            if gi < 0.0
                if bi >= 2
                    gi = max(x[i] - u[i], gi)
                end
            else
                if bi <= 2
                    gi = min(x[i] - l[i], gi)
                end
            end
        end
        pg[i] = gi
    end

    m = 0.0
    @inbounds for i in eachindex(pg)
        v = abs(pg[i])
        if isnan(v)
            return NaN
        end
        if v > m
            m = v
        end
    end
    return m
end


# ── Compact L-BFGS helper functions (R's lbfgsb.c) ──────────────────────

# Cholesky factorization of upper triangle: A = R'R, R stored in upper triangle.
# Only the upper triangle A[i,j] for i<=j is accessed/modified.
# Returns info=0 on success, info=j>0 if not positive definite at column j.
function _dpofa!(A::AbstractMatrix{Float64}, n::Int)
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

# Solve T*x = b where T is n×n upper triangular (stored in rows/cols r:r+n-1 of A).
# b is overwritten with the solution. Returns info=0 on success.
function _dtrsl_upper!(A::AbstractMatrix{Float64}, r::Int, n::Int, b::AbstractVector{Float64}, boff::Int)
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

# Solve T'*x = b where T is n×n upper triangular (stored in rows/cols r:r+n-1 of A).
# T' is lower triangular. b is overwritten with the solution.
function _dtrsl_upper_t!(A::AbstractMatrix{Float64}, r::Int, n::Int, b::AbstractVector{Float64}, boff::Int)
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

# R's bmv: compute p = M^{-1} v where M is the middle matrix of the compact L-BFGS.
# sy is m×m (S'Y), wt is m×m (Cholesky factor J' in upper triangle).
# v and p are length 2*col vectors.
function _bmv!(p::AbstractVector{Float64}, sy::Matrix{Float64}, wt::Matrix{Float64},
    col::Int, m::Int, v::AbstractVector{Float64})
    if col == 0
        return 0
    end
    # PART I: solve [ D^{1/2}      0 ] [p1] = [v1]
    #              [-L*D^{-1/2}   J ] [p2]   [v2]
    # First solve Jp2 = v2 + L*D^{-1}*v1
    p[col+1] = v[col+1]
    @inbounds for i in 2:col
        s = 0.0
        for k in 1:i-1
            s += sy[i, k] * v[k] / sy[k, k]
        end
        p[col+i] = v[col+i] + s
    end
    # Solve J*p2 = rhs (J' is upper triangular in wt, so J is lower → solve J'*x = rhs first? No.)
    # R calls dtrsl(&wt, &m, col, &p[col+1], &c__11, info) → job=11 → solve T'*x=b, T upper
    info = _dtrsl_upper_t!(wt, 1, col, p, col)
    if info != 0
        return info
    end
    # Solve D^{1/2} p1 = v1
    @inbounds for i in 1:col
        p[i] = v[i] / sqrt(sy[i, i])
    end

    # PART II: solve [-D^{1/2}    D^{-1/2}*L'] [p1] = [p1]
    #               [ 0          J'           ] [p2]   [p2]
    # Solve J'*p2 = p2
    info = _dtrsl_upper!(wt, 1, col, p, col)
    if info != 0
        return info
    end
    # Compute p1 = -D^{-1/2}*p1 + D^{-1}*L'*p2
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

# R's hpsolb: heap sort for breakpoints.
# Extracts the minimum element of t[1:n], puts it in t[n], and maintains a min-heap in t[1:n-1].
function _hpsolb!(t::Vector{Float64}, iorder::Vector{Int}, n::Int, iheap::Int)
    if iheap == 0
        # Build a min-heap from t[1:n]
        @inbounds for k in 2:n
            ddum = t[k]
            indxin = iorder[k]
            i = k
            while i > 1
                j = div(i, 2)
                if ddum < t[j]
                    t[i] = t[j]
                    iorder[i] = iorder[j]
                    i = j
                else
                    break
                end
            end
            t[i] = ddum
            iorder[i] = indxin
        end
    end
    # Extract minimum: move t[1] to t[n], restore heap in t[1:n-1]
    if n > 1
        out = t[1]
        indxou = iorder[1]
        ddum = t[n]
        indxin = iorder[n]
        i = 1
        @inbounds while true
            j = 2 * i
            if j > n - 1
                break
            end
            if t[j+1] < t[j]
                j += 1
            end
            if t[j] < ddum
                t[i] = t[j]
                iorder[i] = iorder[j]
                i = j
            else
                break
            end
        end
        t[i] = ddum
        iorder[i] = indxin
        t[n] = out
        iorder[n] = indxou
    end
end


# ── GCP and active set functions ─────────────────────────────────────────

# R's active: initialize iwhere and project x to feasible set.
# Returns (prjctd, cnstnd, boxed).
function _active!(x::Vector{Float64}, l::Vector{Float64}, u::Vector{Float64},
    nbd::Vector{Int32}, iwhere::Vector{Int}, n::Int)
    prjctd = false
    cnstnd = false
    boxed = true
    @inbounds for i in 1:n
        if nbd[i] > 0
            if nbd[i] <= 2 && x[i] <= l[i]
                if x[i] < l[i]
                    prjctd = true
                    x[i] = l[i]
                end
            elseif nbd[i] >= 2 && x[i] >= u[i]
                if x[i] > u[i]
                    prjctd = true
                    x[i] = u[i]
                end
            end
        end
    end
    @inbounds for i in 1:n
        if nbd[i] != 2
            boxed = false
        end
        if nbd[i] == 0
            iwhere[i] = -1  # always free
        else
            cnstnd = true
            if nbd[i] == 2 && u[i] - l[i] <= 0.0
                iwhere[i] = 3  # always fixed
            else
                iwhere[i] = 0
            end
        end
    end
    return prjctd, cnstnd, boxed
end

# R's cauchy: compute the Generalized Cauchy Point (GCP).
# xcp is filled with the GCP. Returns (nint, info).
function _cauchy!(n::Int, x::Vector{Float64}, l::Vector{Float64}, u::Vector{Float64},
    nbd::Vector{Int32}, g::Vector{Float64}, iorder::Vector{Int}, iwhere::Vector{Int},
    t::Vector{Float64}, d::Vector{Float64}, xcp::Vector{Float64}, m::Int,
    wy::Matrix{Float64}, ws::Matrix{Float64}, sy::Matrix{Float64}, wt::Matrix{Float64},
    theta::Float64, col::Int, head::Int,
    p::Vector{Float64}, c::Vector{Float64}, wbp::Vector{Float64}, v::Vector{Float64},
    sbgnrm::Float64)

    if sbgnrm <= 0.0
        copyto!(xcp, x)
        return 0, 0
    end
    bnded = true
    nfree = n + 1
    nbreak = 0
    ibkmin = 0
    bkmin = 0.0
    col2 = 2 * col
    f1 = 0.0
    nint = 1

    # Zero p
    @inbounds for i in 1:col2
        p[i] = 0.0
    end

    # For each variable: determine bound status, breakpoint, and update p
    tl = 0.0
    tu = 0.0
    @inbounds for i in 1:n
        neggi = -g[i]
        if iwhere[i] != 3 && iwhere[i] != -1
            if nbd[i] <= 2
                tl = x[i] - l[i]
            end
            if nbd[i] >= 2
                tu = u[i] - x[i]
            end
            xlower = nbd[i] <= 2 && tl <= 0.0
            xupper = nbd[i] >= 2 && tu <= 0.0
            iwhere[i] = 0
            if xlower
                if neggi <= 0.0
                    iwhere[i] = 1
                end
            elseif xupper
                if neggi >= 0.0
                    iwhere[i] = 2
                end
            else
                if abs(neggi) <= 0.0
                    iwhere[i] = -3
                end
            end
        end
        pointr = head
        if iwhere[i] != 0 && iwhere[i] != -1
            d[i] = 0.0
        else
            d[i] = neggi
            f1 -= neggi * neggi
            # p += W'e_i * neggi
            for j in 1:col
                p[j] += wy[i, pointr] * neggi
                p[col+j] += ws[i, pointr] * neggi
                pointr = pointr % m + 1
            end
            if nbd[i] <= 2 && nbd[i] != 0 && neggi < 0.0
                # x[i] + d[i] will hit lower bound
                nbreak += 1
                iorder[nbreak] = i
                t[nbreak] = tl / (-neggi)
                if nbreak == 1 || t[nbreak] < bkmin
                    bkmin = t[nbreak]
                    ibkmin = nbreak
                end
            elseif nbd[i] >= 2 && neggi > 0.0
                # x[i] + d[i] will hit upper bound
                nbreak += 1
                iorder[nbreak] = i
                t[nbreak] = tu / neggi
                if nbreak == 1 || t[nbreak] < bkmin
                    bkmin = t[nbreak]
                    ibkmin = nbreak
                end
            else
                # unbounded along this direction
                nfree -= 1
                iorder[nfree] = i
                if abs(neggi) > 0.0
                    bnded = false
                end
            end
        end
    end

    if theta != 1.0
        @inbounds for j in col+1:col2
            p[j] *= theta
        end
    end
    # Initialize GCP = x
    copyto!(xcp, x)
    if nbreak == 0 && nfree == n + 1
        return 0, 0
    end
    # Initialize c = W'(xcp - x) = 0
    @inbounds for j in 1:col2
        c[j] = 0.0
    end

    # Initialize f2
    f2 = -theta * f1
    f2_org = f2
    if col > 0
        info = _bmv!(v, sy, wt, col, m, p)
        if info != 0
            return nint, info
        end
        s = 0.0
        @inbounds for i in 1:col2
            s += v[i] * p[i]
        end
        f2 -= s
    end
    dtm = -f1 / f2
    tsum = 0.0

    if nbreak == 0
        # No breakpoints: locate GCP and return
        @goto gcp_found
    end
    nleft = nbreak
    cauchyiter = 1
    tj = 0.0

    # Main loop: walk through breakpoints
    while true
        tj0 = tj
        if cauchyiter == 1
            tj = bkmin
            ibp = iorder[ibkmin]
        else
            if cauchyiter == 2
                if ibkmin != nbreak
                    t[ibkmin] = t[nbreak]
                    iorder[ibkmin] = iorder[nbreak]
                end
            end
            _hpsolb!(t, iorder, nleft, cauchyiter - 2)
            tj = t[nleft]
            ibp = iorder[nleft]
        end
        dt = tj - tj0
        if dtm < dt
            @goto gcp_found
        end
        # Fix variable ibp at its bound
        tsum += dt
        nleft -= 1
        cauchyiter += 1
        dibp = d[ibp]
        d[ibp] = 0.0
        if dibp > 0.0
            zibp = u[ibp] - x[ibp]
            xcp[ibp] = u[ibp]
            iwhere[ibp] = 2
        else
            zibp = l[ibp] - x[ibp]
            xcp[ibp] = l[ibp]
            iwhere[ibp] = 1
        end
        if nleft == 0 && nbreak == n
            dtm = dt
            @goto gcp_update_c
        end
        # Update f1, f2
        nint += 1
        dibp2 = dibp * dibp
        f1 += dt * f2 + dibp2 - theta * dibp * zibp
        f2 -= theta * dibp2
        if col > 0
            # c += dt * p
            @inbounds for j in 1:col2
                c[j] += dt * p[j]
            end
            # wbp = row of W for variable ibp
            pointr = head
            @inbounds for j in 1:col
                wbp[j] = wy[ibp, pointr]
                wbp[col+j] = theta * ws[ibp, pointr]
                pointr = pointr % m + 1
            end
            # Compute M^{-1}*wbp
            info = _bmv!(v, sy, wt, col, m, wbp)
            if info != 0
                return nint, info
            end
            wmc = 0.0; wmp = 0.0; wmw = 0.0
            @inbounds for j in 1:col2
                wmc += c[j] * v[j]
                wmp += p[j] * v[j]
                wmw += wbp[j] * v[j]
            end
            # p -= dibp * wbp
            @inbounds for j in 1:col2
                p[j] -= dibp * wbp[j]
            end
            f1 += dibp * wmc
            f2 += 2.0 * dibp * wmp - dibp2 * wmw
        end
        f2 = max(f2, eps(Float64) * f2_org)
        if nleft > 0
            dtm = -f1 / f2
            continue
        elseif bnded
            f1 = 0.0; f2 = 0.0; dtm = 0.0
        else
            dtm = -f1 / f2
        end
        break
    end

    @label gcp_found
    dtm = max(dtm, 0.0)
    tsum += dtm
    # Move free variables
    @inbounds for i in 1:n
        xcp[i] += tsum * d[i]
    end

    @label gcp_update_c
    if col > 0
        @inbounds for j in 1:col2
            c[j] += dtm * p[j]
        end
    end
    return nint, 0
end

# R's freev: identify free and active variables at GCP.
# Returns (nfree, nenter, ileave, wrk).
function _freev!(n::Int, nfree_prev::Int, indx::Vector{Int}, iwhere::Vector{Int},
    indx2::Vector{Int}, cnstnd::Bool, updatd::Bool, iter::Int)
    nenter = 0
    ileave = n + 1
    if iter > 0 && cnstnd
        for i in 1:nfree_prev
            k = indx[i]
            if iwhere[k] > 0
                ileave -= 1
                indx2[ileave] = k
            end
        end
        for i in nfree_prev+1:n
            k = indx[i]
            if iwhere[k] <= 0
                nenter += 1
                indx2[nenter] = k
            end
        end
    end
    wrk = (ileave < n + 1) || (nenter > 0) || updatd
    nfree = 0
    iact = n + 1
    @inbounds for i in 1:n
        if iwhere[i] <= 0
            nfree += 1
            indx[nfree] = i
        else
            iact -= 1
            indx[iact] = i
        end
    end
    return nfree, nenter, ileave, wrk
end


# ── Subspace minimization functions ──────────────────────────────────────

# R's formk: form LEL^T factorization of the indefinite K matrix.
# Returns info (0 = success).
function _formk!(n::Int, nsub::Int, indx::Vector{Int}, nenter::Int, ileave::Int,
    indx2::Vector{Int}, iupdat::Int, updatd::Bool,
    wn::Matrix{Float64}, wn1::Matrix{Float64}, m::Int,
    ws::Matrix{Float64}, wy::Matrix{Float64}, sy::Matrix{Float64},
    theta::Float64, col::Int, head::Int)

    if updatd
        if iupdat > m
            # Shift old part of wn1
            for jy in 1:m-1
                js = m + jy
                for i in 1:m-jy
                    wn1[jy+i-1, jy] = wn1[jy+i, jy+1]      # block (1,1)
                    wn1[js+i-1, js] = wn1[js+i, js+1]       # block (2,2)
                end
                for i in 1:m-1
                    wn1[m+i, jy] = wn1[m+i+1, jy+1]         # block (2,1)
                end
            end
        end
        # Put new rows in blocks (1,1), (2,1) and (2,2)
        iy = col
        is_ = m + col
        ipntr = (head + col - 2) % m + 1
        jpntr = head
        for jy in 1:col
            js = m + jy
            temp1 = 0.0; temp2 = 0.0; temp3 = 0.0
            # Free variables: pbegin=1, pend=nsub
            for k in 1:nsub
                k1 = indx[k]
                temp1 += wy[k1, ipntr] * wy[k1, jpntr]
            end
            # Active variables: dbegin=nsub+1, dend=n
            for k in nsub+1:n
                k1 = indx[k]
                temp2 += ws[k1, ipntr] * ws[k1, jpntr]
                temp3 += ws[k1, ipntr] * wy[k1, jpntr]
            end
            wn1[iy, jy] = temp1
            wn1[is_, js] = temp2
            wn1[is_, jy] = temp3
            jpntr = jpntr % m + 1
        end
        # Put new column in block (2,1)
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
            wn1[is_, col] = temp3
        end
        upcl = col - 1
    else
        upcl = col
    end

    # Modify old parts due to changes in free variable set
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
            wn1[iy, jy] += temp1 - temp3
            wn1[is_, js] += -temp2 + temp4
            jpntr = jpntr % m + 1
        end
        ipntr = ipntr % m + 1
    end
    # Modify old parts in block (2,1)
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
                wn1[is_, jy] += temp1 - temp3
            else
                wn1[is_, jy] += -temp1 + temp3
            end
            jpntr = jpntr % m + 1
        end
        ipntr = ipntr % m + 1
    end

    # Form upper triangle of wn (K matrix with compressed indexing)
    m2 = 2 * m
    for iy in 1:col
        is_ = col + iy
        is1 = m + iy
        for jy in 1:iy
            js = col + jy
            js1 = m + jy
            wn[jy, iy] = wn1[iy, jy] / theta
            wn[js, is_] = wn1[is1, js1] * theta
        end
        for jy in 1:iy-1
            wn[jy, is_] = -wn1[is1, jy]
        end
        for jy in iy:col
            wn[jy, is_] = wn1[is1, jy]
        end
        wn[iy, iy] += sy[iy, iy]
    end

    # First Cholesky: factor (1,1) block
    # Create a view of wn as if it has leading dimension m2 (already correct since wn is 2m×2m)
    info = _dpofa!(wn, col)
    if info != 0
        return -1
    end
    # Solve L^{-1} * (1,2) block
    col2 = 2 * col
    for js in col+1:col2
        # Solve R' * x = wn[1:col, js] where R is upper triangular in wn[1:col, 1:col]
        info = _dtrsl_upper_t!(wn, 1, col, view(wn, :, js), 0)
        if info != 0
            return -1
        end
    end
    # Update (2,2) block: add (L^{-1}*B12)' * (L^{-1}*B12)
    for is_ in col+1:col2
        for js in is_:col2
            s = 0.0
            @inbounds for k in 1:col
                s += wn[k, is_] * wn[k, js]
            end
            wn[is_, js] += s
        end
    end
    # Second Cholesky: factor (2,2) block
    info = _dpofa!(view(wn, col+1:col2, col+1:col2), col)
    if info != 0
        return -2
    end
    return 0
end

# R's formt: form T = theta*SS + L*D^{-1}*L' and Cholesky factor.
# Returns info (0 = success).
function _formt!(m::Int, wt::Matrix{Float64}, sy::Matrix{Float64},
    ss::Matrix{Float64}, col::Int, theta::Float64)
    for j in 1:col
        wt[1, j] = theta * ss[1, j]
    end
    for i in 2:col
        for j in i:col
            k1 = min(i, j) - 1
            ddum = 0.0
            for k in 1:k1
                ddum += sy[i, k] * sy[j, k] / sy[k, k]
            end
            wt[i, j] = ddum + theta * ss[i, j]
        end
    end
    info = _dpofa!(wt, col)
    return info != 0 ? -3 : 0
end

# R's cmprlb: compute r = -Z'B(xcp-x) - Z'g.
# Returns info.
function _cmprlb!(n::Int, m::Int, x::Vector{Float64}, g::Vector{Float64},
    ws::Matrix{Float64}, wy::Matrix{Float64}, sy::Matrix{Float64}, wt::Matrix{Float64},
    z::Vector{Float64}, r::Vector{Float64}, wa::Vector{Float64},
    indx::Vector{Int}, theta::Float64, col::Int, head::Int, nfree::Int, cnstnd::Bool)

    if !cnstnd && col > 0
        for i in 1:n
            r[i] = -g[i]
        end
    else
        for i in 1:nfree
            k = indx[i]
            r[i] = -theta * (z[k] - x[k]) - g[k]
        end
        info = _bmv!(wa, sy, wt, col, m, view(wa, 2m+1:4m))
        if info != 0
            return -8
        end
        pointr = head
        for j in 1:col
            a1 = wa[j]
            a2 = theta * wa[col+j]
            for i in 1:nfree
                k = indx[i]
                r[i] += wy[k, pointr] * a1 + ws[k, pointr] * a2
            end
            pointr = pointr % m + 1
        end
    end
    return 0
end

# R's subsm: subspace minimization in the free variable subspace.
# z is modified in place with the minimizer. d is used as workspace.
# Returns (iword, info).
function _subsm!(n::Int, m::Int, nsub::Int, indx::Vector{Int},
    l::Vector{Float64}, u::Vector{Float64}, nbd::Vector{Int32},
    z::Vector{Float64}, d::Vector{Float64},
    ws::Matrix{Float64}, wy::Matrix{Float64}, theta::Float64,
    col::Int, head::Int, wv::Vector{Float64}, wn::Matrix{Float64})

    if nsub <= 0
        return 0, 0
    end
    # Compute wv = W'Z*d
    pointr = head
    @inbounds for i in 1:col
        temp1 = 0.0; temp2 = 0.0
        for j in 1:nsub
            k = indx[j]
            temp1 += wy[k, pointr] * d[j]
            temp2 += ws[k, pointr] * d[j]
        end
        wv[i] = temp1
        wv[col+i] = theta * temp2
        pointr = pointr % m + 1
    end
    # Solve K^{-1} * wv using LEL^T factorization stored in wn
    m2 = 2 * m
    col2 = 2 * col
    # Solve L^T * x = wv (upper triangular transpose)
    # The wn matrix has the factorization in compressed form [1:col, 1:col] and [col+1:2col, col+1:2col]
    # But we need to solve the full 2col×2col system using the block structure
    # R: dtrsl(&wn, &m2, &col2, &wv[1], &c__11, info) → T'*x = b, upper T
    # Here wn is 2m×2m and we use the first 2col rows/cols
    info = _dtrsl_upper_t!(wn, 1, col2, wv, 0)
    if info != 0
        return 0, info
    end
    @inbounds for i in 1:col
        wv[i] = -wv[i]
    end
    # R: dtrsl(&wn, &m2, &col2, &wv[1], &c__1, info) → T*x = b, upper T
    info = _dtrsl_upper!(wn, 1, col2, wv, 0)
    if info != 0
        return 0, info
    end
    # Compute d = (1/theta)*d + (1/theta^2)*Z'W*wv
    pointr = head
    @inbounds for jy in 1:col
        js = col + jy
        for i in 1:nsub
            k = indx[i]
            d[i] += (wy[k, pointr] * wv[jy] / theta + ws[k, pointr] * wv[js])
        end
        pointr = pointr % m + 1
    end
    @inbounds for i in 1:nsub
        d[i] /= theta
    end

    # Backtrack to feasible region
    alpha = 1.0
    temp1 = alpha
    ibd = 0
    @inbounds for i in 1:nsub
        k = indx[i]
        dk = d[i]
        if nbd[k] != 0
            if dk < 0.0 && nbd[k] <= 2
                temp2 = l[k] - z[k]
                if temp2 >= 0.0
                    temp1 = 0.0
                elseif dk * alpha < temp2
                    temp1 = temp2 / dk
                end
            elseif dk > 0.0 && nbd[k] >= 2
                temp2 = u[k] - z[k]
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
            z[k] = u[k]
            d[ibd] = 0.0
        elseif dk < 0.0
            z[k] = l[k]
            d[ibd] = 0.0
        end
    end
    @inbounds for i in 1:nsub
        z[indx[i]] += alpha * d[i]
    end
    iword = alpha < 1.0 ? 1 : 0
    return iword, 0
end

# R's matupd: update WS, WY, SY, SS matrices.
# d = stp * search_direction, r = gnew - gold.
# Returns (itail, col, head, theta).
function _matupd!(n::Int, m::Int, ws::Matrix{Float64}, wy::Matrix{Float64},
    sy::Matrix{Float64}, ss::Matrix{Float64},
    d::Vector{Float64}, r::Vector{Float64},
    itail::Int, iupdat::Int, col::Int, head::Int,
    rr::Float64, dr::Float64, stp::Float64, dtd::Float64)

    if iupdat <= m
        col = iupdat
        itail = (head + iupdat - 2) % m + 1
    else
        itail = itail % m + 1
        head = head % m + 1
    end
    # Update WS and WY at column itail
    @inbounds for i in 1:n
        ws[i, itail] = d[i]
        wy[i, itail] = r[i]
    end
    # theta = y'y / y's
    theta = rr / dr
    # Update SY and SS
    if iupdat > m
        # Shift old information
        for j in 1:col-1
            for i in 1:j
                ss[i, j] = ss[i+1, j+1]
            end
            for i in 1:col-j
                sy[j+i-1, j] = sy[j+i, j+1]
            end
        end
    end
    # Add new row/column
    pointr = head
    for j in 1:col-1
        sy[col, j] = dot(view(ws, :, itail), view(wy, :, pointr))
        ss[j, col] = dot(view(ws, :, pointr), view(ws, :, itail))
        pointr = pointr % m + 1
    end
    if stp == 1.0
        ss[col, col] = dtd
    else
        ss[col, col] = stp * stp * dtd
    end
    sy[col, col] = dr
    return itail, col, head, theta
end


# ── Line search (kept from previous implementation) ──────────────────────

@inline function _feasible_step_cap(x::Vector{Float64}, d::Vector{Float64},
    l::Vector{Float64}, u::Vector{Float64})
    alpha_max = Inf
    @inbounds for i in eachindex(x)
        di = d[i]
        if di < 0.0 && isfinite(l[i])
            if x[i] <= l[i]
                return 0.0
            end
            alpha_max = min(alpha_max, (l[i] - x[i]) / di)
        elseif di > 0.0 && isfinite(u[i])
            if x[i] >= u[i]
                return 0.0
            end
            alpha_max = min(alpha_max, (u[i] - x[i]) / di)
        end
    end
    return alpha_max
end


function _line_search_wolfe!(x::Vector{Float64}, fx::Float64, gx::Vector{Float64},
    d::Vector{Float64}, l::Vector{Float64}, u::Vector{Float64},
    f::Function, g::Function, n::Int;
    c1::Float64=1e-3, c2::Float64=0.9, iter::Int=1,
    xtol::Float64=0.1, boxed::Bool=false,
    max_evals::Int=40, stpmx::Float64=Inf)

    phi0prime = dot(gx, d)
    if !(isfinite(phi0prime)) || phi0prime >= 0.0

        return false, 0.0, fx, gx, x, 0, 0
    end

    alpha_max = _feasible_step_cap(x, d, l, u)
    if isfinite(stpmx) && stpmx < alpha_max
        alpha_max = stpmx
    end
    if !(alpha_max > 0.0)
        return false, 0.0, fx, gx, x, 0, 0
    end

    Dnorm = sqrt(dot(d, d))
    # R's lnsrlb (lbfgsb.c:2518-2523): boxed or iter>0 → stp=1.0
    alpha_init = (iter == 1 && !boxed) ? (1.0 / max(Dnorm, eps())) : 1.0
    alpha = isfinite(alpha_max) ? min(alpha_init, alpha_max) : alpha_init

    xtrial = similar(x)
    gtrial = similar(gx)

    fevals = 0
    gevals = 0

    f_prev = fx
    alpha_prev = 0.0
    f_best = fx
    alpha_best = 0.0
    g_best = copy(gx)

    eval_at = function (alphat)
        @inbounds @. xtrial = x + alphat * d
        ft = f(n, xtrial, nothing)
        fevals += 1
        gt = g(n, xtrial, nothing)
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

            ok, alphaz, fz, gz, ez_fe, ez_ge = _zoom!(x, fx, gx, d, l, u, f, g, n,
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
                return false, 0.0, fx, gx, x, fevals, gevals
            end
        end
        if abs(phialpha) <= c2 * abs(phi0prime)
            @inbounds @. xtrial = x + alpha * d
            return true, alpha, ft, gt, xtrial, fevals, gevals
        end
        if phialpha >= 0.0

            ok, alphaz, fz, gz, ez_fe, ez_ge = _zoom!(x, fx, gx, d, l, u, f, g, n,
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
                return false, 0.0, fx, gx, x, fevals, gevals
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
                return false, 0.0, fx, gx, x, fevals, gevals
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
    return false, 0.0, fx, gx, x, fevals, gevals
end

# Cubic interpolation for line search (More-Thuente style).
function _cubic_min(a1, f1, dphi1, a2, f2, dphi2)
    d1 = dphi1 + dphi2 - 3.0 * (f2 - f1) / (a2 - a1)
    disc = d1 * d1 - dphi1 * dphi2
    disc < 0.0 && return NaN
    d2 = copysign(sqrt(disc), a2 - a1)
    return a2 - (a2 - a1) * (dphi2 + d2 - d1) / (dphi2 - dphi1 + 2.0 * d2)
end

function _zoom!(x, fx, gx, d, l, u, f, g, n,
    alpha_lo, alpha_hi, f_lo, f_hi, phi0prime; c1=1e-3, c2=0.9, xtol=0.1, max_evals=30)
    fevals = 0
    gevals = 0
    xtrial = similar(x)
    gtrial = similar(gx)

    @inbounds @. xtrial = x + alpha_lo * d
    f_lo_eval = f_lo
    g_lo = g(n, xtrial, nothing)
    gevals += 1
    phi_lo = dot(g_lo, d)
    g_lo_saved = copy(g_lo)

    phi_hi = NaN

    eval_at = function (alphat)
        @inbounds @. xtrial = x + alphat * d
        ft = f(n, xtrial, nothing)
        fevals += 1
        gt = g(n, xtrial, nothing)
        gevals += 1
        return ft, gt
    end

    for _ in 1:max_evals
        bracket_max = max(alpha_lo, alpha_hi)
        if bracket_max > 0 && abs(alpha_hi - alpha_lo) <= xtol * bracket_max
            return true, alpha_lo, f_lo_eval, g_lo_saved, fevals, gevals
        end
        alpha_j = _cubic_min(alpha_lo, f_lo_eval, phi_lo, alpha_hi, f_hi, phi_hi)
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


# ── Main L-BFGS-B function ──────────────────────────────────────────────

"""
    lbfgsbmin(f, g, x0; mask=trues(length(x0)), l=nothing, u=nothing, options=LBFGSBOptions())

Limited-memory BFGS with box constraints (L-BFGS-B).

Implements the Byrd-Lu-Nocedal-Zhu (1995) algorithm with Generalized Cauchy Point
and subspace minimization, matching R's `optim(..., method="L-BFGS-B")`.

- `f(n, x, ex)` → scalar objective
- `g(n, x, ex)` → gradient vector (size n)
- `mask` freezes variables by setting [l=u=x0] internally
- `l`, `u` optional bounds (`nothing` = unbounded)
- `options`: `m` (history), `factr` (f_tol = factr * eps()), `pgtol` (projected grad ∞-norm),
             `maxit` (iteration cap), `iprint` (0 silent, >0 prints)

Returns named tuple: `x_opt, f_opt, n_iter, fail, fn_evals, gr_evals, message`.
"""
function lbfgsbmin(f::Function, g::Function, x0::Vector{Float64};
    mask=trues(length(x0)),
    l::Union{Nothing,Vector{Float64}}=nothing,
    u::Union{Nothing,Vector{Float64}}=nothing,
    options::LBFGSBOptions=LBFGSBOptions())

    n = length(x0)
    m = options.m
    if length(mask) != n
        error("mask length must equal x0 length")
    end

    l2 = l === nothing ? fill(-Inf, n) : copy(l)
    u2 = u === nothing ? fill(+Inf, n) : copy(u)
    @inbounds for i in 1:n
        if !mask[i]
            l2[i] = x0[i]
            u2[i] = x0[i]
        end
    end

    # Check for infeasible bounds
    @inbounds for i in 1:n
        if l2[i] > u2[i]
            return (x_opt=copy(x0), f_opt=Inf, n_iter=0,
                fail=52, fn_evals=0, gr_evals=0,
                message="ERROR: NO FEASIBLE SOLUTION")
        end
    end

    nbd = _nbd_from_bounds(l2, u2)

    # Compact L-BFGS storage
    ws = zeros(Float64, n, m)    # S vectors
    wy = zeros(Float64, n, m)    # Y vectors
    sy = zeros(Float64, m, m)    # S'Y
    ss = zeros(Float64, m, m)    # S'S
    wt = zeros(Float64, m, m)    # Cholesky of T
    wn = zeros(Float64, 2m, 2m)  # K matrix
    wn1 = zeros(Float64, 2m, 2m) # N matrix

    theta = 1.0
    col = 0
    head = 1
    itail = 0
    iupdat = 0
    updatd = false

    # Working arrays
    z = Vector{Float64}(undef, n)
    r = Vector{Float64}(undef, n)
    d = Vector{Float64}(undef, n)
    t_bp = Vector{Float64}(undef, n)      # breakpoints
    wa = Vector{Float64}(undef, 8m)        # cauchy/subsm workspace
    p_work = Vector{Float64}(undef, 2m)    # W'd for cauchy
    c_work = Vector{Float64}(undef, 2m)    # W'(xcp-x) for cauchy
    wbp = Vector{Float64}(undef, 2m)       # row of W at breakpoint
    v_work = Vector{Float64}(undef, 2m)    # bmv workspace
    wv = Vector{Float64}(undef, 2m)        # subsm workspace
    indx = Vector{Int}(undef, n)
    iwhere = Vector{Int}(undef, n)
    indx2 = Vector{Int}(undef, n)

    # Initialize
    x = copy(x0)
    prjctd, cnstnd, boxed = _active!(x, l2, u2, nbd, iwhere, n)

    fx = f(n, x, nothing)
    fn_evals = 1
    if !isfinite(fx)
        return (x_opt=x, f_opt=fx, n_iter=0,
            fail=52, fn_evals=fn_evals, gr_evals=0,
            message="ERROR: L-BFGS-B NEEDS FINITE VALUES OF FN")
    end

    # copy() is critical: g() may return a shared buffer
    gx = copy(g(n, x, nothing))
    gr_evals = 1

    # Check for non-finite gradient
    for i in eachindex(gx)
        if !isfinite(gx[i])
            return (x_opt=x, f_opt=fx, n_iter=0,
                fail=52, fn_evals=fn_evals, gr_evals=gr_evals,
                message="ERROR: NON-FINITE GRADIENT")
        end
    end

    pg = similar(gx)
    sbgnrm = _proj_grad!(pg, x, gx, l2, u2, nbd)

    if options.iprint > 0
        println("At iterate     0  f= ", fx, "  |proj g|= ", sbgnrm)
    end
    if sbgnrm <= options.pgtol
        return (x_opt=x, f_opt=fx, n_iter=0,
            fail=0, fn_evals=fn_evals, gr_evals=gr_evals,
            message="CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL")
    end

    f_tol = options.factr * eps(Float64)
    fail = 1
    message = "NEW_X"
    iter = 0
    nfree = n
    nact = 0
    ileave = n + 1
    nenter = 0

    while true
        iter += 1

        wrk = false
        if !cnstnd && col > 0
            # Unconstrained with history: skip GCP, set z = x
            copyto!(z, x)
            wrk = updatd
        else
            # Compute Generalized Cauchy Point
            nint, info = _cauchy!(n, x, l2, u2, nbd, gx, indx2, iwhere,
                t_bp, d, z, m, wy, ws, sy, wt, theta, col, head,
                p_work, c_work, wbp, v_work, sbgnrm)
            if info != 0
                # Singular system: refresh L-BFGS memory
                if options.iprint > 0
                    println("Singular triangular system in cauchy; refreshing memory.")
                end
                col = 0; head = 1; theta = 1.0; iupdat = 0; updatd = false
                continue
            end
            # Identify free/active variables at GCP
            nfree, nenter, ileave, wrk = _freev!(n, nfree, indx, iwhere, indx2,
                cnstnd, updatd, iter - 1)
            nact = n - nfree
        end

        # Subspace minimization
        if nfree != 0 && col != 0
            if wrk
                info = _formk!(n, nfree, indx, nenter, ileave, indx2, iupdat, updatd,
                    wn, wn1, m, ws, wy, sy, theta, col, head)
                if info != 0
                    if options.iprint > 0
                        println("Nonpositive definiteness in formk; refreshing memory.")
                    end
                    col = 0; head = 1; theta = 1.0; iupdat = 0; updatd = false
                    continue
                end
            end
            # For cmprlb, wa[2m+1:4m] stores W'(xcp-x) = c from cauchy
            @inbounds for i in 1:2*col
                wa[2m+i] = c_work[i]
            end
            info = _cmprlb!(n, m, x, gx, ws, wy, sy, wt, z, r, wa,
                indx, theta, col, head, nfree, cnstnd)
            if info == 0
                # d[1:nfree] = reduced gradient r[1:nfree]
                @inbounds for i in 1:nfree
                    d[i] = r[i]
                end
                iword, info = _subsm!(n, m, nfree, indx, l2, u2, nbd, z, d,
                    ws, wy, theta, col, head, wv, wn)
            end
            if info != 0
                if options.iprint > 0
                    println("Singular triangular system in subsm; refreshing memory.")
                end
                col = 0; head = 1; theta = 1.0; iupdat = 0; updatd = false
                continue
            end
        end

        # Search direction: d = z - x
        @inbounds for i in 1:n
            d[i] = z[i] - x[i]
        end

        # Check for zero direction
        dnorm_sq = dot(d, d)
        if dnorm_sq < 1e-32
            fail = 0
            message = "CONVERGENCE: ZERO_SEARCH_DIRECTION"
            break
        end
        dnorm = sqrt(dnorm_sq)

        # Line search
        # R's lnsrlb: save old x in t (we use r as temp), old g saved in old_g
        old_x = copy(x)
        old_g = copy(gx)
        fold = fx

        # Compute max step (R's lnsrlb stpmx logic)
        stpmx = 1e10
        if cnstnd
            if iter == 1
                stpmx = 1.0
            else
                @inbounds for i in 1:n
                    a1 = d[i]
                    if nbd[i] != 0
                        if a1 < 0.0 && nbd[i] <= 2
                            a2 = l2[i] - x[i]
                            if a2 >= 0.0
                                stpmx = 0.0
                            elseif a1 * stpmx < a2
                                stpmx = a2 / a1
                            end
                        elseif a1 > 0.0 && nbd[i] >= 2
                            a2 = u2[i] - x[i]
                            if a2 <= 0.0
                                stpmx = 0.0
                            elseif a1 * stpmx > a2
                                stpmx = a2 / a1
                            end
                        end
                    end
                end
            end
        end

        ok, stp, fnew, gnew, xnew, fe, ge = _line_search_wolfe!(x, fx, gx, d, l2, u2,
            f, g, n; iter=iter, boxed=boxed, stpmx=stpmx)
        fn_evals += fe
        gr_evals += ge

        if !ok
            # Restore previous iterate
            x .= old_x
            gx .= old_g
            fx = fold
            if col == 0
                # Abnormal termination
                fail = 52
                message = "ERROR: ABNORMAL_TERMINATION_IN_LNSRCH"
                break
            else
                # Refresh L-BFGS memory and restart
                if options.iprint > 0
                    println("Bad direction in line search; refreshing memory.")
                end
                col = 0; head = 1; theta = 1.0; iupdat = 0; updatd = false
                continue
            end
        end

        if !isfinite(fnew)
            x .= old_x
            gx .= old_g
            fx = fold
            fail = 52
            message = "ERROR: L-BFGS-B NEEDS FINITE VALUES OF FN"
            break
        end

        x .= xnew
        fx = fnew
        gx .= copy(gnew)

        # Check for non-finite gradient
        has_nonfinite_grad = false
        for i in eachindex(gx)
            if !isfinite(gx[i])
                has_nonfinite_grad = true
                break
            end
        end
        if has_nonfinite_grad
            fail = 52
            message = "ERROR: NON-FINITE GRADIENT"
            break
        end

        sbgnrm = _proj_grad!(pg, x, gx, l2, u2, nbd)

        if options.iprint > 0
            println("At iterate ", iter, "  f= ", fx, "  |proj g|= ", sbgnrm)
        end

        # R checks iter > maxit in the driver BEFORE mainlb's convergence tests.
        # This ensures at least one full iteration runs (maxit=0), and that
        # maxit takes priority over convergence on the boundary iteration.
        if iter > options.maxit
            break  # fail=1, message="NEW_X" (defaults)
        end

        # Convergence tests (only reached if iter <= maxit)
        if sbgnrm <= options.pgtol
            fail = 0
            message = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL"
            break
        end
        ddum = max(abs(fold), abs(fx), 1.0)
        if fold - fx <= f_tol * ddum
            fail = 0
            message = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH"
            break
        end

        # Compute s = xnew - old_x (step), y = gnew - gold (gradient diff)
        @inbounds for i in 1:n
            d[i] = x[i] - old_x[i]  # s vector (already scaled by stp)
            r[i] = gx[i] - old_g[i]  # y vector
        end
        rr = dot(r, r)   # y'y
        dr = dot(r, d)   # y's
        dtd = dot(d, d)  # s's
        ddum_skip = -dot(old_g, d)  # -g_old's

        if dr <= eps(Float64) * ddum_skip
            # Skip L-BFGS update
            updatd = false
        else
            updatd = true
            iupdat += 1
            # d is already the full step s = stp*direction, so dtd = s's.
            # Pass stp=1.0 so matupd doesn't double-scale ss[col,col].
            itail, col, head, theta = _matupd!(n, m, ws, wy, sy, ss, d, r,
                itail, iupdat, col, head, rr, dr, 1.0, dtd)
            info = _formt!(m, wt, sy, ss, col, theta)
            if info != 0
                if options.iprint > 0
                    println("Nonpositive definiteness in formt; refreshing memory.")
                end
                col = 0; head = 1; theta = 1.0; iupdat = 0; updatd = false
            end
        end
    end

    return (x_opt=x, f_opt=fx, n_iter=iter,
        fail=fail, fn_evals=fn_evals, gr_evals=gr_evals,
        message=message)
end
