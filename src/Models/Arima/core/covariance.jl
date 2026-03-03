function update_least_squares!(
    n_parameters::Int,
    xnext::AbstractArray,
    xrow::AbstractArray,
    ynext::Float64,
    d::AbstractArray,
    rbar::AbstractArray,
    thetab::AbstractArray,
)

for i = 1:n_parameters
        xrow[i] = xnext[i]
    end

    ithisr = 1
    for i = 1:n_parameters
        if xrow[i] != 0.0
            xi = xrow[i]
            di = d[i]
            dpi = di + xi * xi
            d[i] = dpi
            cbar = dpi != 0.0 ? di / dpi : Inf
            sbar = dpi != 0.0 ? xi / dpi : Inf

            for k = (i+1):n_parameters
                xk = xrow[k]
                rbthis = rbar[ithisr]
                xrow[k] = xk - xi * rbthis
                rbar[ithisr] = cbar * rbthis + sbar * xk
                ithisr += 1
            end

            xk = ynext
            ynext = xk - xi * thetab[i]
            thetab[i] = cbar * thetab[i] + sbar * xk

            if di == 0.0
                return
            end
        else
            ithisr = ithisr + n_parameters - i
        end
    end

    return
end

function compute_v(phi::AbstractArray, theta::AbstractArray, r::Int)
    p = length(phi)
    q = length(theta)
    num_params = r * (r + 1) ÷ 2
    V = zeros(Float64, num_params)

    ind = 0
    for j = 0:(r-1)
        vj = 0.0
        if j == 0
            vj = 1.0
        elseif (j - 1) < q && (j - 1) ≥ 0
            vj = theta[j-1+1]
        end

        for i = j:(r-1)
            vi = 0.0
            if i == 0
                vi = 1.0
            elseif (i - 1) < q && (i - 1) ≥ 0
                vi = theta[i-1+1]
            end

            V[ind+1] = vi * vj
            ind += 1
        end
    end
    return V
end

function handle_r_equals_1(p::Int, phi::AbstractArray)
    res = zeros(Float64, 1, 1)
    if p == 0
        res[1, 1] = 1.0
    else
        res[1, 1] = 1.0 / (1.0 - phi[1]^2)
    end
    return res
end

function handle_p_equals_0(V::AbstractArray, r::Int)
    num_params = r * (r + 1) ÷ 2
    res = zeros(Float64, r * r)

    ind = num_params
    indn = num_params

    for i = 0:(r-1)
        for j = 0:i
            ind -= 1

            res[ind + 1] = V[ind+1]

            if j != 0
                indn -= 1
                res[ind+1] += res[indn+1]
            end
        end
    end
    return res
end

function handle_p_greater_than_0(
    V::AbstractArray,
    phi::AbstractArray,
    p::Int,
    r::Int,
    num_params::Int,
    nrbar::Int,
)

    res = zeros(Float64, r * r)

    rbar = zeros(Float64, nrbar)
    thetab = zeros(Float64, num_params)
    xnext = zeros(Float64, num_params)
    xrow = zeros(Float64, num_params)

    ind = 0
    ind1 = -1
    npr = num_params - r
    npr1 = npr + 1
    indj = npr
    ind2 = npr - 1

    for j = 0:(r-1)

        phij = (j < p) ? phi[j+1] : 0.0

        xnext[indj+1] = 0.0
        indj += 1

        indi = npr1 + j
        for i = j:(r-1)
            ynext = V[ind+1]
            ind += 1

            phii = (i < p) ? phi[i+1] : 0.0

            if j != (r - 1)
                xnext[indj+1] = -phii
                if i != (r - 1)
                    xnext[indi+1] -= phij
                    ind1 += 1
                    xnext[ind1+1] = -1.0
                end
            end

            xnext[npr+1] = -phii * phij
            ind2 += 1
            if ind2 >= num_params
                ind2 = 0
            end
            xnext[ind2+1] += 1.0

            update_least_squares!(num_params, xnext, xrow, ynext, res, rbar, thetab)

            xnext[ind2+1] = 0.0
            if i != (r - 1)
                xnext[indi+1] = 0.0
                indi += 1
                xnext[ind1+1] = 0.0
            end
        end
    end

    ithisr = nrbar - 1
    im = num_params - 1

    for i = 0:(num_params-1)
        bi = thetab[im+1]
        jm = num_params - 1
        for j = 0:(i-1)

            bi -= rbar[ithisr+1] * res[jm+1]

            ithisr -= 1
            jm -= 1
        end
        res[im+1] = bi
        im -= 1
    end

    xcopy = zeros(Float64, r)
    ind = npr
    for i = 0:(r-1)
        xcopy[i+1] = res[ind+1]
        ind += 1
    end

    ind = num_params - 1
    ind1 = npr - 1
    for i = 1:(npr)
        res[ind+1] = res[ind1+1]
        ind -= 1
        ind1 -= 1
    end

    for i = 0:(r-1)
        res[i+1] = xcopy[i+1]
    end

    return res
end

function unpack_full_matrix(res_flat::AbstractArray, r::Int)
    num_params = r * (r + 1) ÷ 2

    for i = (r-1):-1:1
        for j = (r-1):-1:i

            idx = i * r + j
            res_flat[idx+1] = res_flat[num_params]
            num_params -= 1
        end
    end

    for i = 0:(r-1)
        for j = (i+1):(r-1)

            res_flat[j*r+i+1] = res_flat[i*r+j+1]
        end
    end

    return reshape(res_flat, r, r)
end

function compute_q0_covariance_matrix(phi::AbstractArray, theta::AbstractArray)
    p = length(phi)
    q = length(theta)

    r = max(p, q + 1)
    num_params = r * (r + 1) ÷ 2
    nrbar = num_params * (num_params - 1) ÷ 2

    V = compute_v(phi, theta, r)

    if r == 1
        return handle_r_equals_1(p, phi)
    end

    if p > 0
        res_flat = handle_p_greater_than_0(V, phi, p, r, num_params, nrbar)
    else
        res_flat = handle_p_equals_0(V, r)
    end

    res_full = unpack_full_matrix(res_flat, r)
    return res_full
end

function compute_q0_bis_covariance_matrix(phi::AbstractVector{<:Real},
                                          theta::AbstractVector{<:Real},
                                          tol::Real = eps(Float64))

    φ = Float64.(phi)
    θ = Float64.(theta)

    p = length(φ)
    q = length(θ)
    r = max(p, q + 1)

    ttheta = zeros(Float64, r + q)
    @inbounds ttheta[1] = 1.0
    @inbounds for i in 1:q
        ttheta[i + 1] = θ[i]
    end

    P = zeros(Float64, r, r)

    if p > 0
        r2 = max(p + q, p + 1)

        tphi = Vector{Float64}(undef, p + 1)
        @inbounds tphi[1] = 1.0
        @inbounds for i in 1:p
            tphi[i + 1] = -φ[i]
        end

        Γ = zeros(Float64, r2, r2)

        @inbounds for j0 in 0:(r2-1)
            j_idx = j0 + 1
            for i0 in j0:(r2-1)
                d = i0 - j0
                if d <= p
                    Γ[j_idx, i0 + 1] += tphi[d + 1]
                end
            end
        end

        @inbounds for i0 in 0:(r2-1)
            i_idx = i0 + 1
            for j0 in 1:(r2-1)
                s = i0 + j0
                if s <= p
                    Γ[j0 + 1, i_idx] += tphi[s + 1]
                end
            end
        end

        g = zeros(Float64, r2)
        @inbounds g[1] = 1.0
        u = let
            ok = true
            κ = Inf
            try
                κ = cond(Γ)
            catch
                ok = false
            end

            if ok && isfinite(κ) && κ < 1/tol
                Γ \ g
            else
                (Γ + tol * I) \ g
            end
        end

        @inbounds for i0 in 0:(r-1)
            i_idx = i0 + 1
            φ_i_base = i0 + 1
            k_max = p - 1 - i0

            for j0 in i0:(r-1)
                j_idx = j0 + 1
                φ_j_base = j0 + 1
                m_max = p - 1 - j0
                acc = 0.0

                for k0 in 0:k_max
                    φ_ik = φ[φ_i_base + k0]
                    L_start = k0
                    L_end = k0 + q

                    for L0 in L_start:L_end
                        tLk = ttheta[L0 - k0 + 1]
                        φ_ik_tLk = φ_ik * tLk

                        for m0 in 0:m_max
                            φ_jm = φ[φ_j_base + m0]
                            φ_product = φ_ik_tLk * φ_jm
                            n_start = m0
                            n_end = m0 + q

                            for n0 in n_start:n_end
                                tnm = ttheta[n0 - m0 + 1]
                                u_idx = abs(L0 - n0) + 1
                                acc += φ_product * tnm * u[u_idx]
                            end
                        end
                    end
                end
                P[i_idx, j_idx] += acc
            end
        end

        rrz = zeros(Float64, q)
        if q > 0
            @inbounds for i0 in 0:(q-1)
                i_idx = i0 + 1
                val = ttheta[i_idx]
                jstart = max(0, i0 - p)
                for j0 in jstart:(i0-1)
                    val -= rrz[j0 + 1] * tphi[i0 - j0 + 1]
                end
                rrz[i_idx] = val
            end
        end

        @inbounds for i0 in 0:(r-1)
            i_idx = i0 + 1
            k_max_i = p - 1 - i0

            for j0 in i0:(r-1)
                j_idx = j0 + 1
                k_max_j = p - 1 - j0
                acc = 0.0

                for k0 in 0:k_max_i
                    φ_ik = φ[i0 + k0 + 1]
                    L_start = k0 + 1
                    L_end = k0 + q

                    for L0 in L_start:L_end
                        j0_L0 = j0 + L0
                        if j0_L0 < q + 1
                            acc += φ_ik * ttheta[j0_L0 + 1] * rrz[L0 - k0]
                        end
                    end
                end

                for k0 in 0:k_max_j
                    φ_jk = φ[j0 + k0 + 1]
                    L_start = k0 + 1
                    L_end = k0 + q

                    for L0 in L_start:L_end
                        i0_L0 = i0 + L0
                        if i0_L0 < q + 1
                            acc += φ_jk * ttheta[i0_L0 + 1] * rrz[L0 - k0]
                        end
                    end
                end

                P[i_idx, j_idx] += acc
            end
        end
    end

    @inbounds for i0 in 0:(r-1)
        i_idx = i0 + 1
        for j0 in i0:(r-1)
            j_idx = j0 + 1
            k_max = q - j0
            acc = 0.0
            @simd for k0 in 0:k_max
                acc += ttheta[i0 + k0 + 1] * ttheta[j0 + k0 + 1]
            end
            P[i_idx, j_idx] += acc
        end
    end

    @inbounds for i in 1:r
        for j in (i+1):r
            P[j, i] = P[i, j]
        end
    end

    return P
end
