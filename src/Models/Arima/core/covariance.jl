function update_least_squares!(
    n_parameters::Int,
    input_row::AbstractArray,
    work_row::AbstractArray,
    rhs_value::Float64,
    diag::AbstractArray,
    upper_triangle::AbstractArray,
    rhs_solution::AbstractArray,
)

for i = 1:n_parameters
        work_row[i] = input_row[i]
    end

    tri_idx = 1
    for i = 1:n_parameters
        if work_row[i] != 0.0
            pivot = work_row[i]
            diag_old = diag[i]
            diag_new = diag_old + pivot * pivot
            diag[i] = diag_new
            cos_rotation = diag_new != 0.0 ? diag_old / diag_new : Inf
            sin_rotation = diag_new != 0.0 ? pivot / diag_new : Inf

            for k = (i+1):n_parameters
                elem = work_row[k]
                upper_elem = upper_triangle[tri_idx]
                work_row[k] = elem - pivot * upper_elem
                upper_triangle[tri_idx] = cos_rotation * upper_elem + sin_rotation * elem
                tri_idx += 1
            end

            elem = rhs_value
            rhs_value = elem - pivot * rhs_solution[i]
            rhs_solution[i] = cos_rotation * rhs_solution[i] + sin_rotation * elem

            if diag_old == 0.0
                return
            end
        else
            tri_idx = tri_idx + n_parameters - i
        end
    end

    return
end

function compute_v(phi::AbstractArray, theta::AbstractArray, r::Int)
    p = length(phi)
    q = length(theta)
    num_params = r * (r + 1) ÷ 2
    cross_products = zeros(Float64, num_params)

    pack_idx = 0
    for j = 0:(r-1)
        ma_coef_j = 0.0
        if j == 0
            ma_coef_j = 1.0
        elseif (j - 1) < q && (j - 1) ≥ 0
            ma_coef_j = theta[j-1+1]
        end

        for i = j:(r-1)
            ma_coef_i = 0.0
            if i == 0
                ma_coef_i = 1.0
            elseif (i - 1) < q && (i - 1) ≥ 0
                ma_coef_i = theta[i-1+1]
            end

            cross_products[pack_idx+1] = ma_coef_i * ma_coef_j
            pack_idx += 1
        end
    end
    return cross_products
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

    pack_idx = num_params
    accum_idx = num_params

    for i = 0:(r-1)
        for j = 0:i
            pack_idx -= 1

            res[pack_idx + 1] = V[pack_idx+1]

            if j != 0
                accum_idx -= 1
                res[pack_idx+1] += res[accum_idx+1]
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

    packed_cov = zeros(Float64, r * r)

    upper_triangle = zeros(Float64, nrbar)
    rhs_solution = zeros(Float64, num_params)
    equation_row = zeros(Float64, num_params)
    work_row = zeros(Float64, num_params)

    v_idx = 0
    identity_idx = -1
    n_packed_minus_r = num_params - r
    n_packed_minus_r_plus1 = n_packed_minus_r + 1
    col_offset_j = n_packed_minus_r
    wrap_idx = n_packed_minus_r - 1

    for j = 0:(r-1)

        ar_coef_j = (j < p) ? phi[j+1] : 0.0

        equation_row[col_offset_j+1] = 0.0
        col_offset_j += 1

        row_offset_i = n_packed_minus_r_plus1 + j
        for i = j:(r-1)
            ynext = V[v_idx+1]
            v_idx += 1

            ar_coef_i = (i < p) ? phi[i+1] : 0.0

            if j != (r - 1)
                equation_row[col_offset_j+1] = -ar_coef_i
                if i != (r - 1)
                    equation_row[row_offset_i+1] -= ar_coef_j
                    identity_idx += 1
                    equation_row[identity_idx+1] = -1.0
                end
            end

            equation_row[n_packed_minus_r+1] = -ar_coef_i * ar_coef_j
            wrap_idx += 1
            if wrap_idx >= num_params
                wrap_idx = 0
            end
            equation_row[wrap_idx+1] += 1.0

            update_least_squares!(num_params, equation_row, work_row, ynext, packed_cov, upper_triangle, rhs_solution)

            equation_row[wrap_idx+1] = 0.0
            if i != (r - 1)
                equation_row[row_offset_i+1] = 0.0
                row_offset_i += 1
                equation_row[identity_idx+1] = 0.0
            end
        end
    end

    tri_idx = nrbar - 1
    result_idx = num_params - 1

    for i = 0:(num_params-1)
        accum = rhs_solution[result_idx+1]
        backsolve_idx = num_params - 1
        for j = 0:(i-1)

            accum -= upper_triangle[tri_idx+1] * packed_cov[backsolve_idx+1]

            tri_idx -= 1
            backsolve_idx -= 1
        end
        packed_cov[result_idx+1] = accum
        result_idx -= 1
    end

    diag_save = zeros(Float64, r)
    idx = n_packed_minus_r
    for i = 0:(r-1)
        diag_save[i+1] = packed_cov[idx+1]
        idx += 1
    end

    write_idx = num_params - 1
    read_idx = n_packed_minus_r - 1
    for i = 1:(n_packed_minus_r)
        packed_cov[write_idx+1] = packed_cov[read_idx+1]
        write_idx -= 1
        read_idx -= 1
    end

    for i = 0:(r-1)
        packed_cov[i+1] = diag_save[i+1]
    end

    return packed_cov
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

    cross_products = compute_v(phi, theta, r)

    if r == 1
        return handle_r_equals_1(p, phi)
    end

    if p > 0
        res_flat = handle_p_greater_than_0(cross_products, phi, p, r, num_params, nrbar)
    else
        res_flat = handle_p_equals_0(cross_products, r)
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

    extended_theta = zeros(Float64, r + q)
    @inbounds extended_theta[1] = 1.0
    @inbounds for i in 1:q
        extended_theta[i + 1] = θ[i]
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
                        tLk = extended_theta[L0 - k0 + 1]
                        φ_ik_tLk = φ_ik * tLk

                        for m0 in 0:m_max
                            φ_jm = φ[φ_j_base + m0]
                            φ_product = φ_ik_tLk * φ_jm
                            n_start = m0
                            n_end = m0 + q

                            for n0 in n_start:n_end
                                tnm = extended_theta[n0 - m0 + 1]
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
                val = extended_theta[i_idx]
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
                            acc += φ_ik * extended_theta[j0_L0 + 1] * rrz[L0 - k0]
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
                            acc += φ_jk * extended_theta[i0_L0 + 1] * rrz[L0 - k0]
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
                acc += extended_theta[i0 + k0 + 1] * extended_theta[j0 + k0 + 1]
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
