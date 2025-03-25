# C_TSconv
function time_series_convolution(a::AbstractArray, b::AbstractArray)
    na = length(a)
    nb = length(b)
    nab = na + nb - 1
    ab = zeros(Float64, nab)
    
    for i in 1:na
        for j in 1:nb
            ab[i + j - 1] += a[i] * b[j]
        end
    end
    return ab
end

# Helper for compute_q0
function apply_inclusion_transform!(
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


            if abs(ynext) > 1000 || abs(thetab[i]) > 1000
                ynext = sign(ynext) * 1000
                thetab[i] = sign(thetab[i]) * 1000
            end

            if abs(ynext) < 1e-5
                ynext = 0.0  
            end

            if di == 0.0
                return
            end
        else
            ithisr = ithisr + n_parameters - i
        end
    end

    return
end

# Helper for compute_q0
function compute_v(phi::AbstractArray, theta::AbstractArray, r::Int)
    p = length(phi)
    q = length(theta)
    num_params = r * (r + 1) ÷ 2
    V = zeros(Float64, num_params)

    ind = 0 
    for j in 0:(r-1)
        vj = 0.0
        if j == 0
            vj = 1.0
        elseif (j - 1) < q && (j - 1) ≥ 0
            vj = theta[j-1+1]
        end

        for i in j:(r-1)
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

# Helper for compute_q0
function handle_r_equals_1(p::Int, phi::AbstractArray)
    res = zeros(Float64, 1, 1)
    if p == 0
        
        res[1, 1] = 1.0
    else
        
        res[1, 1] = 1.0 / (1.0 - phi[1]^2)
    end
    return res
end

# Helper for compute_q0
function handle_p_equals_0(V::AbstractArray, r::Int)
    num_params = r * (r + 1) ÷ 2
    res = zeros(Float64, r * r)

    ind = num_params
    indn = num_params
    
    for i in 0:(r-1)
        for j in 0:i
            ind -= 1
            
            res[ind] = V[ind]

            if j != 0
                indn -= 1
                res[ind] = 2.0 * res[ind]
            end
        end
    end

    return res
end

# Helper for compute_q0
function handle_p_greater_than_0(V::AbstractArray,
    phi::AbstractArray,
    p::Int,
    r::Int,
    num_params::Int,
    nrbar::Int)
    
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
    
    for j in 0:(r-1)
        
        phij = (j < p) ? phi[j+1] : 0.0
        
        xnext[indj+1] = 0.0
        indj += 1

        indi = npr1 + j
        for i in j:(r-1)
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
            
            apply_inclusion_transform!(num_params, xnext, xrow, ynext, res, rbar, thetab)

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

    for i in 0:(num_params-1)
        bi = thetab[im+1]
        jm = num_params - 1
        for j in 0:(i-1)
            
            bi -= rbar[ithisr+1] * res[jm+1]
            
            ithisr -= 1
            jm -= 1
        end
        res[im+1] = bi
        im -= 1
    end

    xcopy = zeros(Float64, r)
    ind = npr
    for i in 0:(r-1)
        xcopy[i+1] = res[ind+1]
        ind += 1
    end

    ind = num_params - 1
    ind1 = npr - 1
    for i in 1:(npr)
        res[ind+1] = res[ind1+1]
        ind -= 1
        ind1 -= 1
    end

    for i in 0:(r-1)
        res[i+1] = xcopy[i+1]
    end

    return res
end

# Helper for compute_q0
function unpack_full_matrix(res_flat::AbstractArray, r::Int)
    num_params = r * (r + 1) ÷ 2
    
    for i in (r-1):-1:1
        for j in (r-1):-1:i
            
            idx = i * r + j 
            res_flat[idx+1] = res_flat[num_params]
            num_params -= 1
        end
    end

    for i in 0:(r-1)
        for j in (i+1):(r-1)
            
            res_flat[j*r+i+1] = res_flat[i*r+j+1]
        end
    end

    return reshape(res_flat, r, r)
end

# Helper for compute_q0
function compute_q0(phi::AbstractArray, theta::AbstractArray)
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

function compute_q0_bis(phi::AbstractArray, theta::AbstractArray, tol::Float64 = 1e-7)
    p = length(phi)
    q = length(theta)
    r = max(p, q + 1)

    ttheta = zeros(Float64, q + 1)
    ttheta[1] = 1.0
    for i in 2:(q+1)
        ttheta[i] = theta[i - 1]
    end

    P = zeros(Float64, r, r)

    if p > 0
        tphi = zeros(Float64, p + 1)
        tphi[1] = 1.0
        for i in 2:(p+1)
            tphi[i] = -phi[i - 1]
        end

        r2 = max(p+q, p+1)

        Gam = zeros(Float64, r2, r2)

        for jC in 0:(r2-1)
            for iC in jC:(r2-1)
                if (iC - jC) < (p+1)
                    Gam[jC+1, iC+1] += tphi[(iC - jC) + 1]
                end
            end
        end

        for iC in 0:(r2-1)
            for jC in 1:(r2-1)
                if (iC + jC) < (p+1)
                    Gam[jC+1, iC+1] += tphi[(iC + jC) + 1]
                end
            end
        end

        g = zeros(Float64, r2)
        g[1] = 1.0

        u = pinv(Gam; rtol=tol) * g

        for iC in 0:(r-1)
            for jC in iC:(r-1)
                for k in 0:(p-1)
                    if (iC + k) < p
                        for L in k:(k + q)
                            if (L - k) <= q
                                for m in 0:(p-1)
                                    if (jC + m) < p
                                        for n in m:(m + q)
                                            if (n - m) <= q
                                                
                                                idxPhi1    = (iC + k) + 1
                                                idxPhi2    = (jC + m) + 1
                                                idxTtheta1 = (L - k) + 1
                                                idxTtheta2 = (n - m) + 1
                                                idxU       = abs(L - n) + 1
                                                # Check bounds
                                                if 1 <= idxPhi1    <= p    &&
                                                   1 <= idxPhi2    <= p    &&
                                                   1 <= idxTtheta1 <= (q+1) &&
                                                   1 <= idxTtheta2 <= (q+1) &&
                                                   1 <= idxU       <= r2
                                                    P[iC+1, jC+1] += phi[idxPhi1] * phi[idxPhi2] *
                                                                     ttheta[idxTtheta1] * ttheta[idxTtheta2] *
                                                                     u[idxU]
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end

        rrz = zeros(Float64, q)
        if q > 0
            for iC in 0:(q-1)
                rrz[iC+1] = ttheta[iC+1]
                for jC in max(0, iC - p):(iC-1)
                    rrz[iC+1] -= rrz[jC+1] * tphi[(iC - jC) + 1]
                end
            end
        end

        for iC in 0:(r-1)
            for jC in iC:(r-1)
                for k in 0:(p-1)
                    if (iC + k) < p
                        for L in (k+1):q
                            if (jC + L) < (q+1)
                                idxPhi = (iC + k) + 1
                                idxTth = (jC + L) + 1
                                idxRrz = (L - k - 1) + 1
                                if 1 <= idxPhi <= p &&
                                   1 <= idxTth <= (q+1) &&
                                   1 <= idxRrz <= q
                                    P[iC+1, jC+1] += phi[idxPhi] * ttheta[idxTth] * rrz[idxRrz]
                                end
                            end
                        end
                    end
                end

                for k in 0:(p-1)
                    if (jC + k) < p
                        for L in (k+1):q
                            if (iC + L) < (q+1)
                                idxPhi = (jC + k) + 1
                                idxTth = (iC + L) + 1
                                idxRrz = (L - k - 1) + 1
                                if 1 <= idxPhi <= p &&
                                   1 <= idxTth <= (q+1) &&
                                   1 <= idxRrz <= q
                                    P[iC+1, jC+1] += phi[idxPhi] * ttheta[idxTth] * rrz[idxRrz]
                                end
                            end
                        end
                    end
                end
            end
        end
    end 

    for iC in 0:(r-1)
        for jC in iC:(r-1)
            for k in 0:q
                if (iC + k) < (q+1) && (jC + k) < (q+1)
                    P[iC+1, jC+1] += ttheta[(iC + k) + 1] * ttheta[(jC + k) + 1]
                end
            end
        end
    end

    for iC in 0:(r-1)
        for jC in (iC+1):(r-1)
            P[jC+1, iC+1] = P[iC+1, jC+1]
        end
    end

    return P
end

# R C lib equivelent 
function arima_like(y, phi, theta, delta, a, P, Pn, up::Int, use_resid::Bool)
    n = length(y)
    rd = length(a)
    p = length(phi)
    q = length(theta)
    d = length(delta)
    r = rd - d

    sumlog = 0.0
    ssq = 0.0
    nu = 0

    P = copy(reshape(P, 1, :))[:] 
    Pnew = copy(reshape(Pn, 1, :))[:]

    anew = similar(a)
    M = similar(a)

    mm = d > 0 ? zeros(rd, rd) : nothing
    rsResid = use_resid ? fill(NaN, n) : nothing

    for l in 1:n
        for i in 1:r
            tmp = (i < r) ? a[i+1] : 0.0
            tmp += (i <= p) ? phi[i] * a[1] : 0.0
            anew[i] = tmp
        end

        if d > 0
            anew[(r+2):rd] .= a[(r+1):(rd-1)]
            tmp = a[1]
            for i in 1:d
                tmp += delta[i] * a[r + i]
            end
            anew[r+1] = tmp
        end

        if l > up + 1
            if d == 0
                for i in 1:r
                    vi = (i == 1) ? 1.0 : (i-1 <= q ? theta[i-1] : 0.0)
                    for j in 1:r
                        tmp = 0.0
                        tmp += (j == 1) ? vi : (j-1 <= q ? vi * theta[j-1] : 0.0)
                        tmp += (i <= p && j <= p) ? phi[i] * phi[j] * P[1] : 0.0
                        tmp += (i < r && j < r) ? P[i+1 + r*(j-1)] : 0.0
                        tmp += (i <= p && j < r) ? phi[i] * P[j+1] : 0.0
                        tmp += (j <= p && i < r) ? phi[j] * P[i+1] : 0.0
                        Pnew[i + r*(j-1)] = tmp
                    end
                end
            else
                for i in 1:r
                    for j in 1:rd
                        tmp = 0.0
                        tmp += (i <= p) ? phi[i] * P[1 + (j-1)*rd] : 0.0
                        tmp += (i < r) ? P[i+1 + (j-1)*rd] : 0.0
                        mm[i,j] = tmp
                    end
                end

                for j in 1:rd
                    tmp = P[1 + (j-1)*rd]
                    for k in 1:d
                        tmp += delta[k] * P[r + k + (j-1)*rd]
                    end
                    mm[r+1,j] = tmp
                end

                for i in 2:d
                    mm[r+i, :] .= mm[r+i-1, :]
                end

                for i in 1:rd
                    for j in 1:r
                        tmp = 0.0
                        tmp += (j <= p) ? phi[j] * mm[i,1] : 0.0
                        tmp += (j < r) ? mm[i,j+1] : 0.0
                        Pnew[i + (j-1)*rd] = tmp
                    end
                end

                for i in 1:rd
                    tmp = mm[i,1]
                    for k in 1:d
                        tmp += delta[k] * mm[i, r+k]
                    end
                    Pnew[i + r*rd] = tmp
                end

                for i in 2:d
                    for j in 1:rd
                        Pnew[j + (r+i-1)*rd] = mm[j, r+i-1]
                    end
                end

                for i in 1:(q+1)
                    vi = (i == 1) ? 1.0 : theta[i-1]
                    for j in 1:(q+1)
                        Pnew[i + (j-1)*rd] += vi * ((j == 1) ? 1.0 : theta[j-1])
                    end
                end
            end
        end

        if !isnan(y[l])
            resid = y[l] - anew[1]
            for i in 1:d
                resid -= delta[i] * anew[r+i]
            end

            for i in 1:rd
                tmp = Pnew[i]
                for j in 1:d
                    tmp += Pnew[i + (r+j-1)*rd] * delta[j]
                end
                M[i] = tmp
            end

            gain = M[1]
            for j in 1:d
                gain += delta[j] * M[r+j]
            end

            if gain < 1e4
                nu += 1
                ssq += gain != 0.0 ? resid^2 / gain : Inf
                sumlog += log(gain)
            end            

            if use_resid
                rsResid[l] = gain != 0.0 ? resid / sqrt(gain) : Inf
            end

            for i in 1:rd
                a[i] = anew[i] + (gain != 0.0 ? M[i] * resid / gain : Inf)
            end

            for i in 1:rd, j in 1:rd
                P[i + (j-1)*rd] = Pnew[i + (j-1)*rd] - (gain != 0.0 ? M[i]*M[j]/gain : Inf)
            end
        else
            a .= anew
            P .= Pnew
            if use_resid
                rsResid[l] = NaN
            end
        end
    end

    return ssq, sumlog, nu, rsResid
end

function arima_css(y::AbstractArray,
    arma::Vector{Int},
    phi::AbstractArray,
    theta::AbstractArray,
    ncond::Int)
    n = length(y)
    p = length(phi)
    q = length(theta)

    w = copy(y)

    for _ in 1:arma[6]
        for l in n:-1:2
            w[l] -= w[l-1]
        end
    end

    ns = arma[5]
    for _ in 1:arma[7]
        for l in n:-1:(ns+1)
            w[l] -= w[l-ns]
        end
    end

    resid = Vector{Float64}(undef, n)
    for i in 1:ncond
        resid[i] = 0.0
    end

    ssq = 0.0
    nu = 0

    for l in (ncond+1):n
        tmp = w[l]
        for j in 1:p
            if (l - j) < 1
                continue
            end
            tmp -= phi[j] * w[l-j]
        end

        jmax = min(l - ncond, q)
        for j in 1:jmax
            if (l - j) < 1
                continue
            end
            tmp -= theta[j] * resid[l-j]
        end

        resid[l] = tmp

        if !isnan(tmp)
            nu += 1
            ssq += tmp^2
        end
    end

    return ssq / nu, resid
end

function arima_param_transform!(p::Int, raw::AbstractVector, new::AbstractVector)
    if p > 100
        throw(ArgumentError("The function can only transform 100 parameters in arima0"))
    end

    new[1:p] .= tanh.(raw[1:p])
    work = copy(new[1:p])

    for j = 2:p
        a = new[j]
        for k = 1:(j-1)
            work[k] -= a * new[j-k]
        end
        new[1:(j-1)] .= work[1:(j-1)]
    end
end

function arima_gradient_transform(x::AbstractArray, arma::AbstractArray)
    eps = 1e-3
    mp, mq, msp = arma[1:3]
    n = length(x)
    y = Matrix{Float64}(I, n, n)

    w1 = Vector{Float64}(undef, 100)
    w2 = Vector{Float64}(undef, 100)
    w3 = Vector{Float64}(undef, 100)

    if mp > 0

        for i = 1:mp
            w1[i] = x[i]
        end

        arima_param_transform!(mp, w1, w2)

        for i = 1:mp
            w1[i] += eps
            arima_param_transform!(mp, w1, w3)
            for j = 1:mp
                y[i, j] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end

    if msp > 0
        v = mp + mq
        for i = 1:msp
            w1[i] = x[i+v]
        end
        arima_param_transform!(msp, w1, w2)
        for i = 1:msp
            w1[i] += eps
            arima_param_transform!(msp, w1, w3)
            for j = 1:msp
                y[i+v, j+v] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end
    return y
end

#_ARIMA_undoPars
function arima_undo_params(x::AbstractArray, arma::AbstractArray)
    mp, mq, msp = arma[1:3]
    res = copy(x)
    if mp > 0
        arima_param_transform!(mp, x, res)
    end
    v = mp + mq
    if msp > 0
        arima_param_transform!(msp, @view(x[v+1:end]), @view(res[v+1:end]))
    end
    return res
end

# ARIMA_transPars
function arima_transpar(
    params_in::AbstractArray,
    arma::Vector{Int},
    trans::Bool;
    partrans = arima_param_transform!,
)
    mp, mq, msp, msq, ns = arma
    p = mp + ns * msp
    q = mq + ns * msq

    phi = zeros(Float64, p)
    theta = zeros(Float64, q)
    params = copy(params_in)

    if trans
        if mp > 0
            partrans(mp, params_in, params)
        end
        v = mp + mq
        if msp > 0
            partrans(msp, params_in[v+1:end], params[v+1:end])
        end
    end

    if ns > 0
        phi[1:mp] .= params[1:mp]
        theta[1:mq] .= params[mp+1:mp+mq]

        for j = 0:(msp-1)
            phi[(j+1)*ns] += params[mp+mq+j+1]
            for i = 0:(mp-1)
                phi[((j+1)*ns)+(i+1)] -= params[i+1] * params[mp+mq+j+1]
            end
        end

        for j = 0:(msq-1)
            theta[(j+1)*ns] += params[mp+mq+msp+j+1]
            for i = 0:(mq-1)
                theta[((j+1)*ns)+(i+1)] += params[mp+i+1] * params[mp+mq+msp+j+1]
            end
        end
    else
        phi[1:mp] .= params[1:mp]
        theta[1:mq] .= params[mp+1:mp+mq]
    end

    return (phi, theta)
end

function inverse_parameter_transform(phi::AbstractArray, p::Int)
    if p > 100
        throw(ArgumentError("The function can only transform 100 parameters in arima0"))
    end
    new = copy(phi[1:p])
    work = copy(new)
    for j = p:-1:2
        a = new[j]
        for i = 1:(j-1)
            work[i] = (new[i] + a * new[j-i]) / (1 - a^2)
        end
        for i = 1:(j-1)
            new[i] = work[i]
        end
    end
    for i = 1:p
        new[i] = atanh(new[i])
    end
    return new
end

function arima_inverse_transform(x::AbstractArray, arma::Vector{Int})
    mp, mq, msp = arma[1:3]
    y = copy(x)
    if mp > 0
        y[1:mp] = inverse_parameter_transform(x[1:mp], mp)
    end
    v = mp + mq
    if msp > 0
        y[v+1:v+msp] = inverse_parameter_transform(x[v+1:v+msp], msp)
    end
    return y
end

function build_delta(order::Int, seasonal::Int, seasonal_period::Int)
    delta = [1.0]
    for _ in 1:order
        delta = time_series_convolution(delta, [1.0, -1.0])
    end
    for _ in 1:seasonal
        seasonal_filter = [1.0; zeros(seasonal_period - 1); -1.0]
        delta = time_series_convolution(delta, seasonal_filter)
    end
    return -delta[2:end]
end

using Polynomials

function invert_ma(ma::Vector{Float64})
    q = length(ma)
    coeffs = [1.0; ma]
    q0 = findlast(!=(0.0), coeffs) - 1

    if q0 == 0
        return ma
    end

    poly_coeffs = coeffs[1:q0+1]
    poly_coeffs = [1; poly_coeffs]
    rts = roots(Polynomial(poly_coeffs))

    ind = abs.(rts) .< 1.0

    if all(.!ind)
        return ma
    end
    if q0 == 1
        return vcat(1.0 / ma[1], zeros(q - q0))
    end
    rts[ind] .= 1.0 ./ rts[ind]
    x = [1.0]
    for r in rts
        x = vcat(x, 0.0) .- vcat(0.0, x) ./ r
    end
    return vcat(real.(x[2:end]), zeros(q - q0))
end


function is_stationary(ar::Vector{Float64})
    p = findlast(!=(0.0), -ar) - 1

    if p == 0
        return true
    end

    poly_coeffs = ar[1:p]
    poly_coeffs = [1.0; poly_coeffs]
    rts = roots(Polynomial(poly_coeffs))
    return all(abs.(rts) .> 1.0)
end

function update_arima_model!(
    mod::Dict, 
    phi::Vector{Float64}, 
    theta::Vector{Float64}, 
    SSinit::Bool = true
)
    p = length(phi)
    q = length(theta)
    mod[:phi] = phi
    mod[:theta] = theta
    r = max(p, q + 1)

    if p > 0
        mod[:T][1:p, 1] = phi
    end

    if r > 1
        mod[:Pn][1:r, 1:r] .= SSinit ? compute_q0(phi, theta) : compute_q0_bis(phi, theta)
    else
        mod[:Pn][1, 1] = (p > 0) ? 1 / (1 - phi[1]^2) : 1
    end

    mod[:a] .= 0.0
    return mod
end


function arma_loglik(
    p::Vector{Float64},
    coef::Vector{Float64},
    mask::Vector{Bool},
    x::Vector{Float64},
    xreg::Matrix{Float64},
    mod::Dict{Symbol,Any},
    arma::Tuple{Int,Int},
    narma::Int,
    ncxreg::Int,
    trans::Bool,
)

    par = copy(coef)
    par[mask] .= p

    phi, theta = arima_transpar(par, arma, trans)

    try
        mod = update_arima_model!(mod, phi, theta)
    catch
        return float(typemax(Float64))
    end

    if ncxreg > 0
        β = par[narma+1:narma+ncxreg]
        x = x .- xreg * β
    end

    ssq, sumlog, nu, _ = arima_like(x, mod, 0, false)
    s2 = ssq / nu

    return 0.5 * (log(s2) + sumlog / nu)
end


function arma_css(
    p::Vector{Float64},
    fixed::Vector{Float64},
    mask::Vector{Bool},
    x::Vector{Float64},
    xreg::Matrix{Float64},
    arma::Tuple{Int, Int},
    narma::Int,
    ncxreg::Int,
    ncond::Int
)
    par = copy(fixed)
    par[mask] .= p

    phi, theta = arima_transpar(par, arma, false)

    if ncxreg > 0
        β = par[narma+1 : narma+ncxreg]
        x = x .- xreg * β
    end

    res = arima_css(x, arma, phi, theta, ncond, false)

    return 0.5 * log(res)
end

function fit_state_space(
    x::Vector{Float64},
    xreg::Matrix{Float64},
    coef::Vector{Float64},
    narma::Int,
    ncxreg::Int,
    mod::Dict{Symbol, Any}
)
    if ncxreg > 0
        β = coef[narma+1 : narma+ncxreg]
        x = x .- xreg * β
    end

    return arima_like(x, mod, 0, true)
end


function pdq(p::Int = 0, d::Int = 0, q::Int = 0)
    return (p, d, q)
end

function seasonal_pdq(p::Int = 0, d::Int = 0, q::Int = 0, period::Union{Int, Nothing} = nothing)
    return Dict(:order => (p, d, q), :period => period)
end

struct ArimaFit
    coef::Vector{Float64}
    sigma2::Float64
    var_coef::Matrix{Float64}
    mask::Vector{Bool}
    loglik::Float64
    aic::Union{Float64, Nothing}
    residuals::Vector{Float64}
    arma::Vector{Int}
    convergence_code::Int
    n_cond::Int
    nobs::Int
    model::Dict{Symbol, Any}
end

function arima(
    x::Vector{Float64};
    order = (0, 0, 0),
    seasonal = Dict(:order => (0, 0, 0), :period => 1),
    xreg = Matrix{Float64}(undef, length(x), 0),
    include_mean = true,
    transform_pars = true,
    fixed = nothing,
    init = nothing,
    method = "CSS-ML",
    n_cond = nothing,
    SSinit = "Gardner1980",
    optim_method = :BFGS,
    optim_control = Dict(),
    kappa = 1e6,
)

    n = length(x)
    seasonal_period = seasonal[:period] == 0 ? 1 : seasonal[:period]

    arma = [
        order[1],
        order[3],
        seasonal[:order][1],
        seasonal[:order][3],
        seasonal_period,
        order[2],
        seasonal[:order][2],
    ]

    narma = sum(arma[1:4])
    Delta = build_delta(order[2], seasonal[:order][2], seasonal_period)
    nd = order[2] + seasonal[:order][2]

    if include_mean && nd == 0
        xreg = hcat(ones(n), xreg)
    end

    ncxreg = size(xreg, 2)

    if fixed === nothing
        fixed = fill(NaN, narma + ncxreg)
    end

    mask = isnan.(fixed)
    no_optim = !any(mask)
    if no_optim
        transform_pars = false
    end

    init0 = zeros(narma)
    parscale = ones(narma)

    if ncxreg > 0
        β = xreg \ x
        init0 = vcat(init0, β)
        parscale = vcat(parscale, 10 .* std(x))
    end

    init === nothing && (init = init0)
    init = copy(init)
    isnan.(init) .&& (init .= init0)

    if method == "ML"
        if arma[1] > 0 && !is_stationary(init[1:arma[1]])
            error("non-stationary AR part")
        end
        if arma[3] > 0 && !is_stationary(init[sum(arma[1:2])+1:sum(arma[1:3])])
            error("non-stationary seasonal AR part")
        end
        if transform_pars
            init = arima_inverse_transform(init, arma) 
        end
    end

    coef = copy(fixed)

    if !haskey(optim_control, :parscale)
        optim_control[:parscale] = parscale[mask]
    end

    if method == "CSS" || method == "CSS-ML"
        if n_cond === nothing
            n_cond =
                order[2] +
                seasonal[:order][2] * seasonal_period +
                max(order[1], seasonal[:order][1] * seasonal_period)
        end
        res = if no_optim
            (
                converged = true,
                minimizer = Float64[],
                minimum = arma_(
                    Float64[],
                    fixed,
                    mask,
                    x,
                    xreg,
                    arma,
                    narma,
                    ncxreg,
                    n_cond,
                ),
            )
        else
            optimize(
                p -> arma_css(p, fixed, mask, x, xreg, arma, narma, ncxreg, n_cond),
                init[mask],
                Optim.BFGS();
                store_trace = false,
                iterations = 1000,
            )
        end
        coef[mask] .= Optim.minimizer(res)
        trarma = arima_transpar(coef, arma, false)
        mod = make_arima(trarma[1], trarma[2], Delta, kappa, SSinit == "Gardner1980")
        val = arima_css(x, arma, trarma[1], trarma[2], n_cond, true)
        sigma2 = val[1]
        hess_inv = Optim.hessian(res)
        var_coef = no_optim ? zeros(0, 0) : inv(hess_inv * n)
    else
        if method == "CSS-ML"
            css_res = optimize(
                p -> arma_css(p, fixed, mask, x, xreg, arma, narma, ncxreg, n_cond),
                init[mask],
                Optim.BFGS();
                iterations = 1000,
            )
            init[mask] .= Optim.minimizer(css_res)
        end
        if transform_pars
            init = arima_inverse_transform(init, arma)
            if arma[2] > 0
                init[arma[1]+1:arma[1]+arma[2]] = invert_ma(init[arma[1]+1:arma[1]+arma[2]])
            end
            if arma[4] > 0
                idx = sum(arma[1:3])+1:sum(arma[1:4])
                init[idx] = invert_ma(init[idx])
            end
        end
        trarma = arima_transpar(init, arma, transform_pars)
        mod = make_arima(trarma[1], trarma[2], Delta, kappa, SSinit == "Gardner1980")

        res = if no_optim
            (
                converged = true,
                minimizer = Float64[],
                minimum = arma_loglik(
                    Float64[],
                    coef,
                    mask,
                    x,
                    xreg,
                    mod,
                    arma,
                    narma,
                    ncxreg,
                    transform_pars,
                ),
            )
        else
            optimize(
                p -> arma_loglik(
                    p,
                    coef,
                    mask,
                    x,
                    xreg,
                    mod,
                    arma,
                    narma,
                    ncxreg,
                    transform_pars,
                ),
                init[mask],
                Optim.BFGS();
                store_trace = false,
                iterations = 1000,
            )
        end

        coef[mask] .= Optim.minimizer(res)
        if transform_pars
            coef = arima_undo_params(coef, arma)
        end
        trarma = arima_transpar(coef, arma, false)
        mod = make_arima(trarma[1], trarma[2], Delta, kappa, SSinit == "Gardner1980")
        val = fit_state_space(x, xreg, coef, narma, ncxreg, mod)
        sigma2 = val[1][1] / (length(x) - length(Delta))
        hess_inv = Optim.hessian(res)
        var_coef = no_optim ? zeros(0, 0) : inv(hess_inv * (length(x) - length(Delta)))
    end

    n_used = length(x) - length(Delta)
    loglik = -0.5 * (2 * n_used * res.minimum + n_used + n_used * log(2π))
    aic = method == "CSS" ? nothing : -2 * loglik + 2 * count(mask)
    residuals = val[2]

    return ArimaFit(
        coef,
        sigma2,
        var_coef,
        mask,
        loglik,
        aic,
        residuals,
        arma,
        0,
        n_cond,
        n_used,
        mod,
    )
end
