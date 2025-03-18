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