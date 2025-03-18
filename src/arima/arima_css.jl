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
