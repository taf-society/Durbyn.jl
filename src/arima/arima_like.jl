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