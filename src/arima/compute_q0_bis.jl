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
