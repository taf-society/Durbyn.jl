function make_arima(
    phi::Vector{Float64},
    theta::Vector{Float64},
    Delta::Vector{Float64};
    kappa::Float64 = 1e6,
    SSinit::String = "Gardner1980",
    tol::Float64 = eps(Float64),
)
    p = length(phi)
    q = length(theta)
    r = max(p, q + 1)
    d = length(Delta)
    rd = r + d
    Z = vcat([1.0], zeros(r - 1), Delta)
    T = zeros(Float64, rd, rd)
    if p > 0
        for i = 1:p
            T[i, 1] = phi[i]
        end
    end
    if r > 1
        for i = 2:r
            T[i-1, i] = 1.0
        end
    end
    if d > 0
        T[r+1, :] = Z'
        if d > 1
            for i = 2:d
                T[r+i, r+i-1] = 1.0
            end
        end
    end
    if q < r - 1
        theta = vcat(theta, zeros(r - 1 - q))
    end
    R = vcat([1.0], theta, zeros(d))
    V = R * R'
    h = 0.0
    a = zeros(Float64, rd)
    P = zeros(Float64, rd, rd)
    Pn = zeros(Float64, rd, rd)
    if r > 1
        if SSinit == "Gardner1980"
            Pn[1:r, 1:r] = compute_q0(phi, theta)
        elseif SSinit == "Rossignol2011"
            Pn[1:r, 1:r] = compute_q0_bis(phi, theta, tol)
        else
            throw(ArgumentError("Invalid value for SSinit: $SSinit"))
        end
    else
        if p > 0
            Pn[1, 1] = 1.0 / (1.0 - phi[1]^2)
        else
            Pn[1, 1] = 1.0
        end
    end
    if d > 0
        for i = r+1:r+d
            Pn[i, i] = kappa
        end
    end
    return (
        phi = phi,
        theta = theta,
        Delta = Delta,
        Z = Z,
        a = a,
        P = P,
        T = T,
        V = V,
        h = h,
        Pn = Pn,
    )
end