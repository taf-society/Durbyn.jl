# Global variables
const NONE = 0
const ADD = 1
const MULT = 2
const DAMPED = 1
const TOL = 1.0e-10
const HUGEN = 1.0e10
const NA = -99999.0
const smalno = eps(Float64)
const _PHI_LOWER = 0.8
const _PHI_UPPER = 0.98

function ets_base(y, n, x, m,
    error, trend, season,
    alpha, beta,
    gamma, phi, e,
    amse, nmse)

    oldb = 0.0
    olds = zeros(max(24, m))
    s = zeros(max(24, m))
    f = zeros(30)
    denom = zeros(30)

    if m < 1
        m = 1
    end
    if nmse > 30
        nmse = 30
    end

    nstates = m * (season > NONE) + 1 + (trend > NONE)

    # Copy initial state components
    l = x[1]
    if trend > NONE
        b = x[2]
    else
        b = 0.0
    end

    if season > NONE
        for j in 1:m
            s[j] = x[(trend>NONE)+j+1]
        end
    end

    lik = 0.0
    lik2 = 0.0
    for j in 1:nmse
        amse[j] = 0.0
        denom[j] = 0.0
    end

    for i in 1:n
        # Copy previous state
        oldl = l
        if trend > NONE
            oldb = b
        end
        if season > NONE
            for j in 1:m
                olds[j] = s[j]
            end
        end

        # One step forecast
        forecast_ets_base(oldl, oldb, olds, m, trend, season, phi, f, nmse)

        if abs(f[1] - NA) < TOL
            lik = NA
            return lik
        end

        if error == ADD
            e[i] = y[i] - f[1]
        else
            if abs(f[1]) < TOL
                f_0 = f[1] + TOL
            else
                f_0 = f[1]
            end
            e[i] = (y[i] - f[1]) / f_0
        end

        for j in 1:nmse
            if (i + j - 1) <= n
                denom[j] += 1.0
                tmp = y[i+j-1] - f[j]
                amse[j] = (amse[j] * (denom[j] - 1.0) + (tmp * tmp)) / denom[j]
            end
        end

        # Update state
        l, b, s = update_ets_base(oldl, l, oldb, b, olds, s, m, trend, season, alpha, beta, gamma, phi, y[i])

        # Store new state
        x[nstates*i+1] = l
        if trend > NONE
            x[nstates*i+2] = b
        end
        if season > NONE
            for j in 1:m
                x[nstates*i+(trend>NONE)+j+1] = s[j]
            end
        end

        lik += e[i] * e[i]
        val = abs(f[1])
        if val > 0.0
            lik2 += log(val)
        else
            lik2 += log(val + 1e-8)
        end
    end

    if lik > 0.0
        lik = n * log(lik)
    else
        lik = n * log(lik + 1e-8)
    end

    if error == MULT
        lik += 2 * lik2
    end

    return lik
end

function forecast_ets_base(l, b, s, m, trend, season, phi, f, h)
    phistar = phi
    for i in 1:h
        if trend == NONE
            f[i] = l
        elseif trend == ADD
            f[i] = l + phistar * b
        elseif b < 0
            f[i] = NA
        else
            f[i] = l * (b^phistar)
        end

        j = (m - 1 - (i - 1)) % m

        if season == ADD
            #f[i] = f[i] + s[j+1]
            f[i] += s[j + 1]
        elseif season == MULT
            #f[i] = f[i] * s[j+1]
            f[i] *= s[j + 1]
        end

        if i < h
            phistar += phi == 1.0 ? 1.0 : phi^(i + 1)
        end
    end
end

function update_ets_base(oldl, l, oldb, b, olds, s, m, trend, season, alpha, beta, gamma, phi, y)
    # New Level
    if trend == NONE
        q = oldl
        phib = 0
    elseif trend == ADD
        phib = phi * oldb
        q = oldl + phib
    elseif abs(phi - 1.0) < TOL
        phib = oldb
        q = oldl * oldb
    else
        phib = oldb^phi
        q = oldl * phib
    end

    # Season
    if season == NONE
        p = y
    elseif season == ADD
        p = y - olds[m]
    else
        if abs(olds[m]) < TOL
            p = HUGEN
        else
            p = y / olds[m]
        end
    end

    l = q + alpha * (p - q)

    # New Growth
    if trend > NONE
        if trend == ADD
            r = l - oldl
        else
            if abs(oldl) < TOL
                r = HUGEN
            else
                r = l / oldl
            end
        end
        b = phib + (beta / alpha) * (r - phib)
    end

    # New Seasonal
    if season > NONE
        if season == ADD
            t = y - q
        else # if season == MULT
            if abs(q) < TOL
                t = HUGEN
            else
                t = y / q
            end
        end
        s[1] = olds[m] + gamma * (t - olds[m]) # s[t] = s[t - m] + gamma * (t - s[t - m])
        for j in 2:m
            s[j] = olds[j-1] # s[t] = s[t]
        end
    end

    return l, b, s
end

function simulate_ets_base(x, m, error, trend, season, alpha, beta, gamma, phi, h, y, e)
    oldb = 0.0
    olds = zeros(24)
    s = zeros(24)
    f = zeros(10)
    
    if m > 24 && season > NONE
        println("I am here 1")
        return
    elseif m < 1
        m = 1
    end
    
    l = x[1]
    if trend > NONE
        b = x[2]
    end
    
    if season > NONE
        for j in 1:m
            s[j] = x[(trend > NONE) + j + 1]
        end
    end
    
    for i in 1:h
        oldl = l
        if trend > NONE
            oldb = b
        end
        if season > NONE
            for j in 1:m
                olds[j] = s[j]
            end
        end
        
        forecast_ets_base(oldl, oldb, olds, m, trend, season, phi, f, 1)
        
        if abs(f[1] - NA) < TOL
            y[1] = NA
            println("I am here 2")
            return
        end
        
        if error == ADD
            y[i] = f[1] + e[i]
        else
            y[i] = f[1] * (1.0 + e[i])
        end
        
        # Update state
        l, b, s = update_ets_base(oldl, l, oldb, b, olds, s, m, trend, season, alpha, beta, gamma, phi, y[i])
    end
end

function forecast(x::AbstractArray, m::Int, trend::Int, season::Int, phi::Float64, h::Int, f::Vector{Float64})

    if (m > 24) && (season > NONE)
        return
    elseif m < 1
        m = 1
    end

    l = x[1]
    b = ifelse(trend > NONE, x[2], 0.0)
    s = zeros(Float64, 24)

    if season > NONE
        offset = ifelse(trend > NONE, 2, 1)
        for j in 1:m
            s[j] = x[offset + j]
        end
    end

    forecast_ets_base(l, b, s, m, trend, season, phi, f, h)
end