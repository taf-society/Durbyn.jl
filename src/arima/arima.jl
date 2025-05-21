using LinearAlgebra
export arima, ArimaFit, PDQ, ArimaCoef
import Base: show

include("src/diff.jl")
include("src/utils.jl")
include("src/optim/nmmin.jl")
include("src/optim/optim_hessian.jl")

struct ArimaCoef
    ar::Vector{Float64}
    ma::Vector{Float64}
    sar::Vector{Float64}
    sma::Vector{Float64}
    intercept::Vector{Float64}
end

function show(io::IO, coef::ArimaCoef)
    println(io, "ARIMA Coefficients:")
    if !isempty(coef.ar)
        println(
            io,
            "  AR : ",
            join(["ar$(i)=$(round(v, digits=4))" for (i, v) in enumerate(coef.ar)], ", "),
        )
    end
    if !isempty(coef.ma)
        println(
            io,
            "  MA : ",
            join(["ma$(i)=$(round(v, digits=4))" for (i, v) in enumerate(coef.ma)], ", "),
        )
    end
    if !isempty(coef.sar)
        println(
            io,
            "  SAR: ",
            join(["sar$(i)=$(round(v, digits=4))" for (i, v) in enumerate(coef.sar)], ", "),
        )
    end
    if !isempty(coef.sma)
        println(
            io,
            "  SMA: ",
            join(["sma$(i)=$(round(v, digits=4))" for (i, v) in enumerate(coef.sma)], ", "),
        )
    end
    if !isempty(coef.intercept)
        println(
            io,
            "  INTERCEPT: ",
            join(
                ["x$(i)=$(round(v, digits=4))" for (i, v) in enumerate(coef.intercept)],
                ", ",
            ),
        )
    end
end


"""
    ArimaFit

A struct containing the results of an ARIMA model fit. This type holds all relevant output from the estimation process, including coefficients, variance estimates, likelihood information, and model structure.

# Fields

- `coef::Vector{Float64}`  
  A flat vector of estimated AR, MA, seasonal AR, seasonal MA, and regression coefficients. This corresponds to the complete set of model parameters. You can extract and interpret names from the `model[:names]` field.

- `sigma2::Float64`  
  The estimated variance of the model innovations (errors), typically the maximum likelihood estimate (MLE).

- `var_coef::Matrix{Float64}`  
  The estimated variance-covariance matrix of the coefficients in `coef`.

- `mask::Vector{Bool}`  
  A logical vector indicating which parameters were estimated (vs fixed or excluded). This is used internally for model inference.

- `loglik::Float64`  
  The maximized log-likelihood value of the (possibly differenced) data.

- `aic::Union{Float64, Nothing}`  
  The Akaike Information Criterion (AIC) for the fitted model, derived from the log-likelihood. This is only meaningful when the model is fit using maximum likelihood (`method = "ML"`). If not applicable, this field is `nothing`.

- `residuals::Vector{Float64}`  
  The residuals (estimated innovations) from the fitted ARIMA model.

- `arma::Vector{Int}`  
  A compact encoding of the model specification. This vector typically contains:  
  `[p, q, P, Q, s, d, D]`, where:
    - `p`: number of non-seasonal AR terms  
    - `q`: number of non-seasonal MA terms  
    - `P`: number of seasonal AR terms  
    - `Q`: number of seasonal MA terms  
    - `s`: seasonal period  
    - `d`: order of non-seasonal differencing  
    - `D`: order of seasonal differencing  

- `convergence_code::Int`  
  The return code from the optimization routine. A value of `0` typically indicates successful convergence.

- `n_cond::Int`  
  The number of initial observations not used due to conditioning in the likelihood computation (e.g., due to differencing or initial values).

- `nobs::Int`  
  The number of observations actually used for parameter estimation (after differencing or trimming).

- `model::Dict{Symbol,Any}`  
  A dictionary containing model metadata. Keys may include:
    - `:names` — names of the coefficients
    - `:arma` — the raw arma vector
    - `:call` — a string representation of the function call
    - `:series` — the name of the original time series
    - Other implementation-specific values such as design matrices or internal state components

"""
struct ArimaFit
    coef::ArimaCoef
    sigma2::Float64
    var_coef::Matrix{Float64}
    mask::Vector{Bool}
    loglik::Float64
    aic::Union{Float64,Nothing}
    arma::Vector{Int}
    residuals::Vector{Float64}
    convergence_code::Bool
    n_cond::Int
    nobs::Int
    model::NamedTuple
    xreg::Any
end

function Base.show(io::IO, fit::ArimaFit)
    println(io, "ARIMA Fit Summary")
    println(io, "-----------------")

    println(io, "Coefficients:")
    println(io, "  AR:   ", isempty(fit.coef.ar) ? "None" : join(fit.coef.ar, ", "))
    println(io, "  MA:   ", isempty(fit.coef.ma) ? "None" : join(fit.coef.ma, ", "))
    println(io, "  SAR:  ", isempty(fit.coef.sar) ? "None" : join(fit.coef.sar, ", "))
    println(io, "  SMA:  ", isempty(fit.coef.sma) ? "None" : join(fit.coef.sma, ", "))
    println(
        io,
        "  INTERCEPT: ",
        isempty(fit.coef.intercept) ? "None" : join(fit.coef.intercept, ", "),
    )

    println(io, "\nSigma²: ", fit.sigma2)
    println(io, "Log-likelihood: ", fit.loglik)
    println(io, "AIC: ", isnothing(fit.aic) ? "N/A" : fit.aic)
    println(io, "ARMA Order: (p, d, q) = ", fit.arma)
    println(io, "Residuals (first 5): ", join(fit.residuals[1:min(end, 5)], ", "))
    println(io, "Convergence Code: ", fit.convergence_code)
    println(io, "Number of Conditional Terms: ", fit.n_cond)
    println(io, "Number of Observations: ", fit.nobs)
    println(io, "Mask: ", join(fit.mask, ", "))

    println(io, "\nVariance-Covariance Matrix:")
    show(io, "text/plain", fit.var_coef)

    println(io, "\n\nModel Info:")
    for (k, v) in pairs(fit.model)
        println(io, "  $(k): $(v)")
    end
end

"""
A struct representing the parameters of an ARIMA model: autoregressive (`p`), differencing (`d`), and moving average (`q`) terms.

### Fields
- `p::Int`: Number of autoregressive (AR) terms.
- `d::Int`: Degree of differencing.
- `q::Int`: Number of moving average (MA) terms.

### Example
```julia
pdq_instance = PDQ(1, 0, 1)
println(pdq_instance)  # Output: PDQ(1, 0, 1)
```
"""
struct PDQ
    p::Int
    d::Int
    q::Int

    function PDQ(p::Int, d::Int, q::Int)
        if p < 0 || d < 0 || q < 0
            throw(
                ArgumentError(
                    "All PDQ parameters must be non-negative integers. Got: p=$p, d=$d, q=$q",
                ),
            )
        end
        new(p, d, q)
    end
end

function partrans(p::Int, raw::AbstractVector, new::AbstractVector)
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


function arima_gradtrans(x::AbstractArray, arma::AbstractArray)
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

        partrans(mp, w1, w2)

        for i = 1:mp
            w1[i] += eps
            partrans(mp, w1, w3)
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
        partrans(msp, w1, w2)
        for i = 1:msp
            w1[i] += eps
            partrans(msp, w1, w3)
            for j = 1:msp
                y[i+v, j+v] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end
    return y
end

function arima_undopars(x::AbstractArray, arma::AbstractArray)
    mp, mq, msp = arma[1:3]
    res = copy(x)
    if mp > 0
        partrans(mp, x, res)
    end
    v = mp + mq
    if msp > 0
        partrans(msp, @view(x[v+1:end]), @view(res[v+1:end]))
    end
    return res
end

function tsconv(a::AbstractArray, b::AbstractArray)
    na = length(a)
    nb = length(b)
    nab = na + nb - 1
    ab = zeros(Float64, nab)

    for i = 1:na
        for j = 1:nb
            ab[i+j-1] += a[i] * b[j]
        end
    end
    return ab
end

function inclu2(
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

function inv_par_trans(ϕ::AbstractVector)
    p = length(ϕ)
    new = Array{Float64}(undef, p)
    copy!(new, ϕ)
    work = similar(new)
    # backward Durbin–Levinson
    for j in p:-1:2
        a = new[j]
        denom = 1 - a^2
        @assert denom ≠ 0 "Encountered unit root at j=$j (a=±1)."
        for k in 1:j-1
            work[k] = (new[k] + a * new[j-k]) / denom
        end
        new[1:j-1] = work[1:j-1]
    end
    return atanh.(new)
end



function arima_invtrans(θ::AbstractVector, arma::AbstractVector{Int})
    mp, mq, msp = arma
    n = length(θ)
    v = mp + mq
    @assert v + msp ≤ n "Sum mp+mq+msp exceeds length(θ)"
    raw = Array{Float64}(undef, n)
    copy!(raw, θ)
    transformed = copy(raw)

    # non‐seasonal AR
    if mp > 0
        transformed[1:mp] = inv_par_trans(raw[1:mp])
    end

    # seasonal AR
    if msp > 0
        transformed[v+1:v+msp] = inv_par_trans(raw[v+1:v+msp])
    end

    return transformed
end

# Helper for getQ0
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


# Helper for getQ0
function handle_r_equals_1(p::Int, phi::AbstractArray)
    res = zeros(Float64, 1, 1)
    if p == 0

        res[1, 1] = 1.0
    else

        res[1, 1] = 1.0 / (1.0 - phi[1]^2)
    end
    return res
end


# Helper for getQ0
function handle_p_equals_0(V::AbstractArray, r::Int)
    num_params = r * (r + 1) ÷ 2
    res = zeros(Float64, r * r)

    ind = num_params
    indn = num_params

    for i = 0:(r-1)
        for j = 0:i
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


# Helper for getQ0
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

            inclu2(num_params, xnext, xrow, ynext, res, rbar, thetab)

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

# Helper for getQ0
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

function getQ0(phi::AbstractArray, theta::AbstractArray)
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

function arima_transpar(params_in::AbstractArray, arma::Vector{Int}, trans::Bool)
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

function arima_css(y::AbstractArray, arma::Vector{Int}, phi::AbstractArray, theta::AbstractArray, ncond::Int)
    n = length(y)
    p = length(phi)
    q = length(theta)

    w = copy(y)

    for _ = 1:arma[6]
        for l = n:-1:2
            w[l] -= w[l-1]
        end
    end

    ns = arma[5]
    for _ = 1:arma[7]
        for l = n:-1:(ns+1)
            w[l] -= w[l-ns]
        end
    end

    resid = Vector{Float64}(undef, n)
    for i = 1:ncond
        resid[i] = 0.0
    end

    ssq = 0.0
    nu = 0

    for l = (ncond+1):n
        tmp = w[l]
        for j = 1:p
            if (l - j) < 1
                continue
            end
            tmp -= phi[j] * w[l-j]
        end

        jmax = min(l - ncond, q)
        for j = 1:jmax
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

 function make_arima(phi::Vector{Float64}, theta::Vector{Float64}, Delta::Vector{Float64}; kappa::Float64=1e6, SSinit::String="Gardner1980", tol::Float64=eps(Float64))
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
            Pn[1:r, 1:r] = getQ0(phi, theta)
        elseif SSinit == "Rossignol2011"
            Pn[1:r, 1:r] = getQ0bis(phi, theta, tol)
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
        phi=phi,
        theta=theta,
        Delta=Delta,
        Z=Z,
        a=a,
        P=P,
        T=T,
        V=V,
        h=h,
        Pn=Pn,
    )
end


function arima_like(y,
                    phi,
                    theta,
                    delta,
                    a,
                    Pflat,
                    Pnflat,
                    up::Int,
                    use_resid::Bool)
    # dimensions
    n  = length(y)
    rd = length(a)
    p  = length(phi)
    q  = length(theta)
    d  = length(delta)
    r  = rd - d

    # stats
    sumlog = 0.0
    ssq    = 0.0
    nu     = 0

    # operate in-place
    # P    = Pflat
    # Pnew = Pnflat

    P = copy(reshape(Pflat, 1, :))[:]
    Pnew = copy(reshape(Pnflat, 1, :))[:]

    # scratch
    anew   = similar(a)
    M      = similar(a)
    mm     = d > 0 ? zeros(rd, rd) : nothing
    rsResid= use_resid ? fill(NaN, n) : nothing

    for l in 1:n
        # state prediction
        for i in 1:r
            tmp = (i < r) ? a[i+1] : 0.0
            tmp += (i <= p) ? phi[i] * a[1] : 0.0
            anew[i] = tmp
        end
        if d > 0
            anew[(r+2):rd] .= a[(r+1):(rd-1)]
            tmp = a[1]
            for i = 1:d
                tmp += delta[i] * a[r+i]
            end
            anew[r+1] = tmp
        end

        # covariance prediction
        if l > up+1
            if d == 0
                for i = 1:r
                    vi = (i == 1) ? 1.0 : (i - 1 <= q ? theta[i-1] : 0.0)
                    for j = 1:r
                        tmp = 0.0
                        tmp += (j == 1) ? vi : (j - 1 <= q ? vi * theta[j-1] : 0.0)
                        tmp += (i <= p && j <= p) ? phi[i] * phi[j] * P[1] : 0.0
                        tmp += (i < r && j < r) ? P[i+1+r*(j-1)] : 0.0
                        tmp += (i <= p && j < r) ? phi[i] * P[j+1] : 0.0
                        tmp += (j <= p && i < r) ? phi[j] * P[i+1] : 0.0
                        Pnew[i+r*(j-1)] = tmp
                    end
                end
            else
                for i = 1:r
                    for j = 1:rd
                        tmp = 0.0
                        tmp += (i <= p) ? phi[i] * P[1+(j-1)*rd] : 0.0
                        tmp += (i < r) ? P[i+1+(j-1)*rd] : 0.0
                        mm[i, j] = tmp
                    end
                end
                
                for j in 1:rd
                    tmp = P[1+(j-1)*rd]
                    for k = 1:d
                        tmp += delta[k] * P[r+k+(j-1)*rd]
                    end
                    mm[r+1, j] = tmp
                end
                # 
                for k in 2:d, j in 1:rd
                    mm[r+k,j] = P[(r+k-1) + (j-1)*rd]
                end

                for i = 1:rd
                    for j = 1:r
                        tmp = 0.0
                        tmp += (j <= p) ? phi[j] * mm[i, 1] : 0.0
                        tmp += (j < r) ? mm[i, j+1] : 0.0
                        Pnew[i+(j-1)*rd] = tmp
                    end
                end

                for i = 1:rd
                    tmp = mm[i, 1]
                    for k = 1:d
                        tmp += delta[k] * mm[i, r+k]
                    end
                    Pnew[i+r*rd] = tmp
                end

                for i = 2:d
                    for j = 1:rd
                        Pnew[j+(r+i-1)*rd] = mm[j, r+i-1]
                    end
                end
                
                for i in 1:(q+1), j in 1:(q+1)
                    vi = (i == 1) ? 1.0 : theta[i-1]
                    Pnew[i + (j-1)*rd] += vi * ((j == 1) ? 1.0 : theta[j-1])
                end
            end
        end

        # measurement update
        if !isnan(y[l])
            resid = y[l] - anew[1]
            for k in 1:d
                resid -= delta[k] * anew[r+k]
            end

            for i in 1:rd
                tmp = Pnew[i]
                for k in 1:d
                    tmp += Pnew[i + (r+k-1)*rd] * delta[k]
                end
                M[i] = tmp
            end
            gain = M[1]
            for k in 1:d
                gain += delta[k] * M[r+k]
            end

            # original <= 1e4 guard
            if gain < 1e4
                nu += 1
                ssq += resid^2 / gain
                sumlog += log(gain)
            end
            if use_resid
                rsResid[l] = resid / sqrt(gain)
            end

            for i in 1:rd
                a[i] = anew[i] + M[i] * resid / gain
            end
            for i in 1:rd, j in 1:rd
                P[i + (j-1)*rd] = Pnew[i + (j-1)*rd] - M[i]*M[j]/gain
            end
        else
            a .= anew
            P .= Pnew
            if use_resid
                rsResid[l] = NaN
            end
        end
    end

    # return matching R signature
    if use_resid
        return ssq, sumlog, nu, rsResid
    else
        return ssq, sumlog, nu
    end
end


function arima(x::AbstractArray,
    m;
    order::PDQ = PDQ(0, 0, 0),
    seasonal::PDQ = PDQ(0, 0, 0),
    xreg = nothing,
    include_mean = true,
    transform_pars = true,
    fixed = nothing,
    init = nothing,
    method = "CSS",
    n_cond = nothing,
    SSinit = "Gardner1980",
    optim_method = :BFGS,
    optim_control = Dict(:maxiter => 100),
    kappa = 1e6,)

    # Internal helper: convolution
    function TS_add(a, b)
        return TSconv(a, b)
    end

    SS_G = SSinit == "Gardner1980"

    # Update ARIMA structure
    function upARIMA(mod, phi, theta)
        p = length(phi)
        q = length(theta)
        mod.phi = phi
        mod.theta = theta
        r = max(p, q + 1)

        if p > 0
            mod.T[1:p, 1] .= phi
        end

        if r > 1
            mod.Pn[1:r, 1:r] .= SS_G ? getQ0(phi, theta) : getQ0bis(phi, theta, 0.0)
        else
            mod.Pn[1, 1] = (p > 0) ? 1 / (1 - phi[1]^2) : 1.0
        end

        mod.a .= 0.0
        return mod
    end

    # ARIMA likelihood
    function arimaSS(y, mod)
        return arima_like(y, mod.phi, mod.theta, mod.Delta, mod.a, mod.P, mod.Pn, 0, true)
    end

    # Objective for ML optimization
    function armafn(p, trans)
        par = copy(coef)
        par[mask] = p
        trarma = arima_transpar(par, arma, trans)

        try
            Z = upARIMA(mod, trarma[1], trarma[2])
        catch
            @warn "Updating arima failed"
            return typemax(Float64)
        end

        if ncxreg > 0
            x = x .- xreg * par[narma+1:end]
        end
        res = arima_like(x, Z.phi, Z.theta, Z.Delta, Z.a, Z.P, Z.Pn, 0, false)

        s2 = res[1] / res[3]
        return 0.5 * (log(s2) + res[2] / res[3])
    end

    # Conditional sum of squares objective
    function armaCSS(p)
        par = copy(fixed)
        par[mask] .= p
        trarma = arima_transpar(par, arma, false)

        if ncxreg > 0
            x = x .- xreg * par[narma+1:end]
        end

        ross, _ = arima_css(x, arma, trarma[1], trarma[2], ncond)
        return 0.5 * log(ross)
    end

    # Check AR polynomial stationarity
    function arCheck(ar)
        v = vcat(1.0, -ar...)
        last_nz = findlast(x -> x != 0.0, v)
        p = last_nz === nothing ? 0 : last_nz - 1
        if p == 0
            return true
        end

        coeffs = vcat(1.0, -ar[1:p]...)
        rts = roots(Polynomial(coeffs))

        return all(abs.(rts) .> 1.0)
    end

    # Invert MA polynomial
    function maInvert(ma)
        q = length(ma)
        cdesc = vcat(1.0, ma...)
        nz = findall(x -> x != 0.0, cdesc)
        q0 = isempty(nz) ? 0 : maximum(nz) - 1
        if q0 == 0
            return ma
        end
        cdesc_q = cdesc[1:q0+1]
        rts = roots(Polynomial(cdesc_q))
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
            x = vcat(x, 0.0) .- (vcat(0.0, x) ./ r)
        end
        return vcat(real.(x[2:end]), zeros(q - q0))
    end

    SSinit = match_arg(SSinit, ["Gardner1980", "Rossignol2011"])
    method = match_arg(method, ["CSS-ML", "ML", "CSS"])

    n = length(x)
    y = copy(x)
    xreg_original = xreg === nothing ? nothing : copy(xreg)

    arma = [order.p, order.q, seasonal.p, seasonal.q, m, order.d, seasonal.d]

    narma = sum(arma[1:4])

    # Build Delta
    Delta = [1.0]

    for _ = 1:order.d
        Delta = tsconv(Delta, [1.0, -1.0])
    end

    for _ = 1:seasonal.d
        seasonal_filter = [1.0; zeros(m - 1); -1.0]
        Delta = tsconv(Delta, seasonal_filter)
    end
    
    Delta = -Delta[2:end]

    nd = order.d + seasonal.d
    n_used = length(na_omit(x)) - length(Delta)

    # Build build xreg
    if isnothing(xreg)
        xreg = Matrix{Float64}(undef, n, 0)
        ncxreg = 0
    else
        if size(xreg, 1) != n
            throw(ArgumentError("Lengths of x and xreg do not match!"))
        end
        ncxreg = size(xreg, 2)
    end

    nmxreg = ["ex_$(i)" for i = 1:ncxreg]

    if include_mean && nd == 0
        intercept = ones(Float64, n, 1)

        if ncxreg == 0
            xreg = intercept
        else
            xreg = hcat(intercept, xreg)
        end

        ncxreg += 1
        nmxreg = ["intercept"; nmxreg]
    end
    # adjust method
    if method == "CSS-ML"
        anyna = any(isnan, x)
        if ncxreg > 0
            anyna |= any(isnan, xreg)
        end
        if anyna
            method = "ML"
        end
    end

    if method in ["CSS", "CSS-ML"]
        ncond = order.d + seasonal.d * m
        ncond1 = order.p + seasonal.p * m

        if isnothing(n_cond)
            ncond += ncond1
        else
            ncond += max(n_cond, ncond1)
        end
    else
        ncond = 0
    end

    # Handle fixed
    if isnothing(fixed)
        fixed = fill(NaN, narma + ncxreg)
    elseif length(fixed) != narma + ncxreg
        throw(ArgumentError("Wrong length for 'fixed'"))
    end
    mask = isnan.(fixed)
    no_optim = !any(mask)

    if no_optim
        transform_pars = false
    end

    if transform_pars
        ind = arma[1] + arma[2] .+ (1:arma[3])

        if any(.!mask[1:arma[1]]) || any(.!mask[ind])
            @warn "Some AR parameters were fixed: Setting transform_pars = false"
            transform_pars = false
        end
    end

    init0 = zeros(narma)
    parscale = ones(narma)

    # estimate init and scale
    if ncxreg > 0
        orig_xreg = (ncxreg == 1) || any(!mask[narma+1:narma+ncxreg])

        if !orig_xreg
            rows_good = all.(isfinite, eachrow(xreg))
            S = svd(xreg[rows_good, :])
            xreg = xreg * S.V
        end

        dx = copy(x)
        dxreg = copy(xreg)

        if order.d > 0
            dx = diff(dx, dims=1, n=order.d)
            dxreg = diff(dxreg, dims=1, n=order.d)
        end

        if m > 1 && seasonal.d > 0
            dx = diff(dx, m, seasonal.d)
            dxreg = diff(dxreg, m, seasonal.d)
        end

        # Filter out rows with any NaNs in x or xreg
        valid_x_rows = .!(isnan.(x)) .& .!([any(isnan, row) for row in eachrow(xreg)])
        x_clean = x[valid_x_rows]
        xreg_clean = xreg[valid_x_rows, :]

        # Filter out rows with any NaNs in dx or dxreg
        valid_dx_rows = .!(isnan.(dx)) .& .!([any(isnan, row) for row in eachrow(dxreg)])
        dx_clean = dx[valid_dx_rows]
        dxreg_clean = dxreg[valid_dx_rows, :]

        nobs = length(x_clean)  # Already cleaned, so no need for na_omit

        β = nothing
        ses = nothing
        fit_success = false

        # Attempt regression: dx ~ dxreg - 1
        if size(dx_clean, 1) > size(dxreg_clean, 2)
            try
                β = dxreg_clean \ dx_clean
                residuals = dx_clean - dxreg_clean * β
                σ2 = sum(residuals .^ 2) / (length(dx_clean) - size(dxreg_clean, 2))

                XtX = dxreg_clean' * dxreg_clean
                cov_β = σ2 * (cholesky(Symmetric(XtX)) \ I)
                ses = sqrt.(diag(cov_β))

                fit_success = true
            catch e
                @warn "dxreg fit failed: $e"
                fit_success = false
            end
        end

        # Fallback regression: x ~ xreg - 1
        if !fit_success
            try
                β = xreg_clean \ x_clean
                residuals = x_clean - xreg_clean * β
                σ2 = sum(residuals .^ 2) / (length(x_clean) - size(xreg_clean, 2))

                XtX = xreg_clean' * xreg_clean
                cov_β = σ2 * (cholesky(Symmetric(XtX)) \ I)
                ses = sqrt.(diag(cov_β))
            catch e
                error("Least squares fitting failed in fallback: $e")
            end
        end

        # Final updates
        init0 = vcat(init0, β)
        parscale = vcat(parscale, 10 .* ses)

        n_used = nobs - length(Delta)
    end

    if n_used <= 0
        error("Too few non-missing observations")
    end

    if !isnothing(init)
        if length(init) != length(init0)
            error("'init' is of the wrong length")
        end

        ind = map(x -> isnan(x) || ismissing(x), init)
        init[ind] .= init0[ind]

        if method == "ML"
            p, d, q = arma[1:3]
            if p > 0 && !arCheck(init[1:p])
                error("non-stationary AR part")
            end
            if q > 0 && !arCheck(init[p+d+1:p+d+q])
                error("non-stationary seasonal AR part")
            end
            if transform_pars
                init = arima_invtrans(init, arma)
            end
        end
    else
        init = init0
    end

    coef = copy(Float64.(fixed))

    if method == "CSS"
        if no_optim
            res = (converged=true, minimizer=zeros(0), minimum=armaCSS(zeros(0)))
        else
            opt = nmmin(p -> armaCSS(p), init[mask])

            res = (
                converged=isapprox(opt.fail, 0, atol=1e-7),
                minimizer=opt.x_opt,
                minimum=opt.f_opt,
            )
        end

        if !res.converged
            @warn "CSS optimization convergence issue."
        end

        coef[mask] .= res.minimizer

        trarma = arima_transpar(coef, arma, false)
        mod = make_arima(trarma[1], trarma[2], Delta, kappa=kappa, SSinit=SSinit)

        if ncxreg > 0
            x = x - xreg * coef[narma.+(1:ncxreg)]
        end
        
        #keep for now
        arimaSS(x, mod)
        
        val = arima_css(x, arma, trarma[1], trarma[2], ncond)
        sigma2 = val[1]
        

        if no_optim
            var = zeros(0)
        else
            hessian = optim_hessian(res.minimizer, p -> armaCSS(p))
            var = inv(hessian * n_used)
        end

    else
        if method == "CSS-ML"
            if no_optim
                res = (
                    converged=true,
                    minimizer=zeros(sum(mask)),
                    minimum=armaCSS(zeros(0)),
                )
            else
                opt = nmmin(p -> armaCSS(p), init[mask])
                res = (
                    converged=isapprox(opt.fail, 0, atol=1e-7),
                    minimizer=opt.x_opt,
                    minimum=opt.f_opt,
                )
            end

            if res.converged
                init[mask] .= res.minimizer
            end

            if arma[1] > 0 && !arCheck(init[1:arma[1]])
                error("Non-stationary AR part from CSS")
            end

            if arma[3] > 0 && !arCheck(init[sum(arma[1:2])+1:sum(arma[1:3])])
                error("Non-stationary seasonal AR part from CSS")
            end

            n_cond = 0
        end

        if transform_pars
            init = arima_invtrans(init, arma)

            if arma[2] > 0
                ind = (arma[1]+1):(arma[1]+arma[2])
                init[ind] .= maInvert(init[ind])
            end

            if arma[4] > 0
                ind = sum(arma[1:3])+1:sum(arma[4])
                init[ind] .= maInvert(init[ind])
            end
        end

        trarma = arima_transpar(init, arma, transform_pars)
        mod = make_arima(trarma[1], trarma[2], Delta, kappa=kappa, SSinit=SSinit)

        if no_optim

            res = (
                converged=true,
                minimizer=zeros(0),
                minimum=armafn(zeros(0), transform_pars),
            )
        else
            opt = nmmin(p -> armafn(p, transform_pars), init[mask])
            
            res = (
                converged=isapprox(opt.fail, 0, atol=1e-7),
                minimizer=opt.x_opt,
                minimum=opt.f_opt,
            )
        end


        if !res.converged
            @warn "CSS-ML optimization convergence issue."
        end

        coef[mask] .= res.minimizer

        if transform_pars
            if arma[2] > 0
                ind = (arma[1]+1):(arma[1]+arma[2])
                if all(mask[ind])
                    coef[ind] .= maInvert(coef[ind])
                end
            end

            if arma[4] > 0
                ind = sum(arma[1:3])+1:arma[4]
                if all(mask[ind])
                    coef[ind] .= maInvert(coef[ind])
                end
            end

            if any(coef[mask] .!= res.minimizer)
                old_convergence = res.converged
                opt = nmmin(p -> armafn(p, true), coef[mask], maxit=0)
                hessian = optim_hessian(p -> armafn(p, true), opt.x_opt)

                res = (
                    converged=old_convergence,
                    minimizer=opt.x_opt,
                    minimum=opt.f_opt,
                    hessian=hessian,
                )
                coef[mask] .= res.minimizer
            else
                hessian = optim_hessian(p -> armafn_opt(p, true), res.minimizer)
            end

            A = arima_gradtrans(coef, arma)
            A = A[mask, mask]
            var = A' * ((hessian * n_used) \ A)
            coef = arima_undopars(coef, arma)
        else
            if no_optim
                var = zeros(0)
            else
                hessian = optim_hessian(p -> armafn_opt(p, true), res.minimizer)
                var = inv(hessian * n_used)
            end
        end

        trarma = arima_transpar(coef, arma, false)
        mod = make_arima(trarma[1], trarma[2], Delta, kappa=kappa, SSinit=SSinit)

        val = if ncxreg > 0
            arimaSS(x - xreg * coef[narma+(1:ncxreg)], mod)
        else
            arimaSS(x, mod)
        end
        sigma2 = val[1][1] / n_used
    end

    # # Final steps
    value = 2 * n_used * res.minimum + n_used + n_used * log(2 * π)
    aic = method != "CSS" ? value + 2 * sum(mask) + 2 : NaN
    loglik = -0.5 * value

    idx = 1
    ar = arma[1] > 0 ? coef[idx:idx+arma[1]-1] : Float64[]
    idx += arma[1]

    ma = arma[2] > 0 ? coef[idx:idx+arma[2]-1] : Float64[]
    idx += arma[2]

    sar = arma[3] > 0 ? coef[idx:idx+arma[3]-1] : Float64[]
    idx += arma[3]

    sma = arma[4] > 0 ? coef[idx:idx+arma[4]-1] : Float64[]
    idx += arma[4]

    intercept = ncxreg > 0 ? coef[idx:idx+ncxreg-1] : Float64[]

    arima_coef = ArimaCoef(ar, ma, sar, sma, intercept)

    if ncxreg > 0 && !orig_xreg
        ind = narma .+ (1:ncxreg)
        flat_coef[ind] = S_v * flat_coef[ind]
        A = Matrix{Float64}(I, narma + ncxreg, narma + ncxreg)
        A[ind, ind] = S_v
        A = A[mask, mask]
        var = A * var * transpose(A)
    end

    resid = val[2]

    result = ArimaFit(
        arima_coef,
        sum(resid .^ 2) / n_used,
        var,
        mask,
        loglik,
        aic,
        arma,
        resid,
        res.converged,
        ncond,
        n_used,
        mod,
        xreg_original,
    )
    return result

end

function kalman_forecast(
    n::Int,
    Z::Vector{Float64},
    a::Vector{Float64},
    P::Matrix{Float64},
    T::Matrix{Float64},
    V::Matrix{Float64},
    h::Float64,
)
    p = length(a)
    a = copy(a)
    P = copy(P)
    forecasts = zeros(n)
    se = zeros(n)
    for l = 1:n
        anew = T * a
        a .= anew
        forecasts[l] = dot(Z, a)

        mm = T * P
        Pn = V + mm * transpose(T)
        P .= Pn

        se[l] = h + dot(Z, P * Z)
    end
    return forecasts, se
end

function predict_arima(
    model::ArimaFit,
    n_ahead::Int;
    newxreg = nothing,
    se_fit::Bool = true,
)

    myncol(x) = x === nothing ? 0 : size(x, 2)

    coefs_struct = model.coef
    coef_names = String[]
    coefs = Float64[]

    append!(coefs, coefs_struct.ar)
    append!(coef_names, ["ar_$i" for i = 1:length(coefs_struct.ar)])

    append!(coefs, coefs_struct.ma)
    append!(coef_names, ["ma_$i" for i = 1:length(coefs_struct.ma)])

    append!(coefs, coefs_struct.sar)
    append!(coef_names, ["sar_$i" for i = 1:length(coefs_struct.sar)])

    append!(coefs, coefs_struct.sma)
    append!(coef_names, ["sma_$i" for i = 1:length(coefs_struct.sma)])

    append!(coefs, coefs_struct.intercept)
    append!(coef_names, ["intercept" for _ in coefs_struct.intercept])

    ncxreg = count(occursin("ex_", name) for name in coef_names)

    if myncol(newxreg) != ncxreg
        throw(ArgumentError("`xreg` and `newxreg` have different numbers of columns"))
    end

    arma = model.arma
    narma = sum(arma[1:4])
    xm = zeros(n_ahead)

    if length(coefs) > narma

        if coef_names[narma+1] == "intercept"
            intercept = ones(n_ahead, 1)
            newxreg = newxreg === nothing ? intercept : hcat(intercept, newxreg)
            ncxreg += 1
        end

        xm = narma == 0 ? newxreg * coefs : newxreg * coefs[(narma+1):end]
        xm = vec(xm)
    end

    Z, a, P, T, V, h = model.model.Z,
    model.model.a,
    model.model.P,
    model.model.T,
    model.model.V,
    model.model.h
    pred, se = kalman_forecast(n_ahead, Z, a, P, T, V, h)
    pred += xm

    if se_fit
        se = sqrt.(se .* model.sigma2)
        return pred, se
    else
        return pred
    end
end
