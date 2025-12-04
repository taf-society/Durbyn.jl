"""
    ArimaStateSpace

A struct representing a univariate ARIMA state-space model, including AR and MA parameters, differencing, and state-space matrices.

# Fields
- `phi::AbstractVector`   : AR coefficients (length ≥ 0)
- `theta::AbstractVector` : MA coefficients (length ≥ 0)
- `Delta::AbstractVector` : Differencing coefficients (for seasonal/nonseasonal differences)
- `Z::AbstractVector`     : Observation coefficients
- `a::AbstractVector`     : Current state estimate
- `P::AbstractMatrix`     : Current state covariance matrix
- `T::AbstractMatrix`     : Transition/state evolution matrix
- `V::Any`                : Innovations or 'RQR', for process covariance
- `h::Real`               : Observation variance
- `Pn::AbstractMatrix`    : Prior state covariance at time t-1 (not updated by KalmanForecast)
"""
mutable struct ArimaStateSpace
    phi::AbstractVector
    theta::AbstractVector
    Delta::AbstractVector
    Z::AbstractVector
    a::AbstractVector
    P::AbstractMatrix
    T::AbstractMatrix
    V::Any
    h::Real
    Pn::AbstractMatrix
end

function show(io::IO, s::ArimaStateSpace)
    println(io, "ArimaStateSpace:")
    println(io, "  phi   (AR coefficients):         ", s.phi)
    println(io, "  theta (MA coefficients):         ", s.theta)
    println(io, "  Delta (Differencing coeffs):     ", s.Delta)
    println(io, "  Z     (Observation coeffs):      ", s.Z)
    println(io, "  a     (Current state estimate):  ", s.a)
    println(io, "  P     (Current state covariance):")
    show(io, "text/plain", s.P)
    println(io, "\n  T     (Transition matrix):")
    show(io, "text/plain", s.T)
    println(io, "\n  V     (Innovations or 'RQR'):    ", s.V)
    println(io, "  h     (Observation variance):    ", s.h)
    println(io, "  Pn    (Prior state covariance):")
    show(io, "text/plain", s.Pn)
end

"""
    ArimaFit

Holds the results of an ARIMA/SARIMA fit (including ARIMAX). This struct stores the data,
estimated parameters, likelihood and information criteria, residuals, and model metadata.

# Fields
- `y::AbstractArray`  
  The observed time-series data provided to the model.

- `fitted::AbstractArray`  
  In-sample fitted values.

- `coef::NamedMatrix`  
  Estimated coefficients (AR, MA, seasonal AR/MA, regressors). See `model.names` (or
  `model[:names]` if applicable) for parameter names and ordering.

- `sigma2::Float64`  
  Estimated innovation variance.

- `var_coef::Matrix{Float64}`  
  Variance-covariance matrix of the estimated coefficients.

- `mask::Vector{Bool}`  
  Indicates which parameters were estimated (vs. fixed/excluded).

- `loglik::Float64`  
  Maximized log-likelihood.

- `aic::Union{Float64,Nothing}`  
- `bic::Union{Float64,Nothing}`  
- `aicc::Union{Float64,Nothing}`  
  Information criteria computed from the fitted model (may be `nothing` if not applicable).

- `ic::Union{Float64,Nothing}`  
  The value of the information criterion selected for model comparison.

- `arma::Vector{Int}`  
  Compact model specification: `[p, q, P, Q, s, d, D]`.

- `residuals::Vector{Float64}`  
  Model residuals (estimated innovations).

- `convergence_code::Bool`  
  `true` if the optimizer reported successful convergence, `false` otherwise.

- `n_cond::Int`  
  Number of initial observations excluded due to conditioning (e.g., differencing).

- `nobs::Int`  
  Number of observations used in estimation (after differencing/trimming).

- `model::ArimaStateSpace`  
  State-space representation and model metadata.

- `xreg::Any`  
  Exogenous regressors matrix used in fitting (if any).

- `method::String`  
  Estimation method (e.g., `"ML"`, `"CSS"`).

- `lambda::Union{Real,Nothing}`  
  Box-Cox transformation parameter used (if any).

- `biasadj::Bool`  
  Whether bias adjustment was applied when back-transforming from Box-Cox scale.

- `offset::Float64`  
  Constant offset applied internally (e.g., for transformed models).

# Usage
```julia
fit = auto_arima(y, 12)
fit.ic               # selected IC value
fit.arma             # [p, q, P, Q, s, d, D]
fit.coef             # parameter table with names
fit.residuals        # innovations
fit.model            # state-space representation
````

"""
mutable struct ArimaFit
    y::AbstractArray
    fitted::AbstractArray
    coef::NamedMatrix
    sigma2::Float64
    var_coef::Matrix{Float64}
    mask::Vector{Bool}
    loglik::Float64
    aic::Union{Float64,Nothing}
    bic::Union{Float64,Nothing}
    aicc::Union{Float64,Nothing}
    ic::Union{Float64,Nothing}
    arma::Vector{Int}
    residuals::Vector{Float64}
    convergence_code::Bool
    n_cond::Int
    nobs::Int
    model::ArimaStateSpace
    xreg::Any
    method::String
    lambda::Union{Real, Nothing}
    biasadj::Union{Bool, Nothing}
    offset::Union{Float64, Nothing}
end


function Base.show(io::IO, fit::ArimaFit)
    println(io, "ARIMA Fit Summary")
    println(io, "-----------------")

    println(io, "Coefficients:")
    show(io, fit.coef)  # use NamedMatrix's show for aligned table

    println(io, "\nSigma²: ", fit.sigma2)
    println(io, "Log-likelihood: ", fit.loglik)
    if !isnan(fit.aic)
        println(io, "AIC: ", fit.aic)
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

# helper for compute_arima_likelihood
function state_prediction!(anew::AbstractArray, a::AbstractArray, p::Int, r::Int, d::Int, rd::Int, phi::AbstractArray, delta::AbstractArray)
     @inbounds for i in 1:r
        tmp = (i < r) ? a[i + 1] : 0.0
        if i <= p
            tmp += phi[i] * a[1]
        end
        anew[i] = tmp
    end

    if d > 0
        @inbounds for i in (r + 2):(rd)
            anew[i] = a[i - 1]
        end
        tmp = a[1]
        @inbounds for i in 1:d
            tmp += delta[i] * a[r + i]
        end
        anew[r + 1] = tmp
    end
end
# helper for compute_arima_likelihood
function predict_covariance_nodiff!(Pnew::Matrix{Float64}, P::Matrix{Float64},
    r::Int, p::Int, q::Int,
    phi::Vector{Float64}, theta::Vector{Float64})
    @inbounds for i in 1:r

        if i == 1
            vi = 1.0
        elseif i - 1 <= q
            vi = theta[i-1]
        else
            vi = 0.0
        end

        for j in 1:r

            if j == 1
                tmp = vi
            elseif j - 1 <= q
                tmp = vi * theta[j-1]
            else
                tmp = 0.0
            end

            if i <= p && j <= p
                tmp = tmp + phi[i] * phi[j] * P[1, 1]
            end

            if i <= r - 1 && j <= r - 1
                tmp = tmp + P[i+1, j+1]
            end

            if i <= p && j < r - 1
                tmp = tmp + phi[i] * P[j+1, 1]
            end

            if j <= p && i < r - 1
                tmp = tmp + phi[j] * P[i+1, 1]
            end

            Pnew[i, j] = tmp
        end
    end
end
# helper for compute_arima_likelihood
function predict_covariance_with_diff!(Pnew::Matrix{Float64}, P::Matrix{Float64},
    r::Int, d::Int, p::Int, q::Int, rd::Int,
    phi::Vector{Float64}, delta::Vector{Float64},
    theta::Vector{Float64}, mm::Matrix{Float64})
    # Step 1: mm = T * P
    @inbounds for i in 1:r
        for j in 1:rd
            tmp = 0.0
            if i <= p
                tmp = tmp + phi[i] * P[1, j]
            end
            if i < r
                tmp = tmp + P[i+1, j]
            end
            mm[i, j] = tmp
        end
    end

    @inbounds for j in 1:rd
        tmp = P[1, j]
        for k in 1:d
            tmp = tmp + delta[k] * P[r+k, j]
        end
        mm[r+1, j] = tmp
    end

    @inbounds for i in 2:d
        for j in 1:rd
            mm[r+i, j] = P[r+i-1, j]
        end
    end

    # Step 2: Pnew = mm * Tᵀ
    @inbounds for i in 1:r
        for j in 1:rd
            tmp = 0.0
            if i <= p
                tmp = tmp + phi[i] * mm[1, j]
            end
            if i < r
                tmp = tmp + mm[i+1, j]
            end
            Pnew[i, j] = tmp
        end
    end

    @inbounds for j in 1:rd
        tmp = mm[1, j]
        for k in 1:d
            tmp = tmp + delta[k] * mm[r+k, j]
        end
        Pnew[r+1, j] = tmp
    end

    @inbounds for i in 2:d
        for j in 1:rd
            Pnew[r+i, j] = mm[r+i-1, j]
        end
    end

    # Step 3: Add noise (MA(q))
    @inbounds for i in 1:(q+1)
        if i == 1
            vi = 1.0
        else
            vi = theta[i-1]
        end

        for j in 1:(q+1)
            if j == 1
                vj = 1.0
            else
                vj = theta[j-1]
            end
            Pnew[i, j] = Pnew[i, j] + vi * vj
        end
    end
end
# helper for compute_arima_likelihood
# This is a bit confusing: C code uses row major operations. Pnew[i + r * j]
function kalman_update!(y_obs, anew, delta, Pnew, M, d, r, rd, a, P, useResid, rsResid, l, ssq, sumlog, nu,)

    # 1) residual
    resid = y_obs - anew[1]
    @inbounds for i in 1:d
        resid = resid - delta[i] * anew[r+i]
    end

    # 2) build M = Pnew * [1; delta]
    @inbounds for i in 1:rd
        tmp = Pnew[i, 1]
        for j in 1:d
            tmp += Pnew[i, r+j] * delta[j]
        end
        M[i] = tmp
    end

    # 3) compute gain = H* M
    gain = M[1]
    @inbounds for j in 1:d
        gain += delta[j] * M[r+j]
    end

    # 4) update ssq, sumlog, nu if gain is “safe”
    if gain < 1e4
        nu[] += 1
        ssq[] += resid^2 / gain
        sumlog[] += log(gain)
    end

    # 5) optionally store standardized residual
    if useResid
        rsResid[l] = resid / sqrt(gain)
    end

    # 6) state update: a = anew + (M * resid)/gain
    @inbounds for i in 1:rd
        a[i] = anew[i] + M[i] * resid / gain
    end

    # 7) covariance update: P = Pnew - (M Mᵀ)/gain
    @inbounds for i = 1:rd
        for j = 1:rd
            P[i, j] = Pnew[i, j] - (M[i] * M[j]) / gain
        end
    end
end

"""
    compute_arima_likelihood(y::Vector{Float64},
                             model::ArimaStateSpace,
                             update_start::Int,
                             give_resid::Bool)

Compute the Gaussian log-likelihood and related quantities for a univariate ARIMA model using the Kalman filter.
 
It runs a Kalman filter on the observed time series `y`, using the state-space representation stored in `model`.  
It accumulates the innovation sum of squares and the log-determinant contributions required for the Gaussian likelihood.  
If `give_resid` is true, the function also computes and returns the standardized residuals.

# Arguments
- `y::Vector{Float64}`: Observed time series (univariate).
- `model::ArimaStateSpace`: State-space model, as returned by `initialize_arima_state`.
- `update_start::Int`: The time index at which to begin updating the likelihood and residuals (typically 1).
- `give_resid::Bool`: If true, also compute and return standardized residuals.

# Returns
A `Dict` with keys:
- `"ssq"`: Sum of squared innovations.
- `"sumlog"`: Accumulated log-determinants of the prediction error variances.
- `"nu"`: Innovations (prediction errors).
- `"resid"`: (only if `give_resid` is true) Standardized residuals.

# Notes
- The arguments and behavior closely follow the C implementation in R's base ARIMA code.
- For details on the state-space representation, see [`initialize_arima_state`](@ref).

# References
- Durbin, J. & Koopman, S. J. (2001). *Time Series Analysis by State Space Methods*. Oxford University Press.
- Gardner, G., Harvey, A. C. & Phillips, G. D. A. (1980). Algorithm AS 154. *Applied Statistics*, 29, 311-322.

"""
# Tested and it is safe. Possible improvement potatial.
function compute_arima_likelihood( y::Vector{Float64}, model::ArimaStateSpace, update_start::Int, give_resid::Bool,)

    phi = model.phi
    theta = model.theta
    delta = model.Delta
    a = model.a
    P = copy(model.P)
    Pnew = copy(model.Pn)

    n = length(y)
    rd = length(a)
    p = length(phi)
    q = length(theta)
    d = length(delta)
    r = rd - d

    ssq = Ref(0.0)
    sumlog = Ref(0.0)
    nu = Ref(0)

    anew = zeros(rd)
    M = zeros(rd)
    if d > 0
        mm = zeros(rd, rd)
    else
        mm = nothing
    end

    if give_resid
        rsResid = zeros(n)
    else
        rsResid = nothing
    end
    @inbounds for l = 1:n
        state_prediction!(anew, a, p, r, d, rd, phi, delta)

        if !isnan(y[l])
            kalman_update!(y[l], anew, delta, Pnew, M, d, r, rd, a, P, give_resid, rsResid, l, ssq, sumlog, nu)
        else
            a .= anew
            if give_resid
                rsResid[l] = NaN
            end
        end

        if l > update_start
            if d == 0
                predict_covariance_nodiff!(Pnew, P, r, p, q, phi, theta)
            else
                predict_covariance_with_diff!(Pnew, P, r, d, p, q, rd, phi, delta, theta, mm)
            end
            P .= Pnew
        end
    end

    result_stats = [ssq[], sumlog[], nu[]]

    if give_resid
        return (stats = result_stats, residuals = rsResid)
    else
        return result_stats
    end
end


"""
    transform_unconstrained_to_ar_params!(p, raw, dest)

Convert a vector of unconstrained real numbers to autoregressive (AR) coefficients.

This routine implements the two-step transformation used in the R reference
implementation: it first maps the unconstrained `raw` values into the open
interval `(-1, 1)` via the hyperbolic tangent to obtain partial autocorrelation
coefficients (PACF), and then applies the Durbin-Levinson recursion to obtain
the corresponding AR parameters.  The result is written in-place to `dest`.

Arguments
---------
- `p::Int`: the number of parameters to transform.  Must satisfy `p ≤ 100`.
- `raw::AbstractVector`: a vector of length at least `p` containing the
  unconstrained parameters.
- `new::AbstractVector`: a preallocated vector of length at least `p` into
  which the AR coefficients will be written.  On entry its contents are
  ignored; on exit it contains the transformed values.

Throws
------
`ArgumentError` if `p > 100`.

Notes
-----
This function mutates its `new` argument.  A working copy of the first
`p` elements is used internally to avoid aliasing.
"""
# Tested and it is safe. Possible improvement potatial.
function transform_unconstrained_to_ar_params!(
    p::Int,
    raw::AbstractVector,
    new::AbstractVector,
)
    if p > 100
        throw(ArgumentError("The function can only transform 100 parameters in arima0"))
    end

    @inbounds new[1:p] .= tanh.(raw[1:p])
    work = copy(new[1:p])

    @inbounds for j = 2:p
        a = new[j]
        for k = 1:(j-1)
            work[k] -= a * new[j-k]
        end
        new[1:(j-1)] .= work[1:(j-1)]
    end

end


"""
    compute_arima_transform_gradient(x, arma)

Compute the Jacobian matrix of the ARIMA parameter transformation.

This function numerically approximates the gradient (Jacobian) of the
transformation that maps unconstrained parameter vectors to the ARIMA
parameter space.  It mirrors the behaviour of the R/C implementation
`ARIMA_Gradtrans` by perturbing each parameter individually by a small
epsilon (`1e-3`) and computing the resulting change in the transformed
coefficients.  The result is returned as a square matrix where each
row corresponds to the gradient with respect to one parameter.

Arguments
---------
- `x::AbstractArray`: the vector of input parameters (potentially
  partially constrained).  Its length determines the dimension of the
  Jacobian.
- `arma::AbstractArray`: a vector encoding the ARIMA order.  The first
  three entries correspond to the number of non-seasonal AR terms (`p`),
  the number of non-seasonal MA terms (`q`), and the number of
  seasonal AR terms (`P`), respectively.  Only these three values are
  used by this function.

Returns
-------
A dense `n*n` matrix of `Float64` where `n = length(x)`.  Elements
outside the blocks corresponding to AR parameters are zero.
"""
# The function is tested works as expected
function compute_arima_transform_gradient(x::AbstractArray, arma::AbstractArray)
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

        transform_unconstrained_to_ar_params!(mp, w1, w2)

        for i = 1:mp
            w1[i] += eps
            transform_unconstrained_to_ar_params!(mp, w1, w3)
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
        transform_unconstrained_to_ar_params!(msp, w1, w2)
        for i = 1:msp
            w1[i] += eps
            transform_unconstrained_to_ar_params!(msp, w1, w3)
            for j = 1:msp
                y[i+v, j+v] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end
    return y
end

"""
    undo_arima_parameter_transform(x, arma)

Undo the ARIMA parameter transformations applied to the AR coefficients.

Given a vector of transformed parameters `x` and the ARIMA specification `arma`,
this function applies the inverse of the parameter transformation used in
`transform_unconstrained_to_ar_params!` to restore the original (unconstrained)
parameters.  It mirrors the behaviour of the C function `ARIMA_undoPars`.
The result is returned as a copy of `x` with the AR terms replaced by
their inverse-transformed values.

Arguments
---------
- `x::AbstractArray`: a vector containing the transformed parameters.
- `arma::AbstractArray`: a vector encoding the ARIMA order.  Only the
  first three elements (`p`, `q`, `P`) are used here.

Returns
-------
A new vector of the same length as `x` containing the untransformed
parameters.
"""
# The function is tested works as expected
function undo_arima_parameter_transform(x::AbstractArray, arma::AbstractArray)
    mp, mq, msp = arma[1:3]
    res = copy(x)
    if mp > 0
        transform_unconstrained_to_ar_params!(mp, x, res)
    end
    v = mp + mq
    if msp > 0
        transform_unconstrained_to_ar_params!(msp, @view(x[v+1:end]), @view(res[v+1:end]))
    end
    return res
end

"""
    time_series_convolution(a, b)

Perform a discrete convolution between two numeric sequences.

This function computes the convolution of vectors `a` and `b`, returning
an array whose length is `length(a) + length(b) - 1`.  It corresponds to
the helper `TSconv` in the R/C source and is used to construct
difference operators for the ARIMA model.

Arguments
---------
- `a::AbstractArray`: the first sequence.
- `b::AbstractArray`: the second sequence.

Returns
-------
A vector containing the convolution of `a` and `b`.
"""
# The function is tested works as expected
function time_series_convolution(a::AbstractArray, b::AbstractArray)
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

"""
    update_least_squares!(n_parameters, xnext, xrow, ynext, d, rbar, thetab)

Internal helper used by `compute_q0_covariance_matrix` to update the
least-squares regression quantities when processing autocovariances.  This
function closely follows the Fortran routine used in the R implementation.
It updates the arrays `d`, `rbar` and `thetab` in place, based on the
incoming observation `xnext` and response `ynext`.

Arguments
---------
- `n_parameters::Int`: the number of parameters in the regression.
- `xnext::AbstractArray`: the new predictor values.
- `xrow::AbstractArray`: a working array to hold modified predictor values.
- `ynext::Float64`: the new response value.
- `d::AbstractArray`: diagonal of the regression matrix to be updated.
- `rbar::AbstractArray`: upper triangular portion of the regression matrix.
- `thetab::AbstractArray`: regression coefficients to be updated.

This function mutates `d`, `rbar` and `thetab` and returns nothing.
"""
# The function is tested works as expected
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

"""
    inverse_ar_parameter_transform(ϕ)

Compute the inverse transformation from AR coefficients to unconstrained
parameters.

This function reverses the Durbin-Levinson transformation applied by
`transform_unconstrained_to_ar_params!`.  Given a vector of AR
coefficients `ϕ`, it returns the corresponding unconstrained parameters
on the real line by running the recursion backwards and applying the
inverse hyperbolic tangent.

Arguments
---------
- `ϕ::AbstractVector`: vector of AR coefficients.

Returns
-------
A vector of the same length as `ϕ` containing the unconstrained
parameters.
"""
# The function is tested works as expected
function inverse_ar_parameter_transform(ϕ::AbstractVector)
    p = length(ϕ)
    new = Array{Float64}(undef, p)
    copy!(new, ϕ)
    work = similar(new)
    # Perform the backward Durbin-Levinson recursion.  This recovers the
    # partial autocorrelations from the AR coefficients.
    # This is confusing be carriful.
    for j in p:-1:2
        a = new[j]
        denom = 1 - a^2
        @assert denom ≠ 0 "Encountered unit root at j=$j (a=±1)."
        for k in 1:j-1
            work[k] = (new[k] + a * new[j-k]) / denom
        end
        new[1:j-1] = work[1:j-1]
    end
    return map(x -> abs(x) <= 1 ? atanh(x) : NaN, new)
end

"""
    inverse_arima_parameter_transform(θ, arma)

Apply the inverse ARIMA parameter transformation to a parameter vector.

Given a parameter vector `θ` and the ARIMA specification `arma`, this
function applies the inverse transformation used in the ARIMA fitting
process to recover the unconstrained parameters.  It reverses the
seasonal and non-seasonal AR transformations by calling
`inverse_ar_parameter_transform` on the appropriate slices.

Arguments
---------
- `θ::AbstractVector`: vector of transformed parameters.
- `arma::AbstractVector{Int}`: vector encoding the ARIMA order.  The
  first three elements correspond to the non-seasonal AR (`p`), MA (`q`)
  and seasonal AR (`P`) orders.

Returns
-------
A new vector containing the unconstrained parameters.
"""
# The function is tested works as expected
function inverse_arima_parameter_transform(θ::AbstractVector, arma::AbstractVector{Int})
    mp, mq, msp = arma
    n = length(θ)
    v = mp + mq
    @assert v + msp ≤ n "Sum mp+mq+msp exceeds length(θ)"
    raw = Array{Float64}(undef, n)
    copy!(raw, θ)
    transformed = raw

    # non‐seasonal AR
    if mp > 0
        transformed[1:mp] = inverse_ar_parameter_transform(raw[1:mp])
    end

    # seasonal AR
    if msp > 0
        transformed[v+1:v+msp] = inverse_ar_parameter_transform(raw[v+1:v+msp])
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

            res[ind + 1] = V[ind+1]

            if j != 0
                indn -= 1
                res[ind+1] += res[indn+1]
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



"""
    compute_q0_covariance_matrix(phi, theta)

Compute the initial state covariance matrix for the AR component of an ARIMA model.

This function implements the algorithm described in the R `getQ0` function.  It
takes the AR (`phi`) and MA (`theta`) coefficient vectors and returns the
covariance matrix `Q₀` used to initialize the state-space representation of the
ARIMA model.  Internally it constructs the vector of autocovariances and
invokes a series of helper functions to fill in the appropriate blocks of
the matrix.

Arguments
---------
- `phi::AbstractArray`: vector of non-seasonal AR coefficients.
- `theta::AbstractArray`: vector of non-seasonal MA coefficients.

Returns
-------
A symmetric matrix of size `r*r`, where `r = max(length(phi), length(theta) + 1)`.
"""
# The function is tested works as expected
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

"""
    getQ0bis(phi, theta, tol)

Compute the initial covariance matrix for the AR component of an ARIMA model
using the Rossignol (2011) method.

The original C code in R exposes two methods for computing the initial
covariance matrix used by the Kalman filter: the Gardner (1980) approach
(`getQ0`) and the Rossignol (2011) approach (`getQ0bis`).  The latter is
more computationally intensive and relies on numerically solving a set of
Yule-Walker equations.  For the purposes of this translation we provide
a simplified implementation: this function simply delegates to
`compute_q0_covariance_matrix`, which corresponds to the Gardner method.
If a more faithful implementation is required, this function can be
replaced by an appropriate algorithm.  The parameter `tol` is currently
ignored.

Arguments
---------
- `phi::AbstractArray`: vector of non-seasonal AR coefficients.
- `theta::AbstractArray`: vector of non-seasonal MA coefficients.
- `tol::Real`: tolerance parameter (unused).

Returns
-------
A symmetric matrix of size `r*r`, where `r = max(length(phi), length(theta) + 1)`.
"""
function getQ0bis(phi::AbstractArray, theta::AbstractArray, tol::Real)
    return compute_q0_covariance_matrix(phi, theta)
end

"""
    transform_arima_parameters(params_in, arma, trans)

Transform a flat parameter vector into AR and MA coefficient vectors.

This function expands and optionally transforms the parameters of an ARIMA
model.  Given a vector of raw parameters `params_in` and the ARIMA order
`arma`, it produces the non-seasonal and seasonal AR (`phi`) and MA
(`theta`) coefficient vectors.  If `trans` is `true` the unconstrained
parameters are first passed through `transform_unconstrained_to_ar_params!`
for stability.

Arguments
---------
- `params_in::AbstractArray`: input parameter vector.
- `arma::Vector{Int}`: model specification `[p, q, P, Q, s, d, D]`.
- `trans::Bool`: whether to apply the parameter transformation before
  expansion.

Returns
-------
A tuple `(phi, theta)` containing the expanded AR and MA coefficient vectors.
"""
# The function is tested works as expected
function transform_arima_parameters(
    params_in::AbstractArray,
    arma::Vector{Int},
    trans::Bool,
)
mp, mq, msp, msq, ns = arma
    p = mp + ns * msp
    q = mq + ns * msq

    phi = zeros(Float64, p)
    theta = zeros(Float64, q)
    params = copy(params_in)

    if trans
        if mp > 0
            transform_unconstrained_to_ar_params!(mp, params_in, params)
        end
        v = mp + mq
        if msp > 0
            transform_unconstrained_to_ar_params!(msp, params_in[v+1:end], params[v+1:end])
        end
    end

    if ns > 0
        @inbounds phi[1:mp] .= params[1:mp]
        @inbounds theta[1:mq] .= params[mp+1:mp+mq]

        @inbounds for j = 0:(msp-1)
            phi[(j+1)*ns] += params[mp+mq+j+1]
            for i = 0:(mp-1)
                phi[((j+1)*ns)+(i+1)] -= params[i+1] * params[mp+mq+j+1]
            end
        end

        @inbounds for j = 0:(msq-1)
            theta[(j+1)*ns] += params[mp+mq+msp+j+1]
            for i = 0:(mq-1)
                theta[((j+1)*ns)+(i+1)] += params[mp+i+1] * params[mp+mq+msp+j+1]
            end
        end
    else
        @inbounds phi[1:mp] .= params[1:mp]
        @inbounds theta[1:mq] .= params[mp+1:mp+mq]
    end

    return (phi, theta)
end

"""
    compute_css_residuals(y, arma, phi, theta, ncond)

Compute the conditional sum of squares (CSS) and residuals for an ARIMA model.

This routine mirrors the behaviour of the R function `ARIMA_CSS`.  It first
applies the appropriate differencing specified by the ARIMA order and then
computes the residuals of the ARMA model defined by `phi` and `theta`.  The
sum of squared residuals and the number of non-missing residuals are used
to compute the innovation variance estimate.  When called from the
high-level `arima` function, these residuals provide a fast approximate
estimate of the parameters prior to full maximum likelihood estimation.

Arguments
---------
- `y::AbstractArray`: the observed series (may contain missing values).
- `arma::Vector{Int}`: model specification `[p, q, P, Q, s, d, D]`.
- `phi::AbstractArray`: vector of non-seasonal AR coefficients.
- `theta::AbstractArray`: vector of non-seasonal MA coefficients.
- `ncond::Int`: number of initial observations to condition on.

Returns
-------
A `Dict` with keys `"sigma2"` giving the variance estimate and
`"resid"` giving the vector of residuals.
"""
# The function is tested works as expected
function compute_css_residuals(
    y::AbstractArray,
    arma::Vector{Int},
    phi::AbstractArray,
    theta::AbstractArray,
    ncond::Int,
)
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

    return (sigma2 = ssq / nu, residuals = resid)
end

"""
    initialize_arima_state(phi, theta, Delta; kappa=1e6, SSinit="Gardner1980", tol=eps(Float64))

Create and initialize the state-space representation of an ARIMA model.

Given vectors of AR coefficients `phi`, MA coefficients `theta`, and the differencing polynomial `Delta`, 
this function constructs all state-space matrices required for Kalman filtering and smoothing.  
This function mirrors the structure and logic of the corresponding C function used in R, and is used internally 
by high-level ARIMA fitting routines.

The initial state covariance matrix `Pn` is computed either by `compute_q0_covariance_matrix` (for `SSinit="Gardner1980"`)
or by `getQ0bis` (for `SSinit="Rossignol2011"`).

# Arguments
- `phi::Vector{Float64}`: Non-seasonal AR coefficients.
- `theta::Vector{Float64}`: Non-seasonal MA coefficients.
- `Delta::Vector{Float64}`: Differencing polynomial coefficients.
- `kappa::Float64`: Prior variance used to initialize the differenced states (default: `1e6`).
- `SSinit::String`: Method for computing the initial covariance matrix ("Gardner1980" or "Rossignol2011").
- `tol::Float64`: Tolerance parameter used by the Rossignol method.

# Returns
- An [`ArimaStateSpace`](@ref) struct containing the fields:
    - `phi`, `theta`, `Delta`, `Z`, `a`, `P`, `T`, `V`, `h`, `Pn`  
      (see [`ArimaStateSpace`](@ref) for field descriptions).

# Notes
- This function is intended for internal use, typically by higher-level ARIMA fitting routines.
- The returned struct can be used directly with Kalman filtering and smoothing algorithms.

# References
- Gardner, G., Harvey, A. C. & Phillips, G. D. A. (1980). Algorithm AS 154: An algorithm for exact maximum likelihood estimation of autoregressive-moving average models by means of Kalman filtering. *Applied Statistics*, 29, 311-322.
- Durbin, J. & Koopman, S. J. (2001). *Time Series Analysis by State Space Methods*. Oxford University Press.

"""
# The function is tested works as expected
function initialize_arima_state(phi::Vector{Float64}, theta::Vector{Float64}, Delta::Vector{Float64}; kappa::Float64=1e6, SSinit::String="Gardner1980", tol::Float64=eps(Float64))
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
            Pn[1:r, 1:r] = compute_q0_covariance_matrix(phi, theta)
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
    return ArimaStateSpace(phi, theta, Delta, Z, a, P, T, V, h, Pn)
end


"""
    process_xreg(xreg::Union{NamedMatrix, Nothing}, n::Int)

Process an exogenous regressor `xreg` (which may be `nothing` or a `NamedMatrix`).

Returns:
- `xreg::Matrix{Float64}`: the data matrix (guaranteed Float64 type)
- `ncxreg::Int`: number of columns in `xreg`
- `nmxreg::Vector{String}`: column names

# Arguments
- `xreg`: either `nothing` or a `NamedMatrix`
- `n`: number of rows expected

# Throws
- `ArgumentError` if the number of rows in `xreg` does not match `n`
"""
function process_xreg(xreg::Union{NamedMatrix,Nothing}, n::Int)
    if isnothing(xreg)
        xreg_mat = Matrix{Float64}(undef, n, 0)
        ncxreg = 0
        nmxreg = String[]
    else
        if size(xreg.data, 1) != n
            throw(ArgumentError("Lengths of x and xreg do not match!"))
        end
        xreg_mat = xreg.data
        # Ensure Float64 if needed as in R but I am not sure. I will discuss with Rob.
        # For datasets where the integer are used such as dummy variables.
        if !(eltype(xreg_mat) <: Float64)
            xreg_mat = Float64.(xreg_mat)
        end
        ncxreg = size(xreg_mat, 2)
        nmxreg = xreg.colnames
    end
    return xreg_mat, ncxreg, nmxreg
end

"""
    regress_and_update!(init0, parscale, x, xreg, mask, narma, ncxreg, order_d, seasonal_d, m, Delta)

Regression block for exogenous regressors with missing value handling and coefficient scaling.

# Arguments
- `x::Vector{Float64}`: Target variable (can contain NaN for missing values).
- `xreg::Matrix{Float64}`: Regressor matrix (can contain NaN for missing).
- `mask::Vector{Bool}`: Boolean mask for fixed/free parameters (length: narma + ncxreg).
- `narma::Int`: Number of ARMA params (used for mask indexing).
- `ncxreg::Int`: Number of exogenous regressors.
- `order_d::Int`: Order of nonseasonal differencing.
- `seasonal_d::Int`: Order of seasonal differencing.
- `m::Int`: Seasonal period.
- `Delta::Vector`: Differencing indices (used for n_used).

# Returns
Tuple: (`init0`, `parscale`, `n_used`)
"""
function regress_and_update!(
    x::AbstractArray,
    xreg::Matrix,
    mask::AbstractArray,
    narma::Int,
    ncxreg::Int,
    order_d::Int,
    seasonal_d::Int,
    m::Int,
    Delta::AbstractArray,
)

    init0 = zeros(narma)
    parscale = ones(narma)
    # Convert missings to NaN everywhere
    x, xreg = na_omit_pair(x, xreg)

    orig_xreg = (ncxreg == 1) || any(.!mask[(narma+1):(narma+ncxreg)])

    if !orig_xreg
        rows_good = [all(isfinite, row) for row in eachrow(xreg)]
        S = svd(xreg[rows_good, :])
        xreg = xreg * S.V
    else
        S = nothing
    end

    dx = copy(x)
    dxreg = copy(xreg)
    if order_d > 0
        dx = diff(dx; lag = 1, differences = order_d)
        dxreg = diff(dxreg; lag = 1, differences = order_d)
        dx, dxreg = na_omit_pair(dx, dxreg)
    end
    if m > 1 && seasonal_d > 0
        dx = diff(dx; lag = m, differences = seasonal_d)
        dxreg = diff(dxreg; lag = m, differences = seasonal_d)
        dx, dxreg = na_omit_pair(dx, dxreg)
    end

    if length(dx) > size(dxreg, 2)
        try
            fit = Stats.ols(dx, dxreg)
            fit_rank = rank(dxreg)
        catch e
            @warn "Fitting OLS to difference data failed: $e"
            fit = nothing
            fit_rank = 0
        end
    else
        @debug "Not enough observations to fit OLS" length_dx=length(dx) predictors=size(dxreg, 2)
        fit = nothing
        fit_rank = 0
    end

    if fit_rank == 0
        x, xreg = na_omit_pair(x, xreg)
        fit = Stats.ols(x, xreg)
    end

    isna = isnan.(x) .| [any(isnan, row) for row in eachrow(xreg)]
    n_used = sum(.!isna) - length(Delta)
    model_coefs = Stats.coefficients(fit)
    init0 = append!(init0, model_coefs)
    ses = fit.se
    parscale = append!(parscale, 10 * ses)

    return init0, parscale, n_used, orig_xreg, S
end

"""
    prep_coefs(
        arma::Vector{Int}, 
        coef::AbstractArray, 
        cn::Vector{String}, 
        ncxreg::Int
    ) -> NamedMatrix

Construct a `NamedMatrix` representing model coefficients, assigning appropriate names to each coefficient 
according to AR, MA, seasonal, and exogenous (xreg) components.

# Arguments

- `arma::Vector{Int}`: A vector of length 4 specifying the orders of the model in the form 
  `[AR, MA, SAR, SMA]`, where:
    - `AR`: Number of non-seasonal autoregressive coefficients.
    - `MA`: Number of non-seasonal moving average coefficients.
    - `SAR`: Number of seasonal autoregressive coefficients.
    - `SMA`: Number of seasonal moving average coefficients.

- `coef::AbstractArray`: A one-dimensional array of coefficient values, ordered as specified by the model.

- `cn::Vector{String}`: Names of exogenous regressors (if any).

- `ncxreg::Int`: Number of exogenous regressors.

# Returns

- A `NamedMatrix` object containing the coefficients as a 1-row matrix, with column names 
  corresponding to parameter names such as `"ar1"`, `"ma1"`, `"sar1"`, `"sma1"`, and any 
  exogenous regressor names.

# Example

```julia
arma = [2, 1, 0, 0]                  # AR(2), MA(1), no seasonal
coef = [0.8, -0.3, 0.4, 1.2]         # AR1, AR2, MA1, xreg1
cn = ["xreg1"]                       # exogenous name(s)
ncxreg = 1

nm = prep_coefs(arma, coef, cn, ncxreg)
```
# Output: NamedMatrix with columns ["ar1", "ar2", "ma1", "xreg1"]
"""
function prep_coefs(arma::Vector{Int}, coef::AbstractArray, cn::Vector{String}, ncxreg::Int)
    names = String[]
    if arma[1] > 0
        append!(names, ["ar$(i)" for i in 1:arma[1]])
    end
    if arma[2] > 0
        append!(names, ["ma$(i)" for i in 1:arma[2]])
    end
    if arma[3] > 0
        append!(names, ["sar$(i)" for i in 1:arma[3]])
    end
    if arma[4] > 0
        append!(names, ["sma$(i)" for i in 1:arma[4]])
    end
    if ncxreg > 0
        append!(names, cn)
    end
    mat = reshape(coef, 1, :)
    return NamedMatrix(mat, names)
end

function update_arima(mod::ArimaStateSpace, phi, theta; ss_g=true)
    p = length(phi)
    q = length(theta)
    r = max(p, q + 1)

    mod.phi = phi
    mod.theta = theta

    if p > 0
        mod.T[1:p, 1] .= phi
    end

    if r > 1
        if ss_g
            mod.Pn[1:r, 1:r] .= compute_q0_covariance_matrix(phi, theta)
        else
            mod.Pn[1:r, 1:r] .= getQ0bis(phi, theta, 0.0)
        end
    else
        if p > 0
            mod.Pn[1, 1] = 1 / (1 - phi[1]^2)
        else
            mod.Pn[1, 1] = 1.0
        end
    end

    mod.a .= 0.0
    return mod
end

# Check AR polynomial stationarity
function ar_check(ar)
    v = vcat(1.0, -ar...)
    last_nz = findlast(x -> x != 0.0, v)
    p = isnothing(last_nz) ? 0 : last_nz - 1
    if p == 0
        return true
    end

    coeffs = vcat(1.0, -ar[1:p]...)
    rts = roots(Polynomial(coeffs))

    return all(abs.(rts) .> 1.0)
end

# Invert MA polynomial
function ma_invert(ma)
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

function arima(
    x::AbstractArray,
    m::Int;
    order::PDQ = PDQ(0, 0, 0),
    seasonal::PDQ = PDQ(0, 0, 0),
    xreg::Union{Nothing, NamedMatrix} = nothing,
    include_mean::Bool = true,
    transform_pars::Bool = true,
    fixed::Union{Nothing, AbstractArray} = nothing,
    init::Union{Nothing, AbstractArray}= nothing,
    method::String = "CSS-ML",
    n_cond::Union{Nothing, AbstractArray} = nothing,
    SSinit::String = "Gardner1980",
    optim_method::String = "BFGS",
    optim_control::Dict = Dict(),
    kappa::Real = 1e6,)

    SSinit = match_arg(SSinit, ["Gardner1980", "Rossignol2011"])
    method = match_arg(method, ["CSS-ML", "ML", "CSS"])
    SS_G = SSinit == "Gardner1980"

    # State-space likelihood
    function compute_state_space_likelihood(y, model)
        # Delegate to the Kalman filter based likelihood computation.
        return compute_arima_likelihood(y, model, 0, true)
    end

    # Objective for ML optimization
    function armafn(p, trans)
        par = copy(coef)
        par[mask] = p
        trarma = transform_arima_parameters(par, arma, trans)
        xxi = copy(x)

        #Z = upARIMA(mod, trarma[1], trarma[2])
        Z = update_arima(mod, trarma[1], trarma[2]; ss_g=SS_G)

        try
            #Z = upARIMA(mod, trarma[1], trarma[2])
            Z = update_arima(mod, trarma[1], trarma[2]; ss_g=SS_G)
        catch e
            @warn "Updating arima failed $e"
            return typemax(Float64)
        end

        if ncxreg > 0
            xxi = xxi .- xreg * par[narma+1 : narma+ncxreg]
        end
        resss = compute_arima_likelihood(xxi, Z, 0, false)

        s2 = resss[1] / resss[3]
        return 0.5 * (log(s2) + resss[2] / resss[3])
    end
    # Conditional sum of squares objective
    function armaCSS(p)
        par = copy(fixed)
        par[mask] .= p
        trarma = transform_arima_parameters(par, arma, false)
        x_in = copy(x)

        if ncxreg > 0
            x_in = x_in .- xreg * par[narma+1 : narma+ncxreg]
        end

        ross = compute_css_residuals(x_in, arma, trarma[1], trarma[2], ncond)
        return 0.5 * log(ross[:sigma2])
    end
    
    n = length(x)
    y = copy(x)

    arma = [order.p, order.q, seasonal.p, seasonal.q, m, order.d, seasonal.d]

    narma = sum(arma[1:4])

    # Build Delta
    Delta = [1.0]

    for _ = 1:order.d
        Delta = time_series_convolution(Delta, [1.0, -1.0])
    end

    for _ = 1:seasonal.d
        seasonal_filter = [1.0; zeros(m - 1); -1.0]
        Delta = time_series_convolution(Delta, seasonal_filter)
    end

    Delta = -Delta[2:end]

    nd = order.d + seasonal.d
    n_used = length(na_omit(x)) - length(Delta)

    xreg_original = xreg

    if include_mean && (nd == 0)
        if isnothing(xreg)
            xreg = NamedMatrix(zeros(n, 0), String[])
        end
        xreg = add_drift_term(xreg, ones(n), "intercept")
    end

    xreg, ncxreg, nmxreg = process_xreg(xreg, n)

    if method == "CSS-ML"
        has_missing = xi -> (ismissing(xi) || isnan(xi))
        anyna = any(has_missing, x)
        if ncxreg > 0
            anyna |= any(has_missing, xreg)
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

    # estimate init and scale
    if ncxreg > 0
        init0, parscale, n_used, orig_xreg, S = 
        regress_and_update!(x, xreg, mask, narma, ncxreg, order.d, seasonal.d, m, Delta)
    else
        init0 = zeros(narma)
        parscale = ones(narma)
    end

    if n_used <= 0
        error("Too few non-missing observations")
    end

    if !isnothing(init)
        if length(init) != length(init0)
            error("'init' is of the wrong length")
        end

        ind = map(x -> isnan(x) || ismissing(x), init)
        if any(ind)
           init[ind] .= init0[ind] 
        end

        if method == "ML"
            p, d, q = arma[1:3]
            if p > 0
                if !ar_check(init[1:p])
                    error("non-stationary AR part")
                end
            end
            if q > 0
                sa_start = p + d + 1
                sa_stop = p + d + q
                if !ar_check(init[sa_start:sa_stop])
                    error("non-stationary seasonal AR part")
                end
            end
            if transform_pars
                init = inverse_arima_parameter_transform(init, arma)
            end
        end
    else
        init = copy(init0)
    end

    coef = copy(Float64.(fixed))

    if method == "CSS"
        if no_optim
            res = (converged = true, minimizer = zeros(0), minimum = armaCSS(zeros(0)))
        else
            ctrl = copy(optim_control)
            ctrl["parscale"] = parscale

            opt = optim(
                init[mask],
                p -> armaCSS(p);
                method = optim_method,
                control = ctrl,
            )
            res = (
                converged = opt.convergence == 0,
                minimizer = opt.par,
                minimum = opt.value,
            )
        end

        if !res.converged
            @warn "CSS optimization convergence issue: convergence code $(opt.convergence)"
        end

        coef[mask] .= res.minimizer

        trarma = transform_arima_parameters(coef, arma, false)
        mod = initialize_arima_state(
            trarma[1],
            trarma[2],
            Delta;
            kappa = kappa,
            SSinit = SSinit,
        )
        
        if ncxreg > 0
            x = x - xreg * coef[narma+1 : narma+ncxreg]
        end
        # Change a in mod
        compute_state_space_likelihood(x, mod)

        val = compute_css_residuals(x, arma, trarma[1], trarma[2], ncond)
        sigma2 = val[:sigma2]


        if no_optim
            var = zeros(0)
        else
            hessian = optim_hessian(p -> armaCSS(p), res.minimizer)
            var = inv(hessian * n_used)
        end

    else
        if method == "CSS-ML"
            if no_optim
                res = (
                    converged = true,
                    minimizer = zeros(sum(mask)),
                    minimum = armaCSS(zeros(0)),
                )
            else
                ctrl = copy(optim_control)
                ctrl["parscale"] = parscale

                opt = optim(
                    init[mask],
                    p -> armaCSS(p);
                    method = optim_method,
                    control = ctrl,
                )
                res = (
                    converged = opt.convergence == 0,
                    minimizer = opt.par,
                    minimum = opt.value,
                )
            end

            if res.converged
                init[mask] .= res.minimizer
            end

            if arma[1] > 0 && !ar_check(init[1:arma[1]])
                error("Non-stationary AR part from CSS")
            end
            

            if arma[3] > 0 && !ar_check(init[(sum(arma[1:2]) + 1):(sum(arma[1:2]) + arma[3])])
                error("Non-stationary seasonal AR part from CSS")
            end

            n_cond = 0
        end

        if transform_pars
            init = inverse_arima_parameter_transform(init, arma)

            if arma[2] > 0
                ind = (arma[1]+1):(arma[1]+arma[2])
                init[ind] .= ma_invert(init[ind])
            end

            if arma[4] > 0
                ind = (sum(arma[1:3]) + 1) : (sum(arma[1:3]) + arma[4])
                init[ind] .= ma_invert(init[ind])
            end
        end

        trarma = transform_arima_parameters(init, arma, transform_pars)
        mod = initialize_arima_state(
            trarma[1],
            trarma[2],
            Delta;
            kappa = kappa,
            SSinit = SSinit,
        )

        if no_optim

            res = (
                converged = true,
                minimizer = zeros(0),
                minimum = armafn(zeros(0), transform_pars),
            )
        else
            ctrl = copy(optim_control)
            ctrl["parscale"] = parscale

            opt = optim(
                init[mask],
                p -> armafn(p, transform_pars);
                method = optim_method,
                control = ctrl,
            )
            res = (
                converged = opt.convergence == 0,
                minimizer = opt.par,
                minimum = opt.value,
            )
        end

        if !res.converged
            @warn "Possible convergence problem: convergence code $(opt.convergence)"
        end

        coef[mask] .= res.minimizer

        if transform_pars
            if arma[2] > 0
                ind = (arma[1]+1):(arma[1]+arma[2])
                if all(mask[ind])
                    coef[ind] .= ma_invert(coef[ind])
                end
            end

            if arma[4] > 0
                ind = (sum(arma[1:3]) + 1) : (sum(arma[1:3]) + arma[4])
                if all(mask[ind])
                    coef[ind] .= ma_invert(coef[ind])
                end
            end

            if any(coef[mask] .!= res.minimizer)
                old_convergence = res.converged

                ctrl = copy(optim_control)
                ctrl["parscale"] = parscale
                ctrl["maxit"] = 0

                opt = optim(
                    coef[mask],
                    p -> armafn(p, true);
                    method = optim_method,
                    control = ctrl,
                )
                res = (
                    converged = opt.convergence == 0,
                    minimizer = opt.par,
                    minimum = opt.value,
                )

                hessian = optim_hessian(p -> armafn(p, true), res.minimizer)

                coef[mask] .= res.minimizer
            else
                hessian = optim_hessian(p -> armafn(p, true), res.minimizer)
            end

            A = compute_arima_transform_gradient(coef, arma)
            A = A[mask, mask]
            var = A' * ((hessian * n_used) \ A)
            coef = undo_arima_parameter_transform(coef, arma)
        else
            if no_optim
                var = zeros(0)
            else
                hessian = optim_hessian(p -> armafn(p, true), res.minimizer)
                var = inv(hessian * n_used)
            end
        end

        trarma = transform_arima_parameters(coef, arma, false)
        mod = initialize_arima_state(
            trarma[1],
            trarma[2],
            Delta;
            kappa = kappa,
            SSinit = SSinit,
        )

        val = if ncxreg > 0
            compute_state_space_likelihood(x - xreg * coef[narma+1 : narma+ncxreg], mod)
        else
            compute_state_space_likelihood(x, mod)
        end
        sigma2 = val[1][1] / n_used
    end

    # # Final steps
    value = 2 * n_used * res.minimum + n_used + n_used * log(2 * π)
    
    if method != "CSS"
        aic = value + 2 * sum(mask) + 2
    else
        aic = NaN
    end
    loglik = -0.5 * value

    if ncxreg > 0 && !orig_xreg
        ind = narma .+ (1:ncxreg)
        coef[ind] = S.V * coef[ind]
        A = Matrix{Float64}(I, narma + ncxreg, narma + ncxreg)
        A[ind, ind] = S.V
        A = A[mask, mask]
        var = A * var * transpose(A)
    end

    arima_coef = prep_coefs(arma, coef, nmxreg, ncxreg)
    resid = val[:residuals]
    fitted = y .- resid

    if ncxreg > 0
        fit_method = "Regression with ARIMA($(order.p),$(order.d),$(order.q))(" * 
        "$(seasonal.p),$(seasonal.d),$(seasonal.q))[$m]" * 
        " errors"
    else
        fit_method = "ARIMA($(order.p),$(order.d),$(order.q))(" * 
        "$(seasonal.p),$(seasonal.d),$(seasonal.q))[$m]"
    end
    
    if size(var) == (0, )
        var = reshape(var, 0, 0)
    end
    result = ArimaFit(
        y,
        fitted,
        arima_coef,
        sum(resid .^ 2) / n_used,
        var,
        mask,
        loglik,
        aic,
        nothing,
        nothing,
        nothing,
        arma,
        resid,
        res.converged,
        ncond,
        n_used,
        mod,
        xreg_original,
        fit_method,
        nothing,
        nothing,
        nothing,
    )
    return result

end

"""
    kalman_forecast(n_ahead::Int, mod::ArimaStateSpace; update::Bool=false)

Forecast n steps ahead from the current state of the ARIMA state-space model `mod`.
Returns a NamedTuple with fields:
- `pred`: Vector of n_ahead predictions.
- `var`: Vector of corresponding (unscaled) prediction variances.
If `update` is true, the updated model is also returned in the NamedTuple as `mod`.
"""
function kalman_forecast(n_ahead::Int, mod::ArimaStateSpace; update::Bool=false)
    phi = mod.phi
    theta = mod.theta
    delta = mod.Delta
    Z = mod.Z
    a = copy(mod.a)
    P = copy(mod.P)
    Pnew = copy(mod.Pn)
    Tmat = mod.T
    V = mod.V
    h = mod.h

    p = length(phi)
    q = length(theta)
    d = length(delta)
    rd = length(a)
    r = rd - d

    #a[1:r] .= 0.0

    forecasts = Vector{Float64}(undef, n_ahead)
    variances = Vector{Float64}(undef, n_ahead)

    anew = similar(a)
    mm = d > 0 ? zeros(rd, rd) : nothing

    for l in 1:n_ahead
        state_prediction!(anew, a, p, r, d, rd, phi, delta)
        a .= anew

        fc = dot(Z, a)
        forecasts[l] = fc

        if d == 0
            predict_covariance_nodiff!(Pnew, P, r, p, q, phi, theta)
        else
            predict_covariance_with_diff!(Pnew, P, r, d, p, q, rd, phi, delta, theta, mm)
        end
        P .= Pnew

        tmpvar = h + dot(Z, P, Z)
        variances[l] = tmpvar
    end

    result = (pred = forecasts, var = variances)
    if update
        updated_mod = deepcopy(mod)
        updated_mod.a .= a
        updated_mod.P .= P
        result = merge(result, (; mod = updated_mod))
    end
    return result
end

struct ArimaPredictions
    prediction::Vector{Float64}
    se::Vector{Float64}
    y::AbstractVector
    fitted::Vector{Float64}
    residuals::Vector{Float64}
    method::String
end

"""
    plot(pred::ArimaPredictions;
         levels=[80,95],
         show_fitted=true,
         show_residuals=false)

Assumes `pred.se` already = √(variance) if `pred.se_fit` was true, or NaN otherwise.
Draws:
  • dashed history (`pred.y`)
  • solid blue forecast mean
  • shaded PIs at each `levels`%
  • optional dotted fitted-values overlay
  • optional residuals subplot
"""
function plot(
    pred::ArimaPredictions;
    levels = [80, 95],
    show_fitted = true,
    show_residuals = false,
)

    n_hist = length(pred.y)
    n_fore = length(pred.prediction)
    t_hist = 1:n_hist
    t_fore = (n_hist+1):(n_hist+n_fore)
    se_vec = pred.se

    k = length(levels)
    lower = zeros(n_fore, k)
    upper = zeros(n_fore, k)
    for (i, lvl) in enumerate(levels)
        α = 1 - lvl / 100
        z = quantile(Normal(), 1 - α / 2)
        me = se_vec .* z
        lower[:, i] = pred.prediction .- me
        upper[:, i] = pred.prediction .+ me
    end

    title = "Forecast Plot from " * pred.method
    p = Plots.plot(
        t_hist,
        pred.y;
        label = "Historical Data",
        lw = 2,
        linestyle = :dash,
        title = title,
        xlabel = "Time",
        ylabel = "Value",
    )

    Plots.plot!(p, t_fore, pred.prediction; label = "Forecast Mean", lw = 3, color = :blue)

    for i = 1:k
        fillcol = i == k ? "#D5DBFF" : "#596DD5"
        ribbon = upper[:, i] .- lower[:, i]
        Plots.plot!(
            p,
            t_fore,
            upper[:, i];
            ribbon = ribbon,
            fillcolor = fillcol,
            linecolor = fillcol,
            label = "$(levels[i])% PI",
        )
    end

    if show_fitted && !isempty(pred.fitted)
        Plots.plot!(p, t_hist, pred.fitted; label = "Fitted Values", linestyle = :dot)
    end

    if show_residuals && !isempty(pred.residuals)
        pr = Plots.plot(t_hist, pred.residuals; label = "Residuals", lw = 1, color = :red)
        p = Plots.plot(p, pr; layout = (2, 1), link = :x)
    end

    return p
end

function predict_arima(model::ArimaFit, n_ahead::Int=1; 
    newxreg::Union{Nothing, NamedMatrix}= nothing, se_fit::Bool=true)

    myncol(x) = isnothing(x) ? 0 : size(x, 2)

    if newxreg isa NamedMatrix
        newxreg = align_columns(newxreg, model.xreg.colnames)
        newxreg = newxreg.data
    end

    arma = model.arma
    coefs = vec(model.coef.data)
    coef_names = model.coef.colnames
    narma = sum(arma[1:4])
    ncoefs = length(coefs)

    intercept_idx = findfirst(==("intercept"), coef_names)
    has_intercept = !isnothing(intercept_idx)

    ncxreg = model.xreg isa NamedMatrix ? size(model.xreg.data, 2) : 0
    if myncol(newxreg) != ncxreg
        throw(ArgumentError("`xreg` and `newxreg` have different numbers of columns"))
    end
    xm = zeros(n_ahead)
    if ncoefs > narma
        if has_intercept && coef_names[narma+1] == "intercept"
            intercept_col = ones(n_ahead, 1)
            usexreg = isnothing(newxreg) ? intercept_col : hcat(intercept_col, newxreg)
            reg_coef_inds = (narma+1):ncoefs
        else
            usexreg = newxreg
            reg_coef_inds = (narma+1):ncoefs
        end
        if narma == 0
            xm = vec(usexreg * coefs)
        else
            xm = vec(usexreg * coefs[reg_coef_inds])
        end
    end

    pred, se = kalman_forecast(n_ahead, model.model, update=false)
    
    pred = pred .+ xm
    if se_fit
        se = sqrt.(se .* model.sigma2)
    else
        se = fill(NaN, length(pred))
    end

    return ArimaPredictions(pred, se, model.y, model.fitted, model.residuals, model.method)

end


function fitted(model::ArimaFit)
    return model.fitted
end

function residuals(model::ArimaFit)
    return model.residuals
end

# function forecast(model::ArimaFit; h::Int, xreg = nothing, level::Vector{Int} = [80, 95])

#     forecasts = predict_arima(model, h, newxreg = xreg, se_fit = true)

#     se = forecasts.se
#     forecasts = forecasts.prediction

#     z = level .|> l -> quantile(Normal(), 0.5 + l / 200)

#     upper = reduce(hcat, [forecasts .+ zi .* se for zi in z])
#     lower = reduce(hcat, [forecasts .- zi .* se for zi in z])

#     fits = fitted(model)
#     res = residuals(model)

#     return Forecast(
#         model,
#         model.method,
#         forecasts,
#         level,
#         model.y,
#         upper,
#         lower,
#         fits,
#         res,
#     )
# end
