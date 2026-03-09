"""
    Diffusion Model Initialization

Functions to compute initial parameter estimates for each diffusion model type.
These provide starting points for optimization.

# References
- Bass, F. M. (1969). A new product growth for model consumer durables.
  *Management Science*, 15(5), 215–227.
- Srinivasan, V. & Mason, C. H. (1986). Nonlinear least squares estimation of
  new product diffusion models. *Marketing Science*, 5(2), 169–178.
- Jukic, D., Kralik, G. & Scitovski, R. (2004). Least-squares fitting Gompertz
  curve. *Journal of Computational and Applied Mathematics*, 169(2), 359–375.
- Bemmaor, A. C. (1994). Modeling the diffusion of new durable goods: Word-of-mouth
  effect versus consumer heterogeneity. In G. Laurent et al. (Eds.), *Research
  Traditions in Marketing*. Kluwer Academic.
- Sharif, M. N. & Islam, M. N. (1980). The Weibull distribution as a general model
  for forecasting technological change. *Technological Forecasting and Social Change*,
  18(3), 247–256.
- Sultan, F., Farley, J. U. & Lehmann, D. R. (1990). A meta-analysis of applications
  of diffusion models. *Journal of Marketing Research*, 27(1), 70–77.
"""

using LinearAlgebra: dot

"""
    bass_init(y) -> NamedTuple

Initialize Bass model parameters using OLS on the discrete Bass equation.

Regresses period adoptions `y_t` on lagged cumulative adoption `Y_{t-1}` and
its square, following the standard linearization of Bass (1969):

    y_t = p·m + (q − p)·Y_{t-1} − (q/m)·Y_{t-1}²

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Returns
NamedTuple with initial parameters `(m, p, q)`.

# References
- Bass (1969), eq. 2; Srinivasan & Mason (1986), §2.
"""
function bass_init(y::AbstractVector{<:Real})
    T = Float64
    y = collect(T.(y))
    n = length(y)

    Y = cumsum(y)
    Y_lag = vcat(zero(T), Y[1:end-1])

    X = hcat(ones(T, n), Y_lag, Y_lag .^ 2)

    cf = X \ y

    a, b, c = cf[1], cf[2], cf[3]

    # Solve c·m² + b·m + a = 0 for m
    if abs(c) < 1e-12
        if abs(b) > 1e-12
            m = -a / b
        else
            m = Y[end] * 2.0
        end
    else
        discriminant = b^2 - 4 * c * a
        if discriminant < 0
            m = -b / (2 * c)
        else
            sqrt_disc = sqrt(discriminant)
            m1 = (-b + sqrt_disc) / (2 * c)
            m2 = (-b - sqrt_disc) / (2 * c)
            m = max(m1, m2)
        end
    end

    if m < Y[end]
        m = Y[end] * 2.0
    end

    if abs(m) > 1e-12
        p = a / m
        q = -c * m
    else
        # Sultan, Farley & Lehmann (1990) meta-analytic means
        p = T(0.03)
        q = T(0.38)
    end

    return (m=m, p=p, q=q)
end

"""
    gompertz_init(y; use_bass_optim=false, loss=2, mscal=true, method=:lbfgsb, maxiter=500) -> NamedTuple

Initialize Gompertz model parameters using the three-point method of
Jukic, Kralik & Scitovski (2004).

Selects three equally-spaced time points on the cumulative adoption curve and
solves for the Gompertz parameters `(m, a, b)` analytically.

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Keyword Arguments
- `use_bass_optim::Bool=false`: If true, use full Bass optimization for m
- `loss::Int=2`: Loss function for Bass optimization
- `mscal::Bool=true`: Scale market potential for Bass optimization
- `method::Symbol=:lbfgsb`: Optimization method for Bass fitting
- `maxiter::Int=500`: Maximum iterations for Bass fitting

# Returns
NamedTuple with initial parameters `(m, a, b)`.

# References
- Jukic, D., Kralik, G. & Scitovski, R. (2004). Least-squares fitting Gompertz
  curve. *J. Comput. Appl. Math.*, 169(2), 359–375.
"""
function gompertz_init(y::AbstractVector{<:Real};
                       use_bass_optim::Bool=false,
                       loss::Int=2,
                       mscal::Bool=true,
                       method::Symbol=:lbfgsb,
                       maxiter::Int=500)
    T = Float64
    y = collect(T.(y))
    n = length(y)

    y_clean, offset = _cleanzero(y)
    n_clean = length(y_clean)

    if n_clean < 3
        Y = cumsum(y)
        m = Y[end] * 2.0
        a = T(5.0)
        b = T(0.5)
        return (m=m, a=a, b=b)
    end

    Y = cumsum(y_clean)

    if use_bass_optim
        bass_fit = _fit_bass_for_init(y_clean, loss, mscal, method, maxiter)
        m = bass_fit.m
    else
        # Heuristic: market potential ≈ 1.5× observed total (Meade & Islam 2006, §8.3)
        m = Y[end] * 1.5
    end

    # Three equally-spaced points (Jukic et al. 2004, §3)
    t1, t2, t3 = 1, max(1, floor(Int, n_clean / 2)), n_clean
    x = max.([Y[t1], Y[t2], Y[t3]], T(1e-10))

    # Perturb duplicate values for numerical stability
    if x[1] ≈ x[2]
        x[2] = x[2] * (1 + 100 * eps(T))
    end
    if x[2] ≈ x[3]
        x[3] = x[3] * (1 + 100 * eps(T))
    end

    # Gompertz: X_t = m·exp(-a·exp(-b·t))
    # ⟹ log(X_t/m) = -a·exp(-b·t)
    # Three-point solution (Jukic 2004, eqs. 10–11):
    #   b = -2/(t3-t1) · log((log(x3/m) - log(x2/m)) / (log(x2/m) - log(x1/m)))
    #   a = -log(x1/m) / exp(-b·t1)
    lx = log.(x ./ m)

    denom = lx[3] - 2 * lx[2] + lx[1]
    diff1 = lx[2] - lx[1]
    diff2 = lx[3] - lx[2]

    if abs(denom) < 1e-12 || abs(diff1) < 1e-12 || abs(diff2) < 1e-12 || diff2 / diff1 <= 0
        a = T(5.0)
        b = T(0.5)
    else
        b = (-2 / (t3 - t1)) * log(diff2 / diff1)
        a = -lx[1] / exp(-b * t1)

        if a <= 0
            a = T(5.0)
        end
        if b <= 0
            b = T(0.5)
        end
    end

    return (m=m, a=a, b=b)
end

"""
    gsgompertz_init(y; use_bass_optim=false, loss=2, mscal=true, method=:lbfgsb, maxiter=500) -> NamedTuple

Initialize Gamma/Shifted Gompertz model parameters.

When `c = 1`, the GSGompertz CDF reduces to a form equivalent to the Bass model
(Bemmaor 1994). The mapping is: `a = p/q` (innovation-to-imitation ratio) and
`b = p + q` (combined diffusion rate).

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Keyword Arguments
- `use_bass_optim::Bool=false`: If true, use full Bass optimization
- `loss::Int=2`: Loss function for Bass optimization
- `mscal::Bool=true`: Scale market potential for Bass optimization
- `method::Symbol=:lbfgsb`: Optimization method for Bass fitting
- `maxiter::Int=500`: Maximum iterations for Bass fitting

# Returns
NamedTuple with initial parameters `(m, a, b, c)`.

# References
- Bemmaor, A. C. (1994). Modeling the diffusion of new durable goods.
"""
function gsgompertz_init(y::AbstractVector{<:Real};
                         use_bass_optim::Bool=false,
                         loss::Int=2,
                         mscal::Bool=true,
                         method::Symbol=:lbfgsb,
                         maxiter::Int=500)
    T = Float64

    if use_bass_optim
        bass_params = _fit_bass_for_init(y, loss, mscal, method, maxiter)
    else
        bass_params = bass_init(y)
    end

    m = bass_params.m
    p = bass_params.p
    q = bass_params.q

    # Bemmaor (1994): GSGompertz with c=1 ↔ Bass with a=p/q, b=p+q
    a = abs(q) > 1e-12 ? p / q : T(0.1)
    b = p + q
    c = T(1.0)

    if a <= 0
        a = T(0.1)
    end
    if b <= 0
        b = T(0.5)
    end

    return (m=m, a=a, b=b, c=c)
end

"""
    weibull_init(y) -> NamedTuple

Initialize Weibull model parameters using the probability-plot linearization
method from Sharif & Islam (1980).

Linearizes the Weibull CDF `X_t = m·(1 − exp(−(t/a)^b))` as:

    log(−log(1 − X_t/m)) = b·log(t) − b·log(a)

and estimates `(a, b)` by OLS regression.

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Returns
NamedTuple with initial parameters `(m, a, b)`.

# References
- Sharif, M. N. & Islam, M. N. (1980). The Weibull distribution as a general model
  for forecasting technological change. *Technological Forecasting and Social Change*,
  18(3), 247–256.
"""
function weibull_init(y::AbstractVector{<:Real})
    T = Float64
    y = collect(T.(y))
    n = length(y)

    Y = cumsum(y)
    m_init = Y[end] * 1.1  # 10% above observed total to keep F_i < 1

    # Weibull probability-plot linearization (Sharif & Islam 1980):
    # log(-log(1 - X_t/m)) = b·log(t) - b·log(a)
    F = Y ./ m_init
    F = clamp.(F, T(1e-10), T(1.0 - 1e-10))

    W = log.(-log.(1.0 .- F))
    log_t = log.(collect(T, 1:n))

    valid = isfinite.(W) .& isfinite.(log_t)
    W_valid = W[valid]
    log_t_valid = log_t[valid]

    if length(W_valid) < 2
        m = Y[end]
        a = T(n / 2)
        b = T(2.0)
        return (m=m, a=a, b=b)
    end

    X_mat = hcat(ones(length(W_valid)), log_t_valid)
    cf = X_mat \ W_valid

    # cf[1] = -b·log(a), cf[2] = b
    b = cf[2]
    a = b > 1e-12 ? exp(-cf[1] / b) : T(n / 2)
    m = Y[end]

    if m <= 0
        m = Y[end]
    end
    if a <= 0
        a = T(n / 2)
    end
    if b <= 0
        b = T(2.0)
    end

    return (m=m, a=a, b=b)
end

"""
    _fit_bass_for_init(y, loss, mscal, method, maxiter) -> NamedTuple

Internal helper to fit Bass model for use in Gompertz/GSGompertz initialization.
Uses a preliminary Bass fit to seed the market potential parameter `m`.
Respects the caller's method and maxiter settings.
"""
function _fit_bass_for_init(y::AbstractVector{<:Real}, loss::Int, mscal::Bool,
                            method::Symbol, maxiter::Int)
    fit = fit_diffusion(y; model_type=Bass, loss=loss, mscal=mscal, cleanlead=true,
                        cumulative=false, method=method, maxiter=maxiter, initpar=:linearize)
    return fit.params
end

"""
    _cleanzero(y; lead=true) -> (cleaned, offset)

Remove leading (or trailing) zeros from adoption data.

# Arguments
- `y::Vector`: Adoption data
- `lead::Bool=true`: If true, remove leading zeros; if false, remove trailing

# Returns
- `cleaned::Vector`: Data with zeros removed
- `offset::Int`: Number of zeros removed
"""
function _cleanzero(y::AbstractVector; lead::Bool=true)
    y = collect(y)
    n = length(y)

    if lead
        first_nonzero = findfirst(x -> x != 0, y)
        if isnothing(first_nonzero)
            return (y, 0)
        end
        return (y[first_nonzero:end], first_nonzero - 1)
    else
        last_nonzero = findlast(x -> x != 0, y)
        if isnothing(last_nonzero)
            return (y, 0)
        end
        return (y[1:last_nonzero], n - last_nonzero)
    end
end

"""
    preset_init(model_type, y; mscal=true) -> NamedTuple

Return preset initial parameter values for diffusion models based on
meta-analytic means from Sultan, Farley & Lehmann (1990).

# Arguments
- `model_type::DiffusionModelType`: The type of diffusion model
- `y::Vector`: Adoption per period data (used for scaling m)

# Keyword Arguments
- `mscal::Bool=true`: If true, scale market potential by observed total adoption

# Returns
NamedTuple with preset initial parameters.

# Notes
Default preset values (from Sultan, Farley & Lehmann 1990 meta-analysis):
- Bass: (m = 2·Y_end, p = 0.03, q = 0.38)
- Gompertz: (m = 2·Y_end, a = 5.0, b = 0.5)
- GSGompertz: (m = 2·Y_end, a = 0.08, b = 0.41, c = 1.0)
- Weibull: (m = 2·Y_end, a = n/2, b = 2.0)

# References
- Sultan, F., Farley, J. U. & Lehmann, D. R. (1990). A meta-analysis of
  applications of diffusion models. *J. Marketing Research*, 27(1), 70–77.
"""
function preset_init(model_type::DiffusionModelType, y::AbstractVector{<:Real}; mscal::Bool=true)
    T = Float64
    Y_end = max(sum(T.(y)), one(T))
    scale_factor = mscal ? 2.0 * Y_end : one(T)
    n = length(y)

    if model_type == Bass
        return (m=scale_factor, p=T(0.03), q=T(0.38))
    elseif model_type == Gompertz
        return (m=scale_factor, a=T(5.0), b=T(0.5))
    elseif model_type == GSGompertz
        # Bemmaor (1994): a = p/q ≈ 0.03/0.38 ≈ 0.079, b = p+q = 0.41
        return (m=scale_factor, a=T(0.08), b=T(0.41), c=T(1.0))
    elseif model_type == Weibull
        return (m=scale_factor, a=T(n / 2), b=T(2.0))
    else
        throw(ArgumentError("Unknown model type: $model_type"))
    end
end

"""
    get_init(model_type, y; initpar=:linearize, loss=2, mscal=true, method=:lbfgsb, maxiter=500) -> NamedTuple

Dispatch to the appropriate initialization function based on model type and initpar setting.

# Arguments
- `model_type::DiffusionModelType`: The type of diffusion model
- `y::Vector`: Adoption per period data

# Keyword Arguments
- `initpar::Symbol=:linearize`: Initialization method. `:linearize` uses analytical methods.
  `:preset` uses meta-analytic preset values.
- `loss::Int=2`: Loss function for Bass optimization
- `mscal::Bool=true`: Scale market potential for optimization
- `method::Symbol=:lbfgsb`: Optimization method for Bass fitting
- `maxiter::Int=500`: Maximum iterations for Bass fitting

# Returns
NamedTuple with initial parameters for the specified model type.
"""
function get_init(model_type::DiffusionModelType, y::AbstractVector{<:Real};
                  initpar::Symbol=:linearize,
                  loss::Int=2,
                  mscal::Bool=true,
                  method::Symbol=:lbfgsb,
                  maxiter::Int=500)
    if initpar === :preset
        return preset_init(model_type, y; mscal=mscal)
    end

    if model_type == Bass
        return bass_init(y)
    elseif model_type == Gompertz
        return gompertz_init(y; use_bass_optim=false, loss=loss, mscal=mscal,
                            method=method, maxiter=maxiter)
    elseif model_type == GSGompertz
        return gsgompertz_init(y; use_bass_optim=false, loss=loss, mscal=mscal,
                              method=method, maxiter=maxiter)
    elseif model_type == Weibull
        return weibull_init(y)
    else
        throw(ArgumentError("Unknown model type: $model_type"))
    end
end
