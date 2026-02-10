"""
    Diffusion Model Initialization

Functions to compute initial parameter estimates for each diffusion model type.
These provide starting points for optimization.
"""

using LinearAlgebra: dot

"""
    bass_init(y) -> NamedTuple

Initialize Bass model parameters using linear regression approach (Bass 1969).

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Returns
NamedTuple with initial parameters `(m, p, q)`.

# Method
1. Compute cumulative adoption Y = cumsum(y)
2. Fit linear regression: y ~ Y + Y²
3. Solve quadratic to extract m, p, q
"""
function bass_init(y::AbstractVector{<:Real})
    T = Float64
    y = collect(T.(y))
    n = length(y)

    Y = cumsum(y)

    X = hcat(ones(T, n), Y, Y .^ 2)

    cf = X \ y

    c0, c1, c2 = cf[1], cf[2], cf[3]

    discriminant = c1^2 - 4 * c2 * c0

    if discriminant < 0 || abs(c2) < 1e-12
        m = Y[end] * 1.5
        p = T(0.03)
        q = T(0.38)
    else
        sqrt_disc = sqrt(discriminant)
        m1 = (-c1 + sqrt_disc) / (2 * c2)
        m2 = (-c1 - sqrt_disc) / (2 * c2)

        m = max(m1, m2)

        if abs(m) > 1e-12
            p = c0 / m
            q = c1 + p
        else
            p = T(0.03)
            q = T(0.38)
        end
    end

    if m <= 0
        m = Y[end]
    end
    if p <= 0
        p = T(0.03)
    end
    if q <= 0
        q = T(0.38)
    end

    return (m=m, p=p, q=q)
end

"""
    gompertz_init(y; use_bass_optim=false, loss=2, mscal=true, method="L-BFGS-B", maxiter=500) -> NamedTuple

Initialize Gompertz model parameters using the method from Jukic et al. (2004).

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Keyword Arguments
- `use_bass_optim::Bool=false`: If true, use full Bass optimization for m (matches R behavior)
- `loss::Int=2`: Loss function for Bass optimization
- `mscal::Bool=true`: Scale market potential for Bass optimization
- `method::String="L-BFGS-B"`: Optimization method for Bass fitting
- `maxiter::Int=500`: Maximum iterations for Bass fitting

# Returns
NamedTuple with initial parameters `(m, a, b)`.

# Method
1. Use Bass optimization (or linear init) to get market potential estimate m
2. Select three time points and compute a, b analytically (Jukic et al. 2004)
"""
function gompertz_init(y::AbstractVector{<:Real};
                       use_bass_optim::Bool=false,
                       loss::Int=2,
                       mscal::Bool=true,
                       method::String="L-BFGS-B",
                       maxiter::Int=500)
    T = Float64
    y = collect(T.(y))
    n = length(y)

    y_clean, offset = _cleanzero(y)
    n_clean = length(y_clean)

    if n_clean < 3
        Y = cumsum(y)
        m = Y[end] * 1.5
        a = T(5.0)
        b = T(0.5)
        return (m=m, a=a, b=b)
    end

    Y = cumsum(y_clean)

    t0 = [1, max(1, floor(Int, (1 + n_clean) / 2)), n_clean]
    x0 = [Y[t0[1]], Y[t0[2]], Y[t0[3]]]

    seen = Set{T}()
    for i in 1:3
        if x0[i] in seen
            x0[i] = x0[i] + x0[i] * 1e-5
        else
            push!(seen, x0[i])
        end
    end

    if use_bass_optim
        bass_fit = _fit_bass_for_init(y_clean, loss, mscal, method, maxiter)
        m = bass_fit.m
    else
        bass_params = bass_init(y_clean)
        m = bass_params.m
    end

    x0 = max.(x0, T(1e-10))

    log_x0 = log.(x0)

    denom = log_x0[3] - 2 * log_x0[2] + log_x0[1]
    diff1 = log_x0[2] - log_x0[1]
    diff2 = log_x0[3] - log_x0[2]

    if abs(denom) < 1e-12 || abs(diff1) < 1e-12 || abs(diff2) < 1e-12
        a = T(5.0)
        b = T(0.5)
    else
        ratio = diff1 / diff2
        if ratio <= 0
            a = T(5.0)
            b = T(0.5)
        else
            exp_factor = 2 * t0[1] / (t0[3] - t0[1])
            a = (-diff1^2 / denom) * ratio^exp_factor

            b = (-2 / (t0[3] - t0[1])) * log(diff2 / diff1)

            if a <= 0
                a = T(5.0)
            end
            if b <= 0
                b = T(0.5)
            end
        end
    end

    return (m=m, a=a, b=b)
end

"""
    gsgompertz_init(y; use_bass_optim=false, loss=2, mscal=true, method="L-BFGS-B", maxiter=500) -> NamedTuple

Initialize Gamma/Shifted Gompertz model parameters.
Uses Bass model as base (assumes c=1 initially).

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Keyword Arguments
- `use_bass_optim::Bool=false`: If true, use full Bass optimization (matches R behavior)
- `loss::Int=2`: Loss function for Bass optimization
- `mscal::Bool=true`: Scale market potential for Bass optimization
- `method::String="L-BFGS-B"`: Optimization method for Bass fitting
- `maxiter::Int=500`: Maximum iterations for Bass fitting

# Returns
NamedTuple with initial parameters `(m, a, b, c)`.

# Method
From Bemmaor (1994): when c=1, GSGompertz relates to Bass via:
- a = p/q (shape parameter beta)
- b = p + q (scale parameter)
"""
function gsgompertz_init(y::AbstractVector{<:Real};
                         use_bass_optim::Bool=false,
                         loss::Int=2,
                         mscal::Bool=true,
                         method::String="L-BFGS-B",
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

Initialize Weibull model parameters using median-ranked OLS.
Based on R's diffusion package implementation.

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Returns
NamedTuple with initial parameters `(m, a, b)`.

# Method
R approach: regress log(Y) on log(log(L/(L-mdrk))) where Y is cumulative adoption.
This linearizes the Weibull CDF relationship.
"""
function weibull_init(y::AbstractVector{<:Real})
    T = Float64
    y = collect(T.(y))
    n = length(y)

    Y = cumsum(y)

    mdrk = [(i - 0.3) / (n + 0.4) for i in 1:n]

    mdrk = clamp.(mdrk, T(1e-10), T(1.0 - 1e-10))

    L = T(1.0)

    X = log.(log.(L ./ (L .- mdrk)))
    Z = log.(Y)

    valid = isfinite.(X) .& isfinite.(Z)
    X_valid = X[valid]
    Z_valid = Z[valid]

    if length(X_valid) < 2
        m = Y[end] * 1.5
        a = T(n / 2)
        b = T(2.0)
        return (m=m, a=a, b=b)
    end

    X_mat = hcat(ones(length(X_valid)), X_valid)
    cf = X_mat \ Z_valid

    b = cf[2]
    a = exp(-(cf[1] / b))
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
This matches R's behavior of calling diffusionEstim(..., type="bass") in gompertzInit/gsgInit.
Respects the caller's method and maxiter settings.
"""
function _fit_bass_for_init(y::AbstractVector{<:Real}, loss::Int, mscal::Bool,
                            method::String, maxiter::Int)
    fit = fit_diffusion(y; model_type=Bass, loss=loss, mscal=mscal, cleanlead=true,
                        method=method, maxiter=maxiter, initpar="linearize")
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

Return preset initial parameter values for diffusion models (matches R's initpar="preset").

# Arguments
- `model_type::DiffusionModelType`: The type of diffusion model
- `y::Vector`: Adoption per period data (used for scaling m if mscal=true)

# Keyword Arguments
- `mscal::Bool=true`: If true, scale market potential by 10*sum(y)

# Returns
NamedTuple with preset initial parameters.

# Notes
R preset values:
- Bass: (m=0.5, p=0.5, q=0.5)
- Gompertz: (m=1, a=1, b=1)
- GSGompertz: (m=0.5, a=0.5, b=0.5, c=0.5)
- Weibull: (m=0.5, a=0.5, b=0.5)
When mscal=true, m is scaled up by 10*sum(y).
"""
function preset_init(model_type::DiffusionModelType, y::AbstractVector{<:Real}; mscal::Bool=true)
    T = Float64
    y_sum = sum(y)
    scale_factor = mscal && y_sum > 0 ? 10 * y_sum : one(T)

    if model_type == Bass
        m = T(0.5) * scale_factor
        return (m=m, p=T(0.5), q=T(0.5))
    elseif model_type == Gompertz
        m = T(1.0) * scale_factor
        return (m=m, a=T(1.0), b=T(1.0))
    elseif model_type == GSGompertz
        m = T(0.5) * scale_factor
        return (m=m, a=T(0.5), b=T(0.5), c=T(0.5))
    elseif model_type == Weibull
        m = T(0.5) * scale_factor
        return (m=m, a=T(0.5), b=T(0.5))
    else
        error("Unknown model type: $model_type")
    end
end

"""
    get_init(model_type, y; initpar="linearize", loss=2, mscal=true, method="L-BFGS-B", maxiter=500) -> NamedTuple

Dispatch to the appropriate initialization function based on model type and initpar setting.

# Arguments
- `model_type::DiffusionModelType`: The type of diffusion model
- `y::Vector`: Adoption per period data

# Keyword Arguments
- `initpar::String="linearize"`: Initialization method. "linearize" uses analytical methods
  (with Bass optimization for Gompertz/GSGompertz). "preset" uses fixed preset values.
- `loss::Int=2`: Loss function for Bass optimization
- `mscal::Bool=true`: Scale market potential for optimization
- `method::String="L-BFGS-B"`: Optimization method for Bass fitting
- `maxiter::Int=500`: Maximum iterations for Bass fitting

# Returns
NamedTuple with initial parameters for the specified model type.
"""
function get_init(model_type::DiffusionModelType, y::AbstractVector{<:Real};
                  initpar::String="linearize",
                  loss::Int=2,
                  mscal::Bool=true,
                  method::String="L-BFGS-B",
                  maxiter::Int=500)
    if initpar == "preset"
        return preset_init(model_type, y; mscal=mscal)
    end

    use_bass_optim = true

    if model_type == Bass
        return bass_init(y)
    elseif model_type == Gompertz
        return gompertz_init(y; use_bass_optim=use_bass_optim, loss=loss, mscal=mscal,
                            method=method, maxiter=maxiter)
    elseif model_type == GSGompertz
        return gsgompertz_init(y; use_bass_optim=use_bass_optim, loss=loss, mscal=mscal,
                              method=method, maxiter=maxiter)
    elseif model_type == Weibull
        return weibull_init(y)
    else
        error("Unknown model type: $model_type")
    end
end
