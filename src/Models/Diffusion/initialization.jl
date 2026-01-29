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

    # Cumulative adoption
    Y = cumsum(y)

    # Build design matrix for regression: y ~ 1 + Y + Y²
    X = hcat(ones(T, n), Y, Y .^ 2)

    # Solve least squares: coefficients are [c0, c1, c2]
    # where y = c0 + c1*Y + c2*Y²
    cf = X \ y

    c0, c1, c2 = cf[1], cf[2], cf[3]

    # Solve quadratic c2*m² + c1*m + c0 = 0 for m
    # Using quadratic formula
    discriminant = c1^2 - 4 * c2 * c0

    if discriminant < 0 || abs(c2) < 1e-12
        # Fallback: use last cumulative value as market estimate
        m = Y[end] * 1.5
        p = T(0.03)
        q = T(0.38)
    else
        sqrt_disc = sqrt(discriminant)
        m1 = (-c1 + sqrt_disc) / (2 * c2)
        m2 = (-c1 - sqrt_disc) / (2 * c2)

        # Select positive root larger than current cumulative
        if m1 > Y[end] && m1 > 0
            m = m1
        elseif m2 > Y[end] && m2 > 0
            m = m2
        else
            m = max(m1, m2, Y[end] * 1.5)
        end

        # Compute p and q from coefficients
        if abs(m) > 1e-12
            p = c0 / m
            q = c1 + p
        else
            p = T(0.03)
            q = T(0.38)
        end
    end

    # Ensure reasonable bounds
    m = max(m, Y[end] * 1.01)
    p = clamp(p, T(1e-6), T(1.0))
    q = clamp(q, T(1e-6), T(2.0))

    return (m=m, p=p, q=q)
end

"""
    gompertz_init(y) -> NamedTuple

Initialize Gompertz model parameters using the method from Jukic et al. (2004).

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Returns
NamedTuple with initial parameters `(m, a, b)`.

# Method
1. Use Bass initialization to get market potential estimate m
2. Select three time points and compute a, b analytically
"""
function gompertz_init(y::AbstractVector{<:Real})
    T = Float64
    y = collect(T.(y))
    n = length(y)

    # Remove leading zeros for initialization
    y_clean, offset = _cleanzero(y)
    n_clean = length(y_clean)

    if n_clean < 3
        # Fallback to simple estimates
        Y = cumsum(y)
        m = Y[end] * 1.5
        a = T(5.0)
        b = T(0.5)
        return (m=m, a=a, b=b)
    end

    # Cumulative adoption
    Y = cumsum(y_clean)

    # Select three time points: start, middle, end
    t0 = [1, max(1, floor(Int, (1 + n_clean) / 2)), n_clean]
    x0 = [Y[t0[1]], Y[t0[2]], Y[t0[3]]]

    # Use Bass to estimate m
    bass_params = bass_init(y_clean)
    m = bass_params.m

    # Ensure x0 values are positive for log
    x0 = max.(x0, T(1e-10))

    # Compute a and b using Jukic et al. (2004) formulas
    log_x0 = log.(x0)

    denom = log_x0[3] - 2 * log_x0[2] + log_x0[1]

    if abs(denom) < 1e-12
        # Fallback
        a = T(5.0)
        b = T(0.5)
    else
        diff1 = log_x0[2] - log_x0[1]
        diff2 = log_x0[3] - log_x0[2]

        if abs(diff2) < 1e-12 || diff2 / diff1 <= 0
            a = T(5.0)
            b = T(0.5)
        else
            # a = -(diff1^2 / denom) * (diff1 / diff2)^(2*t0[1] / (t0[3] - t0[1]))
            ratio = abs(diff1 / diff2)
            exp_factor = 2 * t0[1] / max(t0[3] - t0[1], 1)
            a = abs(-diff1^2 / denom) * ratio^exp_factor

            # b = -2 / (t0[3] - t0[1]) * log(diff2 / diff1)
            b = abs(-2 / max(t0[3] - t0[1], 1) * log(ratio))
        end
    end

    # Ensure reasonable bounds
    m = max(m, cumsum(y)[end] * 1.01)
    a = clamp(a, T(1e-6), T(100.0))
    b = clamp(b, T(1e-6), T(5.0))

    return (m=m, a=a, b=b)
end

"""
    gsgompertz_init(y) -> NamedTuple

Initialize Gamma/Shifted Gompertz model parameters.
Uses Bass model as base (assumes c=1 initially).

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Returns
NamedTuple with initial parameters `(m, a, b, c)`.

# Method
From Bemmaor (1994): when c=1, GSGompertz relates to Bass via:
- a = p/q (shape parameter beta)
- b = p + q (scale parameter)
"""
function gsgompertz_init(y::AbstractVector{<:Real})
    T = Float64

    # Fit Bass model first
    bass_params = bass_init(y)
    m = bass_params.m
    p = bass_params.p
    q = bass_params.q

    # Convert to GSGompertz parameters (assuming c=1)
    a = abs(q) > 1e-12 ? p / q : T(0.1)
    b = p + q
    c = T(1.0)

    # Ensure reasonable bounds
    a = clamp(a, T(1e-6), T(100.0))
    b = clamp(b, T(1e-6), T(5.0))

    return (m=m, a=a, b=b, c=c)
end

"""
    weibull_init(y) -> NamedTuple

Initialize Weibull model parameters using median-ranked OLS.
Based on Sharif and Islam (1980) and Abernethy (2006).

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Returns
NamedTuple with initial parameters `(m, a, b)`.

# Method
1. Compute cumulative adoption Y = cumsum(y)
2. Calculate median rank using Bernard's approximation
3. Fit log-linear regression to extract a and b
"""
function weibull_init(y::AbstractVector{<:Real})
    T = Float64
    y = collect(T.(y))
    n = length(y)

    # Cumulative adoption
    Y = cumsum(y)

    # Median rank using Bernard's approximation: (i - 0.3) / (n + 0.4)
    mdrk = [(i - 0.3) / (n + 0.4) for i in 1:n]

    # Ensure median ranks are in valid range
    mdrk = clamp.(mdrk, T(1e-10), T(1.0 - 1e-10))

    # Weibull linearization: log(-log(1 - F)) = b*log(t) - b*log(a)
    # Let: X = log(t), Z = log(-log(1-mdrk))
    # Then: Z = b*X - b*log(a)

    # Design matrix for regression
    X = log.(1:n)
    Z = log.(-log.(1.0 .- mdrk))

    # Filter out invalid values
    valid = isfinite.(X) .& isfinite.(Z)
    X_valid = X[valid]
    Z_valid = Z[valid]

    if length(X_valid) < 2
        # Fallback
        m = Y[end] * 1.5
        a = T(n / 2)
        b = T(2.0)
        return (m=m, a=a, b=b)
    end

    # Fit regression: Z = intercept + slope * X
    X_mat = hcat(ones(length(X_valid)), X_valid)
    cf = X_mat \ Z_valid

    intercept = cf[1]
    slope = cf[2]  # This is b

    b = max(slope, T(0.1))
    a = exp(-intercept / b)

    # Market potential from final cumulative
    m = Y[end]

    # Ensure reasonable bounds
    m = max(m, Y[end] * 1.01)
    a = clamp(a, T(1e-6), T(n * 10))
    b = clamp(b, T(0.1), T(10.0))

    return (m=m, a=a, b=b)
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
    get_init(model_type, y) -> NamedTuple

Dispatch to the appropriate initialization function based on model type.

# Arguments
- `model_type::DiffusionModelType`: The type of diffusion model
- `y::Vector`: Adoption per period data

# Returns
NamedTuple with initial parameters for the specified model type.
"""
function get_init(model_type::DiffusionModelType, y::AbstractVector{<:Real})
    if model_type == Bass
        return bass_init(y)
    elseif model_type == Gompertz
        return gompertz_init(y)
    elseif model_type == GSGompertz
        return gsgompertz_init(y)
    elseif model_type == Weibull
        return weibull_init(y)
    else
        error("Unknown model type: $model_type")
    end
end
