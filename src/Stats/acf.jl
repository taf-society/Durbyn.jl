"""
    ACFResult

Container for the results of an ACF (Autocorrelation Function) computation.

# Fields
- `values::Vector{Float64}`: ACF values at each lag (including lag 0)
- `lags::Vector{Int}`: Lag indices (0, 1, 2, ..., nlags)
- `n::Int`: Length of the original time series
- `m::Int`: Frequency/seasonal period of the data
- `ci::Float64`: Critical value for 95% confidence interval (±1.96/√n)
- `type::Symbol`: Always `:acf`

# Usage
```julia
result = acf(y, m)
result = acf(y, m, 20)
plot(result)
```
"""
struct ACFResult
    values::Vector{Float64}
    lags::Vector{Int}
    n::Int
    m::Int
    ci::Float64
    type::Symbol
end

"""
    PACFResult

Container for the results of a PACF (Partial Autocorrelation Function) computation.

# Fields
- `values::Vector{Float64}`: PACF values at each lag (lags 1, 2, ..., nlags)
- `lags::Vector{Int}`: Lag indices (1, 2, ..., nlags)
- `n::Int`: Length of the original time series
- `m::Int`: Frequency/seasonal period of the data
- `ci::Float64`: Critical value for 95% confidence interval (±1.96/√n)
- `type::Symbol`: Always `:pacf`

# Usage
```julia
result = pacf(y, m)
result = pacf(y, m, 20)
plot(result)
```
"""
struct PACFResult
    values::Vector{Float64}
    lags::Vector{Int}
    n::Int
    m::Int
    ci::Float64
    type::Symbol
end

"""
    acf(y, m, nlags=nothing; demean=true) -> ACFResult

Compute the sample autocorrelation function (ACF) of a time series.

# Arguments
- `y::AbstractVector`: Input time series
- `m::Int`: Frequency/seasonal period of the data
- `nlags::Union{Int,Nothing}=nothing`: Number of lags to compute. If `nothing`,
  defaults to `min(10*log10(n), n-1)` following R's convention.
- `demean::Bool=true`: Whether to subtract the mean before computing ACF

# Returns
`ACFResult` containing autocorrelations and metadata. Use `plot(result)` to visualize.

# Details
Uses the standard biased estimator:
```math
\\hat{\\rho}(k) = \\frac{\\sum_{t=1}^{n-k} (y_t - \\bar{y})(y_{t+k} - \\bar{y})}{\\sum_{t=1}^{n} (y_t - \\bar{y})^2}
```

# Example
```julia
y = randn(100)
result = acf(y, 12)
result.values
result.lags
plot(result)
```

# References
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
  Time Series Analysis: Forecasting and Control. Wiley.
"""
function acf(y::AbstractVector{T}, m::Int, nlags::Union{Int,Nothing}=nothing; demean::Bool=true) where T<:Real
    n = length(y)

    if isnothing(nlags)
        nlags = min(floor(Int, 10 * log10(n)), n - 1)
    end

    if nlags < 0
        throw(ArgumentError("nlags must be non-negative"))
    end
    if nlags >= n
        throw(ArgumentError("nlags must be less than length of series"))
    end
    if m < 1
        throw(ArgumentError("frequency m must be at least 1"))
    end

    y_centered = demean ? y .- mean(y) : y
    c0 = sum(y_centered .^ 2) / n

    if c0 == 0
        values = ones(Float64, nlags + 1)
    else
        values = zeros(Float64, nlags + 1)
        values[1] = 1.0

        for k in 1:nlags
            ck = sum(y_centered[1:n-k] .* y_centered[k+1:n]) / n
            values[k + 1] = ck / c0
        end
    end

    lags = collect(0:nlags)
    ci = 1.96 / sqrt(n)

    return ACFResult(values, lags, n, m, ci, :acf)
end

"""
    pacf(y, m, nlags=nothing) -> PACFResult

Compute the sample partial autocorrelation function (PACF) of a time series
using the Durbin-Levinson algorithm.

# Arguments
- `y::AbstractVector`: Input time series
- `m::Int`: Frequency/seasonal period of the data
- `nlags::Union{Int,Nothing}=nothing`: Number of lags to compute. If `nothing`,
  defaults to `min(10*log10(n), n-1)` following R's convention.

# Returns
`PACFResult` containing partial autocorrelations and metadata. Use `plot(result)` to visualize.

# Example
```julia
y = randn(100)
result = pacf(y, 12)
result.values
result.lags
plot(result)
```
"""
function pacf(y::AbstractVector{T}, m::Int, nlags::Union{Int,Nothing}=nothing) where T<:Real
    n = length(y)

    if isnothing(nlags)
        nlags = min(floor(Int, 10 * log10(n)), n - 1)
    end

    if nlags < 1
        throw(ArgumentError("nlags must be at least 1"))
    end
    if nlags >= n
        throw(ArgumentError("nlags must be less than length of series"))
    end
    if m < 1
        throw(ArgumentError("frequency m must be at least 1"))
    end

    acf_result = acf(y, m, nlags)
    r = acf_result.values

    pacf_vals = zeros(Float64, nlags)
    phi = zeros(Float64, nlags, nlags)

    phi[1, 1] = r[2]
    pacf_vals[1] = phi[1, 1]

    for k in 2:nlags
        num = r[k + 1]
        den = 1.0
        for j in 1:k-1
            num -= phi[k-1, j] * r[k - j + 1]
            den -= phi[k-1, j] * r[j + 1]
        end

        if abs(den) < 1e-10
            phi[k, k] = 0.0
        else
            phi[k, k] = num / den
        end
        pacf_vals[k] = phi[k, k]

        for j in 1:k-1
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
        end
    end

    lags = collect(1:nlags)
    ci = 1.96 / sqrt(n)

    return PACFResult(pacf_vals, lags, n, m, ci, :pacf)
end

"""
    Base.show(io::IO, result::ACFResult)

Pretty print an `ACFResult`.
"""
function Base.show(io::IO, result::ACFResult)
    println(io, "ACF Result")
    println(io, "  Series length: ", result.n)
    println(io, "  Frequency (m): ", result.m)
    println(io, "  Number of lags: ", length(result.lags) - 1)
    println(io, "  95% CI: ±", round(result.ci, digits=4))
    println(io, "  ACF values (first 10): ", round.(result.values[1:min(10, end)], digits=4))
    return
end

"""
    Base.show(io::IO, result::PACFResult)

Pretty print a `PACFResult`.
"""
function Base.show(io::IO, result::PACFResult)
    println(io, "PACF Result")
    println(io, "  Series length: ", result.n)
    println(io, "  Frequency (m): ", result.m)
    println(io, "  Number of lags: ", length(result.lags))
    println(io, "  95% CI: ±", round(result.ci, digits=4))
    println(io, "  PACF values (first 10): ", round.(result.values[1:min(10, end)], digits=4))
    return
end

"""
    plot(result::ACFResult; kwargs...)
    plot(result::PACFResult; kwargs...)

Plot ACF or PACF with confidence bands.

This function is implemented in the DurbynPlotsExt extension module.
Load Plots.jl to enable plotting: `using Plots`

# Example
```julia
using Plots
result = acf(y, 12)
plot(result)
```
"""
function plot end
