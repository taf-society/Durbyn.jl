"""
    KWFilterResult

Result of applying a Kolmogorov-Wiener optimal filter to a time series.

The filter minimizes MSE relative to an ideal filter (bandpass, HP, Butterworth)
by accounting for the autocovariance structure of the data-generating process,
estimated via an ARIMA model. Based on Schleicher (2004).

# Fields
- `filtered::Vector{Float64}`: Filtered output (length T)
- `weights::Matrix{Float64}`: T x T filter weight matrix (row t = weights for observation t)
- `ideal_coefs::Vector{Float64}`: Truncated ideal impulse response (length 2*maxcoef+1)
- `filter_type::Symbol`: One of :bandpass, :hp, :butterworth, :custom
- `arima_model::Union{ArimaFit,Nothing}`: Fitted ARIMA model used for autocovariance
- `gamma::Vector{Float64}`: Autocovariance sequence of stationary component
- `d::Int`: Integration order (nonseasonal d + seasonal D)
- `y::Vector{Float64}`: Original series
- `output::Symbol`: :cycle or :trend
- `params::Dict{Symbol,Any}`: Filter-specific parameters (e.g., lambda, low, high)
"""
struct KWFilterResult
    filtered::Vector{Float64}
    weights::Matrix{Float64}
    ideal_coefs::Vector{Float64}
    filter_type::Symbol
    arima_model::Union{ArimaFit,Nothing}
    gamma::Vector{Float64}
    d::Int
    y::Vector{Float64}
    output::Symbol
    params::Dict{Symbol,Any}
end

fitted(r::KWFilterResult) = r.filtered

residuals(r::KWFilterResult) = r.y .- r.filtered
