"""
    ACFResult

Container for sample autocorrelation function (ACF) results.

# Fields
- `values::Vector{Float64}`: Autocorrelations for lags `0:n_lags`.
- `lags::Vector{Int}`: Lag indices `0:n_lags`.
- `n::Int`: Number of observations used.
- `m::Int`: Seasonal frequency metadata provided by the caller.
- `ci::Float64`: Approximate 95% confidence limit `1.96 / sqrt(n)`.
- `type::Symbol`: Always `:acf`.

# References
- Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.), Def. 1.4.4.
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

Container for sample partial autocorrelation function (PACF) results.

# Fields
- `values::Vector{Float64}`: Partial autocorrelations for lags `1:n_lags`.
- `lags::Vector{Int}`: Lag indices `1:n_lags`.
- `n::Int`: Number of observations used.
- `m::Int`: Seasonal frequency metadata provided by the caller.
- `ci::Float64`: Approximate 95% confidence limit `1.96 / sqrt(n)`.
- `type::Symbol`: Always `:pacf`.

# References
- Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.), Sec. 2.5.3.
"""
struct PACFResult
    values::Vector{Float64}
    lags::Vector{Int}
    n::Int
    m::Int
    ci::Float64
    type::Symbol
end

const _ACF_DEFAULT_LAG_FACTOR = 10.0
const _ACF_NORMAL_95_QUANTILE = 1.96

@inline _acf_default_n_lags(sample_size::Int) = min(floor(Int, _ACF_DEFAULT_LAG_FACTOR * log10(sample_size)), sample_size - 1)
@inline _acf_confidence_interval(sample_size::Int) = _ACF_NORMAL_95_QUANTILE / sqrt(sample_size)

function _validate_lag_inputs(sample_size::Int, seasonal_period::Int, n_lags::Int; pacf_mode::Bool=false)
    sample_size >= 2 || throw(ArgumentError("series must contain at least 2 observations"))
    seasonal_period >= 1 || throw(ArgumentError("frequency m must be at least 1"))
    if pacf_mode
        n_lags >= 1 || throw(ArgumentError("n_lags must be at least 1 for PACF"))
    else
        n_lags >= 0 || throw(ArgumentError("n_lags must be non-negative"))
    end
    n_lags < sample_size || throw(ArgumentError("n_lags must be less than length of series"))
    return nothing
end

function _acf_values(clean_series::Vector{Float64}, n_lags::Int; demean::Bool=true)
    sample_size = length(clean_series)
    centered_series = if demean
        series_mean = mean(clean_series)
        centered_buffer = similar(clean_series)
        @inbounds for observation_index in eachindex(clean_series)
            centered_buffer[observation_index] = clean_series[observation_index] - series_mean
        end
        centered_buffer
    else
        clean_series
    end

    gamma_0 = dot(centered_series, centered_series) / sample_size
    acf_values = zeros(Float64, n_lags + 1)
    acf_values[1] = 1.0

    if gamma_0 == 0.0
        acf_values .= 1.0
        return acf_values
    end

    @inbounds for lag_index in 1:n_lags
        autocovariance_lag = dot(
            @view(centered_series[1:(sample_size - lag_index)]),
            @view(centered_series[(lag_index + 1):sample_size]),
        ) / sample_size
        acf_values[lag_index + 1] = autocovariance_lag / gamma_0
    end

    return acf_values
end

"""
    acf(y, m, n_lags=nothing; demean=true) -> ACFResult

Compute the sample autocorrelation function.

Implemented equations:

- `n_lags = min(floor(Int, 10 * log10(n)), n - 1)` when omitted.
- `gamma_hat(k) = (1/n) * sum((y_t - y_bar) * (y_{t+k} - y_bar))`.
- `rho_hat(k) = gamma_hat(k) / gamma_hat(0)` with `rho_hat(0) = 1`.
- `CI_95 = ± 1.96 / sqrt(n)`.

# References
- Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.), Def. 1.4.4 and Sec. 1.6.
"""
function acf(y::AbstractVector{<:Real}, m::Int, n_lags::Union{Int,Nothing}=nothing; demean::Bool=true)
    clean_series = collect(Float64, y)
    sample_size = length(clean_series)
    selected_lags = isnothing(n_lags) ? _acf_default_n_lags(sample_size) : n_lags
    _validate_lag_inputs(sample_size, m, selected_lags; pacf_mode=false)

    acf_values = _acf_values(clean_series, selected_lags; demean=demean)
    lag_indices = collect(0:selected_lags)
    confidence_interval = _acf_confidence_interval(sample_size)

    return ACFResult(acf_values, lag_indices, sample_size, m, confidence_interval, :acf)
end

"""
    pacf(y, m, n_lags=nothing) -> PACFResult

Compute the sample partial autocorrelation function (PACF) using the
Durbin-Levinson recursion on sample ACF values.

Implemented equations:
- `phi_{1,1} = rho_hat(1)`.
- `phi_{k,k} = (rho_hat(k) - sum(phi_{k-1,j} * rho_hat(k-j), j=1..k-1)) /
               (1 - sum(phi_{k-1,j} * rho_hat(j), j=1..k-1))`.
- `phi_{k,j} = phi_{k-1,j} - phi_{k,k} * phi_{k-1,k-j}`.

# References
- Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.), Sec. 2.5.3.
"""
function pacf(y::AbstractVector{<:Real}, m::Int, n_lags::Union{Int,Nothing}=nothing)
    clean_series = collect(Float64, y)
    sample_size = length(clean_series)
    selected_lags = isnothing(n_lags) ? _acf_default_n_lags(sample_size) : n_lags
    _validate_lag_inputs(sample_size, m, selected_lags; pacf_mode=true)

    rho_hat = _acf_values(clean_series, selected_lags; demean=true)

    pacf_values = zeros(Float64, selected_lags)
    previous_reflection_coefficients = zeros(Float64, selected_lags)
    current_reflection_coefficients = zeros(Float64, selected_lags)
    machine_tolerance = eps(Float64)

    previous_reflection_coefficients[1] = rho_hat[2]
    pacf_values[1] = previous_reflection_coefficients[1]

    @inbounds for lag_index in 2:selected_lags
        numerator_term = rho_hat[lag_index + 1]
        denominator_term = 1.0
        for inner_index in 1:(lag_index - 1)
            previous_coefficient = previous_reflection_coefficients[inner_index]
            numerator_term -= previous_coefficient * rho_hat[lag_index - inner_index + 1]
            denominator_term -= previous_coefficient * rho_hat[inner_index + 1]
        end

        reflection_coefficient = if abs(denominator_term) <= machine_tolerance
            0.0
        else
            numerator_term / denominator_term
        end
        pacf_values[lag_index] = reflection_coefficient

        for inner_index in 1:(lag_index - 1)
            current_reflection_coefficients[inner_index] =
                previous_reflection_coefficients[inner_index] -
                reflection_coefficient * previous_reflection_coefficients[lag_index - inner_index]
        end
        current_reflection_coefficients[lag_index] = reflection_coefficient
        previous_reflection_coefficients, current_reflection_coefficients =
            current_reflection_coefficients, previous_reflection_coefficients
    end

    lag_indices = collect(1:selected_lags)
    confidence_interval = _acf_confidence_interval(sample_size)
    return PACFResult(pacf_values, lag_indices, sample_size, m, confidence_interval, :pacf)
end

function Base.show(io::IO, result::ACFResult)
    println(io, "ACF Result")
    println(io, "  Series length: ", result.n)
    println(io, "  Frequency (m): ", result.m)
    println(io, "  Number of lags: ", length(result.lags) - 1)
    println(io, "  95% CI: +/-", round(result.ci, digits=4))
    println(io, "  ACF values (first 10): ", round.(result.values[1:min(10, end)], digits=4))
    return
end

function Base.show(io::IO, result::PACFResult)
    println(io, "PACF Result")
    println(io, "  Series length: ", result.n)
    println(io, "  Frequency (m): ", result.m)
    println(io, "  Number of lags: ", length(result.lags))
    println(io, "  95% CI: +/-", round(result.ci, digits=4))
    println(io, "  PACF values (first 10): ", round.(result.values[1:min(10, end)], digits=4))
    return
end

"""
    plot(result::ACFResult; kwargs...)
    plot(result::PACFResult; kwargs...)

Plot ACF or PACF with confidence bands.

This function is implemented in the DurbynPlotsExt extension module.
Load Plots.jl to enable plotting: `using Plots`.
"""
function plot end
