"""
    STLResult

Container for the results of an STL decomposition.  Fields store the
seasonal, trend and remainder components directly along with the
robustness weights, smoothing windows, local polynomial degrees, jump
parameters and iteration counts.
"""
struct STLResult{T<:Real}
    seasonal::Vector{T}
    trend::Vector{T}
    remainder::Vector{T}
    weights::Vector{T}
    seasonal_window::Int
    trend_window::Int
    lowpass_window::Int
    seasonal_degree::Int
    trend_degree::Int
    lowpass_degree::Int
    seasonal_jump::Int
    trend_jump::Int
    lowpass_jump::Int
    inner_iterations::Int
    outer_iterations::Int
end

struct STLParams
    period::Int
    seasonal_bandwidth::Int
    trend_bandwidth::Int
    lowpass_bandwidth::Int
    seasonal_degree::Int
    trend_degree::Int
    lowpass_degree::Int
    seasonal_jump::Int
    trend_jump::Int
    lowpass_jump::Int
    n_inner::Int
    n_outer::Int
end

mutable struct STLWorkspace
    detrended::Vector{Float64}
    seasonal_ext::Vector{Float64}
    lowpass_buf::Vector{Float64}
    work_a::Vector{Float64}
    work_b::Vector{Float64}
    fit::Vector{Float64}

    function STLWorkspace(n::Int, period::Int)
        n_ext = n + 2 * period
        new(zeros(n_ext), zeros(n_ext), zeros(n_ext),
            zeros(n_ext), zeros(n_ext), zeros(n))
    end
end

function moving_average!(x::AbstractVector{Float64}, n::Int, window::Int, ave::AbstractVector{Float64})

    if window <= 0 || n < window
        return
    end
    output_len = n - window + 1
    window_f = Float64(window)

    running_sum = 0.0
    for i in 1:window
        running_sum += x[i]
    end
    ave[1] = running_sum / window_f
    if output_len > 1
        head = window
        tail = 0
        for j in 2:output_len
            head += 1
            tail += 1
            running_sum = running_sum - x[tail] + x[head]
            ave[j] = running_sum / window_f
        end
    end
    return
end

function lowpass_filter!(x::AbstractVector{Float64}, n::Int, period::Int,
                         trend::AbstractVector{Float64}, work::AbstractVector{Float64})

    moving_average!(x, n, period, trend)

    moving_average!(trend, n - period + 1, period, work)

    moving_average!(work, n - 2 * period + 2, 3, trend)
    return
end

function seasonal_smooth!(y::AbstractVector{Float64}, n::Int, period::Int, seasonal_bandwidth::Int, seasonal_degree::Int,
                           seasonal_jump::Int, use_weights::Bool, robustness_weights::AbstractVector{Float64},
                           season_ext::AbstractVector{Float64},
                           work1::AbstractVector{Float64}, work2::AbstractVector{Float64},
                           work3::AbstractVector{Float64}, work4::AbstractVector{Float64})

    if period < 1
        return
    end

    for j in 1:period

        k = ((n - j) ÷ period) + 1

        for i in 1:k
            idx = (i - 1) * period + j
            work1[i] = y[idx]
        end

        if use_weights
            for i in 1:k
                idx = (i - 1) * period + j
                work3[i] = robustness_weights[idx]
            end
        end

        loess_smooth!(work1, k, seasonal_bandwidth, seasonal_degree, seasonal_jump, use_weights, work3, view(work2, 2:(k + 1)), work4)

        eval_point = 0.0
        right_bound = min(seasonal_bandwidth, k)
        yfit, ok = loess_estimate!(work1, k, seasonal_bandwidth, seasonal_degree, eval_point, 1, right_bound, work4, use_weights, work3)
        if !ok
            yfit = work2[2]
        end
        work2[1] = yfit

        eval_point = float(k + 1)
        left_bound = max(1, k - seasonal_bandwidth + 1)
        yfit, ok = loess_estimate!(work1, k, seasonal_bandwidth, seasonal_degree, eval_point, left_bound, k, work4, use_weights, work3)
        if !ok
            yfit = work2[k + 1]
        end
        work2[k + 2] = yfit

        for idx in 1:(k + 2)
            pos = (idx - 1) * period + j
            season_ext[pos] = work2[idx]
        end
    end
    return
end


function robustness_weights!(y::AbstractVector{Float64}, fit::AbstractVector{Float64},
                             weights::AbstractVector{Float64},
                             residuals::AbstractVector{Float64})
    n = min(length(y), length(fit), length(weights))

    for i in 1:n
        residuals[i] = abs(y[i] - fit[i])
    end

    six_mad = 6.0 * median(view(residuals, 1:n))

    threshold_upper = 0.999 * six_mad
    threshold_lower = 0.001 * six_mad
    for i in 1:n
        r = abs(y[i] - fit[i])
        if r <= threshold_lower
            weights[i] = 1.0
        elseif r <= threshold_upper && six_mad > 0.0
            u = r / six_mad
            weights[i] = (1.0 - u^2)^2
        else
            weights[i] = 0.0
        end
    end
    return
end

function stl_inner_loop!(ws::STLWorkspace, y::AbstractVector{Float64}, n::Int,
                         season::AbstractVector{Float64}, trend::AbstractVector{Float64},
                         use_weights::Bool, robustness_weights::AbstractVector{Float64},
                         params::STLParams)

    n_extended = n + 2 * params.period
    for _iter in 1:params.n_inner

        for i in 1:n
            ws.detrended[i] = y[i] - trend[i]
        end

        seasonal_smooth!(ws.detrended, n, params.period, params.seasonal_bandwidth,
                         params.seasonal_degree, params.seasonal_jump, use_weights,
                         robustness_weights, ws.seasonal_ext, ws.lowpass_buf,
                         ws.work_a, ws.work_b, season)
        lowpass_filter!(ws.seasonal_ext, n_extended, params.period, ws.lowpass_buf, ws.detrended)
        loess_smooth!(ws.lowpass_buf, n, params.lowpass_bandwidth, params.lowpass_degree,
                      params.lowpass_jump, false, ws.work_a, ws.detrended, ws.work_b)
        for i in 1:n
            season[i] = ws.seasonal_ext[params.period + i] - ws.detrended[i]
        end
        for i in 1:n
            ws.detrended[i] = y[i] - season[i]
        end
        loess_smooth!(ws.detrended, n, params.trend_bandwidth, params.trend_degree,
                      params.trend_jump, use_weights, robustness_weights, trend, ws.lowpass_buf)
    end
    return
end

function stl_outer_loop!(ws::STLWorkspace, y::AbstractVector{Float64},
                         season::AbstractVector{Float64}, trend::AbstractVector{Float64},
                         weights::AbstractVector{Float64}, params::STLParams)
    n = length(y)
    fill!(trend, 0.0)

    adj_params = STLParams(
        max(2, params.period),
        _ensure_odd(max(3, params.seasonal_bandwidth)),
        _ensure_odd(max(3, params.trend_bandwidth)),
        _ensure_odd(max(3, params.lowpass_bandwidth)),
        params.seasonal_degree,
        params.trend_degree,
        params.lowpass_degree,
        params.seasonal_jump,
        params.trend_jump,
        params.lowpass_jump,
        params.n_inner,
        params.n_outer,
    )

    use_weights = false
    k = 0
    while true
        stl_inner_loop!(ws, y, n, season, trend, use_weights, weights, adj_params)
        k += 1
        if k > adj_params.n_outer
            break
        end
        for i = 1:n
            ws.fit[i] = trend[i] + season[i]
        end
        robustness_weights!(y, ws.fit, weights, ws.detrended)
        use_weights = true
    end
    if adj_params.n_outer <= 0
        for i = 1:n
            weights[i] = 1.0
        end
    end
    return
end

_ensure_odd(x::Int) = iseven(x) ? x + 1 : x

function stl_decompose(y::AbstractVector{Float64}, params::STLParams)
    n = length(y)
    ws = STLWorkspace(n, params.period)
    season = zeros(Float64, n)
    trend = zeros(Float64, n)
    weights = zeros(Float64, n)
    stl_outer_loop!(ws, y, season, trend, weights, params)
    return season, trend, weights
end

function _validate_degree(deg, name::AbstractString)
    d = Int(deg)
    if d < 0 || d > 1
        throw(ArgumentError("$name must be 0 or 1"))
    end
    return d
end

"""
    stl(x, m; kwargs...)

Seasonal-trend decomposition based on Loess (STL).

Decomposes the one-dimensional numeric array `x` into **seasonal**,
**trend** and **remainder** components.  `m` specifies the seasonal
period (number of observations per cycle) and must be at least two.

# References

- R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning (1990).
  *STL: A Seasonal-Trend Decomposition Procedure Based on Loess.*
  Journal of Official Statistics, 6, 3–73.
- G. Bodin, *SeasonalTrendLoess.jl*, <https://github.com/guilhermebodin/SeasonalTrendLoess.jl>

# Arguments

* `x`: A numeric vector containing the time series to be decomposed.
* `m`: An integer specifying the frequency (periodicity) of the series.

# Keyword arguments

Default parameters follow Cleveland et al. (1990).

* `seasonal_window`: Span of the seasonal smoothing window.  May be an
  integer (interpreted as a span and rounded to the nearest odd value) or
  the symbol `:periodic` to request a periodic seasonal component.
  Defaults to `:periodic`.
* `seasonal_degree`: Degree of the local polynomial used for seasonal
  smoothing (0 or 1).  Defaults to 0.
* `trend_window`: Span of the trend smoothing window.  If omitted, a
  default based on `m` and `seasonal_window` is computed.  Must be odd.
* `trend_degree`: Degree of the local polynomial used for trend
  smoothing (0 or 1).  Defaults to 1.
* `lowpass_window`: Span of the low-pass filter.  Defaults to the nearest
  odd integer greater than or equal to `m`.
* `lowpass_degree`: Degree of the local polynomial used for the low-pass
  filter.  Defaults to the value of `trend_degree`.
* `seasonal_jump`, `trend_jump`, `lowpass_jump`: Subsampling step sizes
  used when evaluating the loess smoother.  Defaults are one tenth of
  the corresponding window lengths (rounded up).
* `robust`: Logical flag indicating whether to compute robustness
  weights.  When true up to 15 outer iterations are performed; when
  false no robustness iterations are used.
* `inner`: Number of inner loop iterations.  Defaults to 1 when
  `robust` is true and 2 otherwise.
* `outer`: Number of outer robustness iterations.  Defaults to 15
  when `robust` is true and 0 otherwise.

# Returns

An [`STLResult`](@ref) containing the seasonal, trend and remainder
components along with ancillary information.

# Examples
```julia
res = stl(AirPassengers, 12)                          # periodic seasonal
res = stl(AirPassengers, 12; seasonal_window=7)       # fixed window
res = stl(AirPassengers, 12; robust=true)             # robust fitting
```
"""
function stl(
    x::AbstractVector{T},
    m::Integer;
    seasonal_window::Union{Int,Symbol} = :periodic,
    seasonal_degree::Integer = 0,
    trend_window::Union{Nothing,Integer} = nothing,
    trend_degree::Integer = 1,
    lowpass_window::Union{Nothing,Integer} = nothing,
    lowpass_degree::Integer = trend_degree,
    seasonal_jump::Union{Nothing,Integer} = nothing,
    trend_jump::Union{Nothing,Integer} = nothing,
    lowpass_jump::Union{Nothing,Integer} = nothing,
    robust::Bool = false,
    inner::Union{Nothing,Integer} = nothing,
    outer::Union{Nothing,Integer} = nothing,
) where {T<:Real}

    n = length(x)
    if m < 2 || n <= 2 * m
        throw(ArgumentError(
            "seasonal period must be at least 2 and the series must contain at least two full cycles"))
    end

    if any(ismissing, x)
        throw(ArgumentError(
            "input series contains missing values; impute or remove them before decomposition"))
    end

    periodic = false
    if isa(seasonal_window, Symbol)
        if seasonal_window === :periodic
            periodic = true
            seasonal_window_val = 10 * n + 1
            seasonal_degree = 0
        else
            throw(ArgumentError(
                "seasonal_window must be an integer or :periodic; got :$seasonal_window"))
        end
    elseif isa(seasonal_window, Integer)
        seasonal_window_val = nearest_odd(seasonal_window)
    else
        seasonal_window_val = nearest_odd(round(Int, seasonal_window))
    end

    seasonal_degree = _validate_degree(seasonal_degree, "seasonal_degree")
    trend_degree = _validate_degree(trend_degree, "trend_degree")
    lowpass_degree = _validate_degree(lowpass_degree, "lowpass_degree")

    if isnothing(trend_window)
        trend_window_val = nearest_odd(ceil(Int, 1.5 * m / (1.0 - 1.5 / seasonal_window_val)))
    else
        trend_window_val = nearest_odd(trend_window)
    end

    if isnothing(lowpass_window)
        lowpass_window_val = nearest_odd(m)
    else
        lowpass_window_val = nearest_odd(lowpass_window)
    end

    if isnothing(seasonal_jump)
        seasonal_jump_val = max(1, Int(ceil(seasonal_window_val / 10)))
    else
        seasonal_jump_val = seasonal_jump
    end
    if isnothing(trend_jump)
        trend_jump_val = max(1, Int(ceil(trend_window_val / 10)))
    else
        trend_jump_val = trend_jump
    end
    if isnothing(lowpass_jump)
        lowpass_jump_val = max(1, Int(ceil(lowpass_window_val / 10)))
    else
        lowpass_jump_val = lowpass_jump
    end

    if isnothing(inner)
        inner_val = robust ? 1 : 2
    else
        inner_val = inner
    end
    if isnothing(outer)
        outer_val = robust ? 15 : 0
    else
        outer_val = outer
    end

    xvec = collect(float.(x))

    params = STLParams(
        m, seasonal_window_val, trend_window_val, lowpass_window_val,
        seasonal_degree, trend_degree, lowpass_degree,
        seasonal_jump_val, trend_jump_val, lowpass_jump_val,
        inner_val, outer_val,
    )
    season, trend, weights = stl_decompose(xvec, params)
    remainder = xvec .- season .- trend

    if periodic
        cycle = [(i - 1) % m + 1 for i = 1:n]
        mean_by_cycle = zeros(Float64, m)
        counts = zeros(Int, m)
        for i = 1:n
            idx = cycle[i]
            mean_by_cycle[idx] += season[i]
            counts[idx] += 1
        end
        for j = 1:m
            if counts[j] > 0
                mean_by_cycle[j] /= counts[j]
            end
        end
        for i = 1:n
            season[i] = mean_by_cycle[cycle[i]]
        end
        remainder = xvec .- season .- trend
    end
    return STLResult{Float64}(
        season,
        trend,
        remainder,
        weights,
        seasonal_window_val,
        trend_window_val,
        lowpass_window_val,
        seasonal_degree,
        trend_degree,
        lowpass_degree,
        seasonal_jump_val,
        trend_jump_val,
        lowpass_jump_val,
        inner_val,
        outer_val,
    )
end
