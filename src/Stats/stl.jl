@inline _tricube(u::Float64) = (0.0 <= u < 1.0) ? (1.0 - u^3)^3 : 0.0
@inline _bisquare(u::Float64) = (0.0 <= u < 1.0) ? (1.0 - u^2)^2 : 0.0

@inline function _mean_range(y::AbstractVector{Float64}, left::Int, right::Int)
    total = 0.0
    @inbounds for i in left:right
        total += y[i]
    end
    return total / (right - left + 1)
end

@inline function _weighted_mean_range(
    y::AbstractVector{Float64},
    weights::AbstractVector{Float64},
    left::Int,
    right::Int,
)
    weighted_sum = 0.0
    total_weight = 0.0
    @inbounds for i in left:right
        weight = weights[i]
        weighted_sum += weight * y[i]
        total_weight += weight
    end
    return total_weight > 0.0 ? weighted_sum / total_weight : _mean_range(y, left, right)
end

function _nearest_neighbour_window(x::AbstractVector{Float64}, x0::Float64, bandwidth::Int)
    n = length(x)
    n > 0 || throw(ArgumentError("loess requires at least one observation"))
    q = clamp(Int(bandwidth), 1, n)
    if q == n
        return 1, n
    end

    left_ptr = searchsortedlast(x, x0)
    right_ptr = left_ptr + 1
    left = right_ptr
    right = left_ptr

    for _ in 1:q
        if left_ptr < 1
            right = right_ptr
            right_ptr += 1
        elseif right_ptr > n
            left = left_ptr
            left_ptr -= 1
        else
            dist_left = abs(x0 - x[left_ptr])
            dist_right = abs(x[right_ptr] - x0)
            if dist_left <= dist_right
                left = left_ptr
                left_ptr -= 1
            else
                right = right_ptr
                right_ptr += 1
            end
        end
    end

    return left, right
end

function _loess_estimate_window(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    x0::Float64,
    left::Int,
    right::Int,
    degree::Int,
    robustness_weights::Union{Nothing,AbstractVector{Float64}},
)
    radius = max(abs(x0 - x[left]), abs(x[right] - x0))

    if radius == 0.0
        return isnothing(robustness_weights) ?
               _mean_range(y, left, right) :
               _weighted_mean_range(y, robustness_weights, left, right)
    end

    sum_w = 0.0
    sum_wδ = 0.0
    sum_wδ2 = 0.0
    sum_wy = 0.0
    sum_wδy = 0.0

    if isnothing(robustness_weights)
        @inbounds for i in left:right
            δ = x[i] - x0
            weight = _tricube(abs(δ) / radius)
            if weight <= 0.0
                continue
            end

            yi = y[i]
            sum_w += weight
            sum_wδ += weight * δ
            sum_wδ2 += weight * δ * δ
            sum_wy += weight * yi
            sum_wδy += weight * δ * yi
        end
    else
        @inbounds for i in left:right
            δ = x[i] - x0
            weight = _tricube(abs(δ) / radius) * robustness_weights[i]
            if weight <= 0.0
                continue
            end

            yi = y[i]
            sum_w += weight
            sum_wδ += weight * δ
            sum_wδ2 += weight * δ * δ
            sum_wy += weight * yi
            sum_wδy += weight * δ * yi
        end
    end

    sum_w > 0.0 || return _mean_range(y, left, right)
    degree == 0 && return sum_wy / sum_w

    determinant = sum_w * sum_wδ2 - sum_wδ^2
    scale = max(sum_w * sum_wδ2, sum_wδ^2, 1.0)
    if determinant <= eps(Float64) * scale
        return sum_wy / sum_w
    end

    return (sum_wδ2 * sum_wy - sum_wδ * sum_wδy) / determinant
end

function _loess_estimate_regular(
    y::AbstractVector{Float64},
    x_start::Float64,
    x_step::Float64,
    x0::Float64,
    left::Int,
    right::Int,
    degree::Int,
    robustness_weights::Union{Nothing,AbstractVector{Float64}},
)
    left_x = x_start + (left - 1) * x_step
    right_x = x_start + (right - 1) * x_step
    radius = max(abs(x0 - left_x), abs(right_x - x0))

    if radius == 0.0
        return isnothing(robustness_weights) ?
               _mean_range(y, left, right) :
               _weighted_mean_range(y, robustness_weights, left, right)
    end

    sum_w = 0.0
    sum_wδ = 0.0
    sum_wδ2 = 0.0
    sum_wy = 0.0
    sum_wδy = 0.0

    xi = left_x
    if isnothing(robustness_weights)
        @inbounds for i in left:right
            δ = xi - x0
            weight = _tricube(abs(δ) / radius)
            if weight > 0.0
                yi = y[i]
                sum_w += weight
                sum_wδ += weight * δ
                sum_wδ2 += weight * δ * δ
                sum_wy += weight * yi
                sum_wδy += weight * δ * yi
            end
            xi += x_step
        end
    else
        @inbounds for i in left:right
            δ = xi - x0
            weight = _tricube(abs(δ) / radius) * robustness_weights[i]
            if weight > 0.0
                yi = y[i]
                sum_w += weight
                sum_wδ += weight * δ
                sum_wδ2 += weight * δ * δ
                sum_wy += weight * yi
                sum_wδy += weight * δ * yi
            end
            xi += x_step
        end
    end

    sum_w > 0.0 || return _mean_range(y, left, right)
    degree == 0 && return sum_wy / sum_w

    determinant = sum_w * sum_wδ2 - sum_wδ^2
    scale = max(sum_w * sum_wδ2, sum_wδ^2, 1.0)
    if determinant <= eps(Float64) * scale
        return sum_wy / sum_w
    end

    return (sum_wδ2 * sum_wy - sum_wδ * sum_wδy) / determinant
end

function loess_estimate(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    x0::Float64,
    bandwidth::Int,
    degree::Int;
    robustness_weights::Union{Nothing,AbstractVector{Float64}} = nothing,
)
    left, right = _nearest_neighbour_window(x, x0, bandwidth)
    return _loess_estimate_window(x, y, x0, left, right, degree, robustness_weights)
end

function loess_smooth!(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    x_eval::AbstractVector{Float64},
    bandwidth::Int,
    degree::Int,
    jump::Int,
    robustness_weights::Union{Nothing,AbstractVector{Float64}},
    output::AbstractVector{Float64},
)
    n_eval = length(x_eval)
    n_eval == length(output) || throw(ArgumentError("output length must match x_eval length"))
    n_eval == 0 && return

    step = max(1, Int(jump))
    last_idx = 0

    for eval_idx in 1:step:n_eval
        output[eval_idx] = loess_estimate(
            x, y, x_eval[eval_idx], bandwidth, degree;
            robustness_weights = robustness_weights,
        )

        if last_idx > 0
            x_left = x_eval[last_idx]
            x_right = x_eval[eval_idx]
            y_left = output[last_idx]
            y_right = output[eval_idx]
            span = x_right - x_left
            for fill_idx in (last_idx + 1):(eval_idx - 1)
                frac = span == 0.0 ? 0.0 : (x_eval[fill_idx] - x_left) / span
                output[fill_idx] = (1.0 - frac) * y_left + frac * y_right
            end
        end

        last_idx = eval_idx
    end

    if last_idx != n_eval
        output[n_eval] = loess_estimate(
            x, y, x_eval[n_eval], bandwidth, degree;
            robustness_weights = robustness_weights,
        )

        x_left = x_eval[last_idx]
        x_right = x_eval[n_eval]
        y_left = output[last_idx]
        y_right = output[n_eval]
        span = x_right - x_left
        for fill_idx in (last_idx + 1):(n_eval - 1)
            frac = span == 0.0 ? 0.0 : (x_eval[fill_idx] - x_left) / span
            output[fill_idx] = (1.0 - frac) * y_left + frac * y_right
        end
    end

    return
end

function loess_smooth_regular!(
    y::AbstractVector{Float64},
    x_start::Float64,
    x_step::Float64,
    x_eval_start::Float64,
    x_eval_step::Float64,
    bandwidth::Int,
    degree::Int,
    jump::Int,
    robustness_weights::Union{Nothing,AbstractVector{Float64}},
    output::AbstractVector{Float64},
)
    n = length(y)
    n_eval = length(output)
    n_eval == 0 && return

    q = clamp(Int(bandwidth), 1, n)
    half = (q - 1) >>> 1
    max_left = n - q + 1
    step = max(1, Int(jump))
    last_idx = 0

    @inbounds for eval_idx in 1:step:n_eval
        x0 = x_eval_start + (eval_idx - 1) * x_eval_step
        center = round(Int, (x0 - x_start) / x_step) + 1
        center = clamp(center, 1, n)
        left = clamp(center - half, 1, max_left)
        right = left + q - 1
        output[eval_idx] = _loess_estimate_regular(
            y, x_start, x_step, x0, left, right, degree, robustness_weights,
        )

        if last_idx > 0
            inv_gap = 1.0 / (eval_idx - last_idx)
            y_left = output[last_idx]
            y_right = output[eval_idx]
            for fill_idx in (last_idx + 1):(eval_idx - 1)
                frac = (fill_idx - last_idx) * inv_gap
                output[fill_idx] = (1.0 - frac) * y_left + frac * y_right
            end
        end

        last_idx = eval_idx
    end

    if last_idx != n_eval
        x0 = x_eval_start + (n_eval - 1) * x_eval_step
        center = round(Int, (x0 - x_start) / x_step) + 1
        center = clamp(center, 1, n)
        left = clamp(center - half, 1, max_left)
        right = left + q - 1
        output[n_eval] = _loess_estimate_regular(
            y, x_start, x_step, x0, left, right, degree, robustness_weights,
        )

        inv_gap = 1.0 / (n_eval - last_idx)
        y_left = output[last_idx]
        y_right = output[n_eval]
        @inbounds for fill_idx in (last_idx + 1):(n_eval - 1)
            frac = (fill_idx - last_idx) * inv_gap
            output[fill_idx] = (1.0 - frac) * y_left + frac * y_right
        end
    end

    return
end

# ── STL types and decomposition ─────────────────────────────────────────────

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
    periodic::Bool
end

mutable struct STLWorkspace
    x_full::Vector{Float64}
    detrended::Vector{Float64}
    deseasonalized::Vector{Float64}
    seasonal_raw::Vector{Float64}
    lowpass::Vector{Float64}
    fit::Vector{Float64}
    residuals::Vector{Float64}
    ma1::Vector{Float64}
    ma2::Vector{Float64}
    ma3::Vector{Float64}
    sub_x::Vector{Float64}
    sub_y::Vector{Float64}
    sub_weights::Vector{Float64}
    sub_fit::Vector{Float64}

    function STLWorkspace(n::Int, period::Int)
        max_subseries_len = cld(n, period)
        new(collect(1.0:1.0:n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
            zeros(n - period + 1), zeros(n - 2 * period + 2), zeros(n - 2 * period),
            zeros(max_subseries_len), zeros(max_subseries_len), ones(max_subseries_len), zeros(max_subseries_len))
    end
end

function moving_average!(x::AbstractVector{Float64}, n::Int, window::Int, ave::AbstractVector{Float64})

    if window <= 0 || n < window
        return
    end
    output_len = n - window + 1
    window_f = Float64(window)

    running_sum = 0.0
    @inbounds for i in 1:window
        running_sum += x[i]
    end
    ave[1] = running_sum / window_f
    if output_len > 1
        head = window
        tail = 0
        @inbounds for j in 2:output_len
            head += 1
            tail += 1
            running_sum = running_sum - x[tail] + x[head]
            ave[j] = running_sum / window_f
        end
    end
    return
end

function lowpass_filter!(
    x::AbstractVector{Float64},
    period::Int,
    lowpass_bandwidth::Int,
    lowpass_degree::Int,
    lowpass_jump::Int,
    ws::STLWorkspace,
)
    n = length(x)
    moving_average!(x, n, period, ws.ma1)
    moving_average!(ws.ma1, length(ws.ma1), period, ws.ma2)
    moving_average!(ws.ma2, length(ws.ma2), 3, ws.ma3)

    loess_smooth_regular!(
        ws.ma3,
        period + 1.0,
        1.0,
        1.0,
        1.0,
        lowpass_bandwidth,
        lowpass_degree,
        lowpass_jump,
        nothing,
        ws.lowpass,
    )
    return
end

function seasonal_smooth!(
    y::AbstractVector{Float64},
    period::Int,
    seasonal_bandwidth::Int,
    seasonal_degree::Int,
    seasonal_jump::Int,
    use_weights::Bool,
    robustness_weights::AbstractVector{Float64},
    periodic::Bool,
    ws::STLWorkspace,
)
    n = length(y)

    for phase in 1:period
        count = 0
        @inbounds for idx in phase:period:n
            count += 1
            ws.sub_y[count] = y[idx]
            if use_weights
                ws.sub_weights[count] = robustness_weights[idx]
            end
        end

        if periodic
            total = 0.0
            total_weight = 0.0
            if use_weights
                @inbounds for i in 1:count
                    weight = ws.sub_weights[i]
                    total += weight * ws.sub_y[i]
                    total_weight += weight
                end
            else
                @inbounds for i in 1:count
                    total += ws.sub_y[i]
                end
                total_weight = count
            end
            seasonal_mean = total_weight > 0.0 ? total / total_weight : _mean_range(ws.sub_y, 1, count)

            @inbounds for idx in phase:period:n
                ws.seasonal_raw[idx] = seasonal_mean
            end
            continue
        end

        y_sub = view(ws.sub_y, 1:count)
        sub_fit = view(ws.sub_fit, 1:count)
        sub_ρ = use_weights ? view(ws.sub_weights, 1:count) : nothing

        loess_smooth_regular!(
            y_sub,
            float(phase),
            float(period),
            float(phase),
            float(period),
            seasonal_bandwidth,
            seasonal_degree,
            seasonal_jump,
            sub_ρ,
            sub_fit,
        )

        assign_idx = 0
        @inbounds for idx in phase:period:n
            assign_idx += 1
            ws.seasonal_raw[idx] = sub_fit[assign_idx]
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

    scale = 6.0 * median(view(residuals, 1:n))
    if scale <= eps(Float64)
        fill!(weights, 1.0)
        return
    end

    for i in 1:n
        weights[i] = _bisquare(residuals[i] / scale)
    end
    return
end

function stl_inner_loop!(ws::STLWorkspace, y::AbstractVector{Float64}, n::Int,
                         season::AbstractVector{Float64}, trend::AbstractVector{Float64},
                         use_weights::Bool, robustness_weights::AbstractVector{Float64},
                         params::STLParams)
    for _iter in 1:params.n_inner
        @inbounds for i in 1:n
            ws.detrended[i] = y[i] - trend[i]
        end

        seasonal_smooth!(
            ws.detrended,
            params.period,
            params.seasonal_bandwidth,
            params.seasonal_degree,
            params.seasonal_jump,
            use_weights,
            robustness_weights,
            params.periodic,
            ws,
        )
        lowpass_filter!(
            ws.seasonal_raw,
            params.period,
            params.lowpass_bandwidth,
            params.lowpass_degree,
            params.lowpass_jump,
            ws,
        )

        @inbounds for i in 1:n
            season[i] = ws.seasonal_raw[i] - ws.lowpass[i]
            ws.deseasonalized[i] = y[i] - season[i]
        end

        loess_smooth_regular!(
            ws.deseasonalized,
            1.0,
            1.0,
            1.0,
            1.0,
            params.trend_bandwidth,
            params.trend_degree,
            params.trend_jump,
            use_weights ? robustness_weights : nothing,
            trend,
        )
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
        params.periodic,
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
        periodic,
    )
    season, trend, weights = stl_decompose(xvec, params)
    remainder = xvec .- season .- trend
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

# ── STL display methods ─────────────────────────────────────────────────────

function Base.show(io::IO, result::STLResult)
    println(io, "STL decomposition")
    println(io, "Seasonal component (first 10 values): ", result.seasonal[1:min(end,10)])
    println(io, "Trend component    (first 10 values): ", result.trend[1:min(end,10)])
    println(io, "Remainder          (first 10 values): ", result.remainder[1:min(end,10)])
    println(io, "Windows: seasonal=", result.seasonal_window, ", trend=", result.trend_window, ", lowpass=", result.lowpass_window)
    println(io, "Degrees: seasonal=", result.seasonal_degree, ", trend=", result.trend_degree, ", lowpass=", result.lowpass_degree)
    println(io, "Jumps: seasonal=", result.seasonal_jump, ", trend=", result.trend_jump, ", lowpass=", result.lowpass_jump)
    println(io, "Inner iterations: ", result.inner_iterations, ", Outer iterations: ", result.outer_iterations)
    return
end

function summary(result::STLResult; digits::Integer=4)
    n = length(result.seasonal)
    data = result.seasonal .+ result.trend .+ result.remainder
    comps = Dict(
        :seasonal => result.seasonal,
        :trend    => result.trend,
        :remainder=> result.remainder,
        :data     => data,
    )

    function iqr(v::AbstractVector)
        q25, q75 = quantile(v, [0.25, 0.75])
        return q75 - q25
    end
    println("STL decomposition summary")
    println("Time series components:")
    for (name, vec) in comps
        println("  ", name)
        mv = mean(vec)
        sv = std(vec)
        mn = minimum(vec)
        mx = maximum(vec)
        iqr_v = iqr(vec)

        component_fmt = string("% .", digits, "f")
        full_fmt = "    mean=" * component_fmt * "  sd=" * component_fmt *
                   "  min=" * component_fmt * "  max=" * component_fmt *
                   "  IQR=" * component_fmt
        f = Printf.Format(full_fmt)
        println(Printf.format(f, mv, sv, mn, mx, iqr_v))
    end
    println("IQR as percentage of total:")
    iqr_vals = Dict(name => iqr(vec) for (name, vec) in comps)
    total_iqr = iqr_vals[:data]
    for (name, v) in iqr_vals
        pct = total_iqr == 0 ? NaN : 100.0 * v / total_iqr
        pct_str = isnan(pct) ? "NaN" : string(round(pct; digits=1))
        println("  ", Symbol(name), ": ", pct_str, "%")
    end

    if all(w -> w == 1.0, result.weights)
        println("Weights: all equal to 1")
    else
        w = result.weights
        mv = mean(w)
        sv = std(w)
        mn = minimum(w)
        mx = maximum(w)
        iqr_w = iqr(w)
        println("Weights summary:")
        s_mean = string(round(mv; digits=digits))
        s_sd   = string(round(sv; digits=digits))
        s_min  = string(round(mn; digits=digits))
        s_max  = string(round(mx; digits=digits))
        s_iqr  = string(round(iqr_w; digits=digits))
        println("  mean=", s_mean, "  sd=", s_sd,
                "  min=", s_min, "  max=", s_max, "  IQR=", s_iqr)
    end
    println("Other components: seasonal_window=", result.seasonal_window,
            ", trend_window=", result.trend_window,
            ", lowpass_window=", result.lowpass_window,
            ", inner=", result.inner_iterations,
            ", outer=", result.outer_iterations)
    return nothing
end
