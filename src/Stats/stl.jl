"""
    loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound,
                    w, use_weights, robustness_weights) -> (ŷ, ok)

Compute a single locally weighted polynomial regression estimate at `eval_point`.

This implements the core LOESS (LOcally Estimated Scatterplot Smoothing) kernel
evaluation from Cleveland (1979). For each evaluation point, it:

1. **Computes tricube weights** for observations in `[left_bound, right_bound]`:

       u = |j - eval_point| / half_width
       w(j) = (1 - u³)³    if u < 1
              0             otherwise

   where `half_width = max(eval_point - left_bound, right_bound - eval_point)`.

2. **Applies robustness weights** (if `use_weights`): `w(j) *= robustness_weights[j]`

3. **Normalizes weights**: `w(j) /= Σ w(j)`

4. **Fits local polynomial** of the specified `degree` (0 = constant, 1 = linear):
   - Degree 0: `ŷ = Σ w(j) · y(j)` (weighted mean)
   - Degree 1: `ŷ = Σ w(j) · y(j)` after adjusting weights for the linear term:

         x̄_w = Σ w(j) · j                    (weighted mean of positions)
         β = (eval_point - x̄_w) / Σ w(j)·(j - x̄_w)²
         w(j) ← w(j) · (1 + β · (j - x̄_w))

# Returns
- `(ŷ::Float64, true)` on success
- `(0.0, false)` if all weights are zero (no data in window)

# References
- Cleveland, W. S. (1979). *Robust Locally Weighted Regression and Smoothing
  Scatterplots.* JASA, 74(368), 829–836.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990).
  *STL: A Seasonal-Trend Decomposition Procedure Based on Loess.*
  Journal of Official Statistics, 6, 3–73.
"""
function loess_estimate!(y::AbstractVector{Float64}, n::Int, bandwidth::Int, degree::Int,
                         eval_point::Float64, left_bound::Int, right_bound::Int,
                         w::AbstractVector{Float64}, use_weights::Bool, robustness_weights::AbstractVector{Float64})

    data_range = float(n) - 1.0
    half_width = max(eval_point - float(left_bound), float(right_bound) - eval_point)
    if bandwidth > n
        half_width += float((bandwidth - n) ÷ 2)
    end
    upper_threshold = 0.999 * half_width
    lower_threshold = 0.001 * half_width

    weight_sum = 0.0
    for j in left_bound:right_bound
        dist = abs(float(j) - eval_point)
        if dist <= upper_threshold
            if dist <= lower_threshold || half_width == 0.0
                w[j] = 1.0
            else
                normalized_dist = dist / half_width
                w[j] = (1.0 - normalized_dist^3)^3
            end
            if use_weights
                w[j] *= robustness_weights[j]
            end
            weight_sum += w[j]
        else
            w[j] = 0.0
        end
    end

    if weight_sum <= 0.0
        return 0.0, false
    end

    inv_weight_sum = 1.0 / weight_sum
    for j in left_bound:right_bound
        w[j] *= inv_weight_sum
    end

    if half_width > 0.0 && degree > 0

        weighted_mean_x = 0.0
        for j in left_bound:right_bound
            weighted_mean_x += w[j] * float(j)
        end
        slope_num = eval_point - weighted_mean_x
        slope_denom = 0.0
        for j in left_bound:right_bound
            dev = float(j) - weighted_mean_x
            slope_denom += w[j] * dev^2
        end

        if sqrt(slope_denom) > 0.001 * data_range
            slope_num /= slope_denom
            for j in left_bound:right_bound
                w[j] = w[j] * (slope_num * (float(j) - weighted_mean_x) + 1.0)
            end
        end
    end

    ys = 0.0
    for j in left_bound:right_bound
        ys += w[j] * y[j]
    end
    return ys, true
end

"""
    loess_smooth!(y, n, bandwidth, degree, jump, use_weights, robustness_weights, ys, res)

Apply LOESS smoothing to the full series `y` of length `n`, writing results into `ys`.

Evaluates [`loess_estimate!`](@ref) at every `jump`-th point, then linearly
interpolates between evaluated points to fill the gaps. The `res` buffer is
used as scratch space for weight computation.
"""
function loess_smooth!(y::AbstractVector{Float64}, n::Int, bandwidth::Int, degree::Int, jump::Int,
                       use_weights::Bool, robustness_weights::AbstractVector{Float64},
                       ys::AbstractVector{Float64}, res::AbstractVector{Float64})

    if n < 2

        ys[firstindex(ys)] = y[1]
        return
    end

    step_size = min(jump, n - 1)

    left_bound = 1
    right_bound = min(bandwidth, n)

    if bandwidth >= n
        left_bound = 1
        right_bound = n
        i = 1
        while i <= n
            eval_point = float(i)
            ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
            if ok
                ys[firstindex(ys) - 1 + i] = ysi
            else
                ys[firstindex(ys) - 1 + i] = y[i]
            end
            i += step_size
        end
    else
        if step_size == 1
            half_bandwidth = (bandwidth + 1) ÷ 2
            left_bound = 1
            right_bound = bandwidth
            for i in 1:n
                if (i > half_bandwidth) && (right_bound != n)
                    left_bound += 1
                    right_bound += 1
                end
                eval_point = float(i)
                ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
                if ok
                    ys[firstindex(ys) - 1 + i] = ysi
                else
                    ys[firstindex(ys) - 1 + i] = y[i]
                end
            end
        else
            half_bandwidth = (bandwidth + 1) ÷ 2
            i = 1
            while i <= n
                if i < half_bandwidth
                    left_bound = 1
                    right_bound = bandwidth
                elseif i >= n - half_bandwidth + 1
                    left_bound = n - bandwidth + 1
                    right_bound = n
                else
                    left_bound = i - half_bandwidth + 1
                    right_bound = bandwidth + i - half_bandwidth
                end
                eval_point = float(i)
                ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
                if ok
                    ys[firstindex(ys) - 1 + i] = ysi
                else
                    ys[firstindex(ys) - 1 + i] = y[i]
                end
                i += step_size
            end
        end
    end

    if step_size != 1
        i = 1
        while i <= n - step_size
            ysi = ys[firstindex(ys) - 1 + i]
            ysj = ys[firstindex(ys) - 1 + i + step_size]
            interp_slope = (ysj - ysi) / float(step_size)
            for j in (i + 1):(i + step_size - 1)
                ys[firstindex(ys) - 1 + j] = ysi + interp_slope * float(j - i)
            end
            i += step_size
        end

        k = ((n - 1) ÷ step_size) * step_size + 1
        if k != n

            eval_point = float(n)
            ysn, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
            if ok
                ys[firstindex(ys) - 1 + n] = ysn
            else
                ys[firstindex(ys) - 1 + n] = y[n]
            end
            if k != n - 1

                val_at_k = ys[firstindex(ys) - 1 + k]
                val_at_n = ys[firstindex(ys) - 1 + n]
                interp_slope = (val_at_n - val_at_k) / float(n - k)
                for j in (k + 1):(n - 1)
                    ys[firstindex(ys) - 1 + j] = val_at_k + interp_slope * float(j - k)
                end
            end
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
