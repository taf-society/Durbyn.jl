"""
    struct MSTLResult{T<:Real}

Result of a multi-seasonal STL decomposition (`mstl`).

# Fields
- `data::Vector{T}`: Original input series (on the **original scale**; i.e., before any Box-Cox transform).
- `trend::Vector{T}`: Trend component (taken from the **last STL fit**).
- `seasonals::Vector{Vector{T}}`: Seasonal components; one vector per seasonal
  period, ordered to match `m`.
- `m::Vector{Int}`: Seasonal periods actually used (after dropping periods
  with < 2 full cycles).
- `remainder::Vector{T}`: Remainder (`data - trend - sum(seasonals)` on the
  transformed scale if `lambda` was given, then reported on the same scale as `data` if
  no inverse transform is applied; align with your implementation choice).
- `lambda::Union{Nothing,Float64}`: Box-Cox λ used for the decomposition, or `nothing`.

# Notes
- When `lambda` is provided, components are estimated on the transformed series.
  Whether components in this struct are returned on the transformed or original
  scale depends on the chosen implementation; document or convert accordingly.
- The **sum of all components** (trend + total seasonal + remainder) reconstructs
  the transformed series used in the STL fits.

"""
struct MSTLResult{T<:Real}
    data::Vector{T}
    trend::Vector{T}
    seasonals::Vector{Vector{T}}
    m::Vector{Int}
    remainder::Vector{T}
    lambda::Union{Nothing,Float64}
end

function smooth_trend(x::AbstractVector{<:Real})
    n = length(x)
    window_width = max(3, min(n, Int(clamp(round(n ÷ 10), 5, 101))))
    half_window = (window_width - 1) ÷ 2
    cumulative_sums = vcat(0.0, cumsum(Float64.(x)))
    smoothed = similar(x, Float64)
    @inbounds for t in 1:n
        left = max(1, t - half_window)
        right = min(n, t + half_window)
        smoothed[t] = (cumulative_sums[right + 1] - cumulative_sums[left]) / (right - left + 1)
    end
    return smoothed
end

"""
    mstl(x, m; kwargs...)

Multiple seasonal decomposition by iterative STL.

Decomposes a univariate time series into **seasonal**, **trend**, and **remainder**
components. Seasonal components are estimated **iteratively** using [`stl`](@ref)
with one seasonal period at a time; **multiple seasonal periods are allowed**.
The **trend** component is taken from the **last STL fit** in the final iteration.
If no valid seasonal period remains, the series is decomposed into
**trend + remainder** only (the trend is estimated using a simple smoother).

Optionally, a **Box-Cox** transform can be applied to the series prior to
decomposition.

# References

R. J. Hyndman and G. Athanasopoulos (2021)
*Forecasting: Principles and Practice* (3rd ed).
OTexts, Melbourne. <https://otexts.com/fpp3/>

# Arguments
- `x`: Univariate time series (`AbstractVector{<:Real}`).
- `m`: A single seasonal period (`Int`) or a vector of periods.  Periods with
  fewer than **two complete cycles** in `x` are **dropped** automatically.
- `lambda`: Optional Box-Cox λ.  If provided, the decomposition is performed on
  the transformed series.  (`λ = 0` corresponds to log transform.)
- `iterate`: Number of **outer iterations** cycling over seasonal periods to
  refine seasonal estimates (default `2`).
- `seasonal_window`: Seasonal LOESS window(s).  If a scalar, the same value is
  used for all seasonal components.  If a vector, it is recycled or trimmed to
  match the number of periods.  When `nothing`, a default sequence of odd windows
  (`11, 15, 19, 23, 27, 31`) is used and repeated as needed.
- `stl_kwargs...`: Additional keyword arguments forwarded to [`stl`](@ref)
  (e.g. `seasonal_degree`, `trend_window`, `trend_degree`, `lowpass_window`,
  `robust`, `inner`, `outer`, etc.).

# Details
- **Missing values**: any internal missing/NaN values are interpolated before
  decomposition.
- **Seasonal refinement**: for each iteration and each period, the current
  estimate of that seasonal component is **added back**, STL is run with the
  given frequency, the seasonal is updated, then **removed** — repeating for
  all periods.
- **Trend**: the final trend is copied from the **last** STL model fitted in
  the final iteration.
- **No seasonality**: if all candidate periods are dropped (or `m == 1`),
  the trend is computed by a simple smoother (analogous to R's `supsmu` fallback).

# Returns
An [`MSTLResult`](@ref) containing:
- `data`: original (untransformed) data,
- `trend`: trend component,
- `seasonals`: a vector of seasonal components (one per period, in ascending order),
- `m`: the seasonal periods actually used,
- `remainder`: residual component,
- `lambda`: the Box-Cox λ used (or `nothing`).

# Examples
```julia
y = rand(200) .+ 2sin.(2π*(1:200)/7) .+ 0.5sin.(2π*(1:200)/30)
res = mstl(y, [7, 30]; iterate=2, seasonal_window=[11, 23], robust=true)
```

"""
function mstl(
    x::AbstractVector,
    m::Union{Integer,AbstractVector{<:Integer}};
    lambda::Union{Nothing,Real, Symbol} = nothing,
    iterate::Integer = 2,
    seasonal_window = nothing,
    stl_kwargs...,
)

    n = length(x)
    n > 0 || throw(ArgumentError("x must be non-empty"))

    original = Vector{Float64}([ismissing(v) ? NaN : Float64(v) for v in x])
    transformed = copy(original)

    if any(isnan, transformed)
        max_period = isa(m, Integer) ? Int(m) : Int(floor(maximum(m)))
        transformed = interpolate_missing(transformed; m=max_period)
    end

    λ = lambda
    if !isnothing(lambda)
        max_period = isa(m, Integer) ? Int(m) : maximum(Int.(m))
        transformed, λ = box_cox(transformed, max_period; lambda = λ)
    end

    periods = isa(m, Integer) ? [Int(m)] : sort(collect(Int.(m)))
    periods = [p for p in periods if p > 1 && 2 * p < n]

    if isempty(periods)
        trend = smooth_trend(transformed)
        remainder = transformed .- trend
        return MSTLResult{Float64}(
            original, trend, Vector{Vector{Float64}}(), Int[], remainder, λ
        )
    end

    default_windows = collect(11:4:31)
    seasonal_windows = if isnothing(seasonal_window)
        [default_windows[mod1(i, length(default_windows))] for i in 1:length(periods)]
    elseif isa(seasonal_window, Integer)
        fill(Int(seasonal_window), length(periods))
    else
        window_vec = collect(Int.(seasonal_window))
        !isempty(window_vec) || throw(ArgumentError("seasonal_window cannot be empty."))
        [window_vec[mod1(i, length(window_vec))] for i in 1:length(periods)]
    end

    seasonals = [zeros(Float64, n) for _ in periods]
    deseasonalized = copy(transformed)
    n_iterations = max(1, Int(iterate))

    last_stl_fit = nothing
    for _ in 1:n_iterations
        for (idx, period) in pairs(periods)
            deseasonalized .+= seasonals[idx]
            stl_fit = stl(deseasonalized, period; seasonal_window = seasonal_windows[idx], stl_kwargs...)
            seasonals[idx] = collect(Float64.(stl_fit.seasonal))
            deseasonalized .-= seasonals[idx]
            last_stl_fit = stl_fit
        end
    end

    trend = collect(Float64.(last_stl_fit.trend))
    remainder = transformed .- trend
    for seasonal_component in seasonals
        remainder .-= seasonal_component
    end

    return MSTLResult{Float64}(
        original, trend, seasonals, periods, remainder, λ)
end
