"""
    struct MSTLResult{T<:Real}

Result of a multi-seasonal STL decomposition (`mstl`).

# Fields
- `data::Vector{T}`: Original input series (on the **original scale**; i.e., before any Box-Cox transform).
- `trend::Vector{T}`: Trend component (taken from the **last STL fit**).
- `seasonals::Vector{Vector{T}}`: Seasonal components; one vector per seasonal
  period, ordered to match `periods`.
- `periods::Vector{Int}`: Seasonal periods actually used (after dropping periods
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
    w = max(3, min(n, Int(clamp(round(n ÷ 10), 5, 101))))
    k = (w - 1) ÷ 2
    cumsums = vcat(0.0, cumsum(Float64.(x)))
    y = similar(x, Float64)
    @inbounds for t in 1:n
        a = max(1, t - k); b = min(n, t + k)
        y[t] = (cumsums[b+1] - cumsums[a]) / (b - a + 1)
    end
    return y
end

"""
    mstl(x::AbstractVector{T};
         periods::Union{Integer,AbstractVector{<:Integer}} = 1,
         lambda::Union{Nothing,Real}=nothing,
         iterate::Integer=2,
         s_window=nothing,
         stl_kwargs...) where {T<:Real}

Multiple seasonal decomposition.

Decomposes a univariate time series into **seasonal**, **trend**, and **remainder**
components. Seasonal components are estimated **iteratively** using `stl` with one
seasonal period at a time; **multiple seasonal periods are allowed**. The **trend**
component is taken from the **last STL fit** in the final iteration. If no valid
seasonal period remains, the series is decomposed into **trend + remainder** only
(the trend is estimated using a simple smoother).

Optionally, a **Box-Cox** transform can be applied to the series prior to
decomposition.

# Arguments
- `x`: Univariate time series (`AbstractVector{<:Real}`).
- `m`: A single seasonal period (Int) or a vector of periods. Periods with
  fewer than **two complete cycles** in `x` are **dropped** automatically.
- `lambda`: Optional Box-Cox λ. If provided, the decomposition is performed on the
  transformed series. (`λ = 0` corresponds to log transform.)
- `iterate`: Number of **outer iterations** cycling over seasonal periods to refine
  seasonal estimates (default `2`).
- `s_window`: Seasonal LOESS window(s). If a scalar, the same value is used for all
  seasonal components. If a vector, it is recycled or trimmed to match the number
  of periods. When `nothing`, a default sequence similar to R
  (`11, 15, 19, 23, 27, 31`) is used and repeated as needed.
- `stl_kwargs...`: Additional keyword arguments forwarded to `stl` (e.g.
  `s_degree`, `t_window`, `t_degree`, `l_window`, `robust`, `inner`, `outer`, etc.).

# Details
- **Missing values**: any internal missing/NaN values are interpolated before
  decomposition.
- **Seasonal refinement**: for each iteration and each period `m`, the current
  estimate of that seasonal component is **added back**, STL is run with frequency
  `m`, the seasonal is updated, then **removed**—repeating for all periods.
- **Trend**: the final trend is copied from the **last** STL model fitted in the
  final iteration.
- **No seasonality**: if all candidate periods are dropped (or `periods == 1`),
  the trend is computed by a simple smoother (analogous to R’s `supsmu` fallback).

# Returns
An [`MSTLResult`](@ref) containing:
- `data`: original (untransformed) data,
- `trend`: trend component,
- `seasonals`: a vector of seasonal components (one per period, in ascending order),
- `periods`: the seasonal periods actually used,
- `remainder`: residual component,
- `lambda`: the Box-Cox λ used (or `nothing`).

# Examples
```julia
y = rand(200) .+ 2sin.(2π*(1:200)/7) .+ 0.5sin.(2π*(1:200)/30)
res = mstl(y; m=[7,30], iterate=2, s_window=[11,23], robust=true)
```

"""
function mstl(
    x::AbstractVector,
    m::Union{Integer,AbstractVector{<:Integer}};
    lambda::Union{Nothing,Real, String} = nothing,
    iterate::Integer = 2,
    s_window = nothing,
    stl_kwargs...,
)

    n = length(x)
    @assert n > 0 "x must be non-empty"

    orig = Vector{Float64}([ismissing(v) ? NaN : Float64(v) for v in x])
    xu   = copy(orig)

    if any(isnan, xu)
        interp_m = isa(m, Integer) ? Int(m) : Int(floor(maximum(m)))
        xu = na_interp(xu; m=interp_m)
    end

    λ = lambda
    if !isnothing(lambda)
        bc_m = isa(m, Integer) ? Int(m) : maximum(Int.(m))
        xu, λ = box_cox(xu, bc_m; lambda = λ)
    end

    pers = isa(m, Integer) ? [Int(m)] : sort(collect(Int.(m)))
    pers = [p for p in pers if p > 1 && 2 * p < n]
    
    if isempty(pers)
        trend = smooth_trend(xu)
        rem   = xu .- trend
        return MSTLResult{Float64}(
            orig, trend, Vector{Vector{Float64}}(), Int[], rem, λ
        )
    end

    default_windows = collect(11:4:31)
    swin = if isnothing(s_window)
        [default_windows[mod1(i, length(default_windows))] for i in 1:length(pers)]
    elseif isa(s_window, Integer)
        fill(Int(s_window), length(pers))
    else
        v = collect(Int.(s_window))
        @assert !isempty(v) "s_window cannot be empty."
        [v[mod1(i, length(v))] for i in 1:length(pers)]
    end

    seas = [zeros(Float64, n) for _ in pers]
    deseas = copy(xu)
    iters = max(1, Int(iterate))

    last_fit = nothing
    for _ in 1:iters
        for (idx, m) in pairs(pers)
            deseas .+= seas[idx]
            fit = stl(deseas, m; s_window = swin[idx], stl_kwargs...)
            s   = fit.time_series.seasonal
            seas[idx] = collect(Float64.(s))
            deseas .-= seas[idx]
            last_fit = fit
        end
    end

    trend = collect(Float64.(last_fit.time_series.trend))
    remainder = xu .- trend
    for s in seas
        remainder .-= s
    end

    return MSTLResult{Float64}(
        orig, trend, seas, pers, remainder, λ)
end

"""
    Base.show(io::IO, res::MSTLResult)

Pretty print an `MSTLResult`. Shows first values for each component and
basic metadata such as seasonal periods and optional Box-Cox λ.
"""
function Base.show(io::IO, res::MSTLResult)
    n = length(res.data)
    k = min(n, 10)
    println(io, "MSTL decomposition")
    println(io, "  length: ", n)
    if isempty(res.m)
        println(io, "  periods: (none)")
    else
        println(io, "  periods: ", res.m)
    end
    println(io, "  lambda: ", isnothing(res.lambda) ? "nothing" : string(res.lambda))

    println(io, "Trend     (first $k): ", res.trend[1:k])
    if !isempty(res.seasonals)
        for (i, p) in enumerate(res.m)
            s = res.seasonals[i]
            println(io, "Seasonal($p) (first $k): ", s[1:k])
        end
    else
        println(io, "Seasonal: (none)")
    end
    println(io, "Remainder (first $k): ", res.remainder[1:k])
    return
end


"""
    summary(res::MSTLResult; digits=4)

Display descriptive statistics for an `MSTLResult`.

Reports mean, sd, min, max, IQR for:
  - data (reconstruction),
  - each seasonal component,
  - total seasonal (sum of seasonals),
  - trend,
  - remainder.

Also reports each component's IQR as a percentage of the data IQR.
"""
function summary(res::MSTLResult; digits::Integer=4)
    S = isempty(res.seasonals) ? zeros(eltype(res.data), length(res.data)) :
                                 reduce(+, res.seasonals)
    data = res.trend .+ S .+ res.remainder

    comps = Dict{Symbol,AbstractVector{<:Real}}(
        :data      => data,
        :trend     => res.trend,
        :remainder => res.remainder,
    )

    if !isempty(res.seasonals)
        comps[:seasonal_total] = S
        for (i, p) in enumerate(res.m)
            comps[Symbol("seasonal_$p")] = res.seasonals[i]
        end
    end

    iqr(v) = begin
        q25, q75 = quantile(v, (0.25, 0.75))
        q75 - q25
    end

    println("MSTL decomposition summary")
    println("Components (mean, sd, min, max, IQR):")
    fmt(x) = isnan(x) ? "NaN" : string(round(x; digits=digits))

    for (name, vec) in sort(collect(comps); by=first)
        mv, sv = mean(vec), std(vec)
        mn, mx = minimum(vec), maximum(vec)
        iq = iqr(vec)
        println("  ", rpad(string(name), 16), " ",
                "mean=", fmt(mv), "  sd=", fmt(sv),
                "  min=", fmt(mn), "  max=", fmt(mx),
                "  IQR=", fmt(iq))
    end

    println("IQR as % of data IQR:")
    data_iqr = iqr(comps[:data])
    for (name, vec) in sort(collect(comps); by=first)
        iq = iqr(vec)
        pct = iszero(data_iqr) ? NaN : 100 * iq / data_iqr
        println("  ", rpad(string(name), 16), " ", fmt(pct), "%")
    end

    println("Metadata: periods=", isempty(res.m) ? "[]" : string(res.m),
            ", lambda=", isnothing(res.lambda) ? "nothing" : string(res.lambda))
    return nothing
end

"""
    plot(res::MSTLResult; labels=nothing, col_range="lightgray", main=nothing, range_bars=true, kwargs...)

Multi-panel plot for `MSTLResult` using Plots.jl.

Requires loading Plots.jl first:
```julia
using Plots
plot(mstl_result)
```

Panels (from top):
  1. Data (reconstructed)
  2..(1+S) Seasonal(p) for each period p in `res.m`
  next: Trend
  last: Remainder

Keyword arguments are forwarded to `Plots.plot!`.
"""
