"""
    struct DecomposedTimeSeries

A structure representing a decomposed time series, which includes the original time series, seasonal, trend, random components, and associated metadata.

# Fields
- `x::AbstractVector`: The original time series data.
- `seasonal::AbstractVector`: The seasonal component of the time series.
- `trend::AbstractVector`: The trend component of the time series.
- `random::AbstractVector`: The random or residual component of the time series.
- `figure::AbstractVector`: The estimated seasonal figure only.
- `type::String`: A string indicating the type of decomposition or any other relevant type information.
- `m::Int`: The frequency of the x.

# Example
```julia
# Creating an example decomposed time series
x = [1.0, 2.0, 3.0, 4.0, 5.0]
seasonal = [0.1, 0.2, 0.1, 0.2, 0.1]
trend = [0.5, 1.0, 1.5, 2.0, 2.5]
random = [0.4, 0.8, 1.4, 1.8, 2.4]
figure = []  # This can be filled with figures related to the decomposition
type = "Additive"
m = 2
```
"""
struct DecomposedTimeSeries
    x::AbstractVector
    seasonal::AbstractVector
    trend::AbstractVector
    random::AbstractVector
    figure::AbstractVector
    type::String
    m::Int
end

isbad(v) = (ismissing(v) || (v isa AbstractFloat && isnan(v)))

mean_skip(a) = begin
    if isempty(a)
        NaN
    else
        if eltype(a) <: Union{Missing,Number}
            xs = collect(skipmissing(a))
            xs = xs[.!isnan.(Float64.(xs))]
            isempty(xs) ? NaN : mean(Float64.(xs))
        else
            xs = Float64.(a)
            xs = xs[.!isnan.(xs)]
            isempty(xs) ? NaN : mean(xs)
        end
    end
end


function filter2sided_non_circular(
    x::AbstractVector{<:Number},
    w::AbstractVector{<:Number},
)::Vector{Float64}
    nx = length(x)
    nf = length(w)
    out = fill(NaN, nx)
    shift = nf ÷ 2

    xf = Float64.(x)
    wf = Float64.(w)

    for i = 1:nx
        i1 = i + shift - (nf - 1)
        i2 = i + shift
        if i1 < 1 || i2 > nx
            out[i] = NaN
            continue
        end

        s = 0.0
        valid = true
        @inbounds for j = 1:nf
            idx = i + shift - j + 1
            xv = xf[idx]
            if isbad(xv)
                valid = false
                break
            end
            s += wf[j] * xv
        end
        out[i] = valid ? s : NaN
    end
    return out
end

function filter2sided_circular(
    x::AbstractVector{<:Number},
    w::AbstractVector{<:Number},
)::Vector{Float64}
    nx = length(x)
    nf = length(w)
    out = fill(NaN, nx)
    shift = nf ÷ 2
    xf = Float64.(x)
    wf = Float64.(w)

    for i = 1:nx
        s = 0.0
        valid = true
        @inbounds for j = 1:nf
            idx = i + shift - j + 1
            while idx < 1
                idx += nx
            end
            while idx > nx
                idx -= nx
            end
            xv = xf[idx]
            if isbad(xv)
                valid = false
                break
            end
            s += wf[j] * xv
        end
        out[i] = valid ? s : NaN
    end
    return out
end


"""
 decompose(;x::Vector, m::Int, type::String, filter)
 
 Classical Seasonal Decomposition by Moving Averages

 Decompose a vector of time series into seasonal, trend and irregular components using moving averages. 
 Deals with additive or multiplicative seasonal component.
 
 Return An object of class "DecomposedTimeSeries" with following components:
 x: The original series.
 seasonal: The seasonal component (i.e., the repeated seasonal figure).
 trend: The trend component.
 random: The remainder part.
 figure: The estimated seasonal figure only.
 type: The value of type.
 
 # Arguments
 - `x::AbstractVector`: A AbstractVector of one time series.
 - `m::Int`: The frequency of the time serie
 - `type::String`: The type of seasonal component. Can be either additive or multiplicative.
 - `filter`: A AbstractVector of filter coefficients in reverse time order 
 (as for AR or MA coefficients), used for filtering out the seasonal component.
  If NULL, a moving average with symmetric window is performed.
 
 # Examples
 ```julia-repl
 julia> ap = air_passengers();
 julia> decompose(x = ap, m = 12, type= "multiplicative", filter = NaN)
 julia> decompose(x = ap, m = 12, type= "additive", filter = NaN)
 
 ```
 """
function decompose(;
    x::AbstractVector,
    m::Int,
    type::String = "additive",
    filter::Union{Nothing,AbstractVector} = nothing,
)

    n = length(x)
    
    if m <= 1 || length([v for v in x if !isbad(v)]) < 2 * m
        error("time series has no or less than 2 periods")
    end
    t = lowercase(type)
    if t != "additive" && t != "multiplicative"
        error("type must be \"additive\" or \"multiplicative\"")
    end

    w = if filter === nothing
        (m % 2 == 0) ? vcat([0.5], ones(m - 1), [0.5]) ./ m : ones(m) ./ m
    else
        Float64.(filter)
    end

    trend = filter2sided_non_circular(x, w)

    xf = Float64.(x)
    season_pre = similar(trend)
    if t == "additive"
        @inbounds for i = 1:n
            xv = xf[i]
            tv = trend[i]
            season_pre[i] = (isbad(xv) || isbad(tv)) ? NaN : (xv - tv)
        end
    else
        @inbounds for i = 1:n
            xv = xf[i]
            tv = trend[i]
            season_pre[i] = (isbad(xv) || isbad(tv) || tv == 0.0) ? NaN : (xv / tv)
        end
    end

    figure = fill(NaN, m)
    for i = 1:m
        vals = season_pre[i:m:n]
        μ = mean_skip(vals)
        figure[i] = μ
    end
 
    μfig = mean_skip(figure)
    if t == "additive"
        figure .= figure .- μfig
    else
        figure .= figure ./ μfig
    end

    rep = ceil(Int, n / m)
    seasonal = repeat(figure, rep)[1:n]

    random = similar(trend)
    if t == "additive"
        @inbounds for i = 1:n
            xv = xf[i]
            sv = seasonal[i]
            tv = trend[i]
            random[i] = (isbad(xv) || isbad(sv) || isbad(tv)) ? NaN : (xv - sv - tv)
        end
    else
        @inbounds for i = 1:n
            xv = xf[i]
            sv = seasonal[i]
            tv = trend[i]
            random[i] =
                (isbad(xv) || isbad(sv) || isbad(tv) || sv == 0.0 || tv == 0.0) ? NaN :
                (xv / (sv * tv))
        end
    end

    out = DecomposedTimeSeries(x, seasonal, trend, random, figure, type, m)
    return (out)
end
