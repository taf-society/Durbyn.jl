"""
    struct DecomposedTimeSeries

A structure representing a decomposed time series, which includes the original time series, seasonal, trend, random components, and associated metadata.

# Fields
- `x::Vector`: The original time series data.
- `seasonal::Vector`: The seasonal component of the time series.
- `trend::Vector`: The trend component of the time series.
- `random::Vector`: The random or residual component of the time series.
- `figure::Vector`: The estimated seasonal figure only.
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
    x::Vector
    seasonal::Vector
    trend::Vector
    random::Vector
    figure::Vector
    type::String
    m::Int
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
 - `x::Vector`: A vector of one time series.
 - `m::Int`: The frequency of the time serie
 - `type::String`: The type of seasonal component. Can be either additive or multiplicative.
 - `filter`: A vector of filter coefficients in reverse time order 
 (as for AR or MA coefficients), used for filtering out the seasonal component.
  If NULL, a moving average with symmetric window is performed.
 
 # Examples
 ```julia-repl
 julia> ap = air_passengers();
 julia> decompose(x = ap, m = 12, type= "multiplicative", filter = NaN)
 julia> decompose(x = ap, m = 12, type= "additive", filter = NaN)
 
 ```
 """
function decompose(; x::Vector, m::Int, type::String,
    filter::Union{Nothing,AbstractVector{<:Number},AbstractMatrix{<:Number}}=nothing)
    len = length(x)
    if m <= 1 | length(na_omit(x)) < 2 * m
        @error "time series has no or less than 2 periods"
    end
    if isnothing(filter)
        if m % 2 == 0
            filter = append!([0.5], repeat([1], m - 1), [0.5]) ./ m
        else
            filter = repeat([1], m - 1) ./ m
        end
    end
    trend = cfilter_non_circular(x, filter, 2)
    if type == "additive"
        season = x .- trend
    else
        season = x ./ trend
    end
    periods = floor(len ./ m)
    index = range(start=1, stop=len, step=m) .- 1
    figure = zeros(m)
    for i in range(start=1, stop=m, step=1)
        figure[i] = mean2(season[index.+i], omit_na=true)
    end
    if type == "additive"
        figure = figure .- mean(figure)
    else
        figure = figure ./ mean(figure)
    end
    season = repeat(figure, as_integer(periods + 1))
    season = season[range(start=1, stop=len, step=1)]

    if type == "additive"
        random = x .- season .- trend
    else
        random = x ./ season ./ trend
    end
    out = DecomposedTimeSeries(x, season, trend, random, figure, type, m)
    return (out)
end

function cfilter_circular(x, filter, sides)
    nx = length(x)
    nf = length(filter)
    out = zeros(nx)
    if sides == 2
        nshift = nf / 2
    else
        nshift = 0
    end

    nshift = as_integer(nshift)

    for i in range(start=1, stop=nx, step=1)
        z = 0.0
        valid = true

        for j in range(start=1, stop=nf, step=1)
            ii = i + nshift - j

            if ii < 1
                ii += nx
            end

            if ii >= nx
                ii -= nx
            end
            ii = as_integer(ii)

            if ii < 1
                tmp = NaN
            else
                tmp = x[ii]
            end

            if isnan(tmp)
                z += filter[j] * tmp
            else
                out[i] = NaN
                valid = false
                break
            end

        end
        if valid
            out[i] = z
        end

    end
    return (out)
end

function cfilter_non_circular(x, filter, sides)
    nx = length(x)
    nf = length(filter)
    out = zeros(nx)
    if sides == 2
        nshift = nf / 2
    else
        nshift = 0
    end
    nshift = as_integer(ceil(nshift))

    for i in range(start=1, stop=nx, step=1)
        z = zero(1)
        valid = true

        if i + nshift - (nf - 1) < 0 | i + nshift - 2 >= nx
            out[i] = NaN
            continue
        end

        for j in range(start=max(1, nshift + i - nx), stop=min(nf, i + nshift + 1), step=1)

            tmp_indx = i + nshift - j

            if tmp_indx >= 1
                tmp = x[tmp_indx]
            else
                tmp = NaN
            end

            if !isnan(tmp)
                z += filter[j] * tmp
            else
                out[i] = NaN
                valid = false
                break
            end
        end

        if valid
            out[i] = z
        end
    end
    return (out)
end

function cfilter(x::Vector{}, filter::Vector{}, sides::Int, circular::Bool)
    nx = length(x)
    nf = length(filter)
    out = zeros(nx)
    if sides == 2
        nshift = nf / 2
    else
        nshift = 0
    end

    if !circular
        for i in range(start=1, stop=nx, step=1)
            z = 0
            if i + nshift - (nf - 1) < 0 | i + nshift >= nx
                out[i] = NaN
                continue
            end

            for j in range(start=max(0, nshift + i - nx), stop=min(nf, i + nshift + 1), step=1)
                tmp = x[i+nshift-j]
                if ismissing(tmp)
                    z += filter[j] * tmp
                else
                    out[i] = NaN
                end
            end
            out[i] = z
        end
    else
        for i in range(start=1, stop=nx, step=1)
            z = 0
            for j in range(start=1, stop=nf, step=1)
                ii = i + nshift - j
                if ii < 0
                    ii += nx
                end
                if ii >= nx
                    ii -= nx
                end
                tmp = x[ii]
                if ismissing(tmp)
                    z += filter[j] * tmp
                else
                    out[i] = NaN
                end
            end
            out[i] = z
        end
    end

    return (ans)
end