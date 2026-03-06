"""
    Decomposition{T<:Real}

Generic result of decomposing a time series into trend, seasonal, and remainder
components.  Works for any decomposition method (KW, classical, STL, MSTL).

The exact semantics of the components depend on `type`:
- `:additive`        — `data = trend + sum(seasonals) + remainder`
- `:multiplicative`  — `data = trend * prod(seasonals) * remainder`

# Fields
- `data::Vector{T}`:           Original series.
- `trend::Vector{T}`:          Trend component.
- `seasonals::Vector{Vector{T}}`:  Seasonal components (empty for KW, 1 for STL/classical, N for MSTL).
- `remainder::Vector{T}`:      Remainder (cycle for KW, random for classical).
- `method::Symbol`:            Decomposition method (:kw, :classical, :stl, :mstl).
- `type::Symbol`:              :additive or :multiplicative.
- `m::Vector{Int}`:            Seasonal periods (empty if non-seasonal).
- `metadata::Dict{Symbol,Any}`: Method-specific data (arima_model, gamma, etc.).
"""
struct Decomposition{T<:Real}
    data::Vector{T}
    trend::Vector{T}
    seasonals::Vector{Vector{T}}
    remainder::Vector{T}
    method::Symbol
    type::Symbol
    m::Vector{Int}
    metadata::Dict{Symbol,Any}
end

"""
    fitted(d::Decomposition)

Return the fitted (reconstructed) values from a decomposition.

- Additive:        `trend + sum(seasonals)`
- Multiplicative:  `trend .* prod(seasonals)`

When there are no seasonal components, returns the trend.
"""
function fitted(d::Decomposition)
    if isempty(d.seasonals)
        return d.trend
    end
    if d.type === :multiplicative
        out = copy(d.trend)
        for s in d.seasonals
            out .*= s
        end
        return out
    else
        out = copy(d.trend)
        for s in d.seasonals
            out .+= s
        end
        return out
    end
end

"""
    residuals(d::Decomposition)

Return the remainder component of a decomposition.
"""
residuals(d::Decomposition) = d.remainder

function show(io::IO, d::Decomposition)
    T = length(d.data)
    println(io, "Decomposition (", d.method, ")")
    println(io, "  Type:          ", d.type)
    println(io, "  Series length: ", T)
    if !isempty(d.m)
        println(io, "  Seasonal periods: ", d.m)
    end
    if !isempty(d.seasonals)
        println(io, "  Seasonal components: ", length(d.seasonals))
    end
    # Method-specific metadata
    for key in (:filter_type, :d, :arima_model)
        if haskey(d.metadata, key)
            println(io, "  ", key, ": ", d.metadata[key])
        end
    end
    if haskey(d.metadata, :params)
        params = d.metadata[:params]
        if !isempty(params)
            println(io, "  Parameters:")
            for (k, v) in params
                println(io, "    ", k, " = ", v)
            end
        end
    end
end
