"""
    Kolmogorov-Wiener Filter Specification and Fitted Models

Provides `KwFilterSpec` for specifying KW filter forecasting models using the
grammar interface, and `FittedKwFilter` for fitted KW models.
"""


"""
    KwFilterSpec(formula::ModelFormula; m=nothing, kwargs...)

Specify a Kolmogorov-Wiener optimal filter forecasting model using Durbyn's
forecasting grammar.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - KW filter options: `kw_filter()` with optional parameters

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (can also be specified in fit())
- Additional kwargs passed to `kolmogorov_wiener` during fitting

# Examples
```julia
# HP filter with default lambda
spec = KwFilterSpec(@formula(gdp = kw_filter()))

# HP filter for trend extraction
spec = KwFilterSpec(@formula(gdp = kw_filter(filter=:hp, lambda=1600, output=:trend)))

# Bandpass filter for business cycles
spec = KwFilterSpec(@formula(gdp = kw_filter(filter=:bandpass, low=6, high=32)))

# Fit to data
fitted = fit(spec, data, m=12)

# Generate forecasts
fc = forecast(fitted, h=12)

# Panel data support
fitted = fit(spec, data, m=12, groupby=[:country])
```
"""
struct KwFilterSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    options::Dict{Symbol, Any}

    function KwFilterSpec(formula::ModelFormula; m::Union{Int, Nothing}=nothing, kwargs...)
        new(formula, m, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedKwFilter

A fitted Kolmogorov-Wiener filter model containing the specification, fitted
result, and metadata needed for forecasting.

# Fields
- `spec::KwFilterSpec` - Original specification
- `fit::KWFilterResult` - Fitted KW filter result
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation
- `m::Int` - Seasonal period used for fitting
"""
struct FittedKwFilter <: AbstractFittedModel
    spec::KwFilterSpec
    fit::KWFilterResult
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedKwFilter(spec::KwFilterSpec,
                            fit,
                            target_col::Symbol,
                            data,
                            m::Int)
        schema = Dict{Symbol, Type}()
        for (k, v) in pairs(data)
            schema[k] = eltype(v)
        end
        new(spec, fit, target_col, schema, m)
    end
end

function extract_metrics(model::FittedKwFilter)
    metrics = Dict{Symbol, Float64}()
    r = residuals(model.fit)
    metrics[:mse] = sum(r .^ 2) / length(r)
    return metrics
end

function Base.show(io::IO, spec::KwFilterSpec)
    print(io, "KwFilterSpec: ", spec.formula.target)
    if !isnothing(spec.m)
        print(io, ", m=", spec.m)
    end
end

function Base.show(io::IO, fitted::FittedKwFilter)
    print(io, "FittedKwFilter: ", fitted.fit.filter_type)
    print(io, ", output=", fitted.fit.output)
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedKwFilter)
    println(io, "FittedKwFilter")
    println(io, "  Target: ", fitted.target_col)
    println(io, "  Filter: ", fitted.fit.filter_type)
    println(io, "  Output: ", fitted.fit.output)
    println(io, "  m: ", fitted.m)
    if !isempty(fitted.fit.params)
        println(io, "  Parameters:")
        for (k, v) in fitted.fit.params
            println(io, "    ", k, ": ", v)
        end
    end
    r = residuals(fitted.fit)
    mse = sum(r .^ 2) / length(r)
    println(io, "  MSE: ", round(mse, digits=6))
    println(io, "  n: ", length(fitted.fit.y))
end
