"""
    ARAR Model Specification and Fitted Models

Provides `ArarSpec` for specifying ARAR models using the grammar interface,
and `FittedArar` for fitted ARAR models.
"""


"""
    ArarSpec(formula::ModelFormula; kwargs...)

Specify an ARAR (AutoRegressive with Adaptive Reduction) model structure using Durbyn's forecasting grammar.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - ARAR options: `arar()` with optional `max_ar_depth` and `max_lag` parameters

# Keyword Arguments
- Additional kwargs passed to `arar` during fitting

# ARAR Specification

**Default parameters**:
```julia
@formula(sales = arar())
```

**Custom parameters**:
```julia
@formula(sales = arar(max_ar_depth=20, max_lag=20))
```

# Examples
```julia
# ARAR with defaults
spec = ArarSpec(@formula(sales = arar()))

# ARAR with custom parameters
spec = ArarSpec(@formula(sales = arar(max_ar_depth=15)))

# ARAR with both parameters
spec = ArarSpec(@formula(sales = arar(max_ar_depth=20, max_lag=20)))

# Fit to data
fitted = fit(spec, data)

# Generate forecasts
fc = forecast(fitted, h = 12)
```

# See Also
- [`@formula`](@ref)
- [`arar`](@ref)
- [`fit`](@ref)
"""
struct ArarSpec <: AbstractModelSpec
    formula::ModelFormula
    options::Dict{Symbol, Any}

    function ArarSpec(formula::ModelFormula; kwargs...)
        new(formula, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedArar

A fitted ARAR model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::ArarSpec` - Original specification
- `fit::Any` - Fitted ARAR model
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation

# Examples
```julia
spec = ArarSpec(@formula(sales = arar()))
fitted = fit(spec, data)

# Access underlying ARAR fit
fitted.fit.best_lag     # Selected AR lags
fitted.fit.best_phi     # AR coefficients
fitted.fit.sigma2       # Residual variance

# Generate forecasts
fc = forecast(fitted, h = 12)
```

# See Also
- [`ArarSpec`](@ref)
- [`fit`](@ref)
- [`forecast`](@ref)
"""
struct FittedArar <: AbstractFittedModel
    spec::ArarSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}

    function FittedArar(spec::ArarSpec,
                        fit,
                        target_col::Symbol,
                        data)
        schema = Dict{Symbol, Type}()
        for (k, v) in pairs(data)
            schema[k] = eltype(v)
        end
        new(spec, fit, target_col, schema)
    end
end

function extract_metrics(model::FittedArar)
    metrics = Dict{Symbol, Float64}()
    metrics[:sigma2] = model.fit.sigma2
    return metrics
end

function Base.show(io::IO, spec::ArarSpec)
    print(io, "ArarSpec: ", spec.formula.target)
end

function Base.show(io::IO, fitted::FittedArar)
    print(io, "FittedArar: lags=", fitted.fit.best_lag)
    print(io, ", σ² = ", round(fitted.fit.sigma2, digits=4))
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedArar)
    println(io, "FittedArar")
    println(io, "  Target: ", fitted.target_col)
    println(io, "  Selected lags: ", fitted.fit.best_lag)
    println(io, "  AR coefficients: ", round.(fitted.fit.best_phi, digits=4))
    println(io, "  σ²: ", round(fitted.fit.sigma2, digits=6))
    println(io, "  n: ", length(fitted.fit.y))
end
