"""
    Abstract types for model specifications and fitted models.

This module defines the abstract type hierarchy for Durbyn's fable-like
forecasting interface.

# Type Hierarchy

```
AbstractModelSpec
├── ArimaSpec
├── EtsSpec (future)
└── ... (extensible)

AbstractFittedModel
├── FittedArima
├── FittedEts (future)
└── ... (extensible)
```

# Design Philosophy

1. **Specification vs Fitting**: Model specs are declarative (no data),
   fitting applies specs to data.

2. **Generic Interface**: All model types support:
   - `fit(spec, data)` → fitted model
   - `forecast(fitted, h)` → forecasts

3. **Composable**: Single models or collections for comparison/ensembling.
"""

# ============================================================================
# Model Specification Types
# ============================================================================

"""
    AbstractModelSpec

Abstract base type for all model specifications.

A model specification describes the structure of a model (e.g., ARIMA orders,
ETS components) without being tied to any particular dataset. Specifications
are declarative and can be applied to data via `fit()`.

# Interface Requirements

Subtypes must implement:
- `fit(spec::YourSpec, data; kwargs...)` → returns `AbstractFittedModel`

Optional:
- Custom constructors for ergonomic specification
- Validation in constructor

# Examples
```julia
# ARIMA specification
spec = ArimaSpec(@formula(sales = p(1,2) + q(2,3)))

# Apply to data
fitted = fit(spec, data, m = 12)
```

# See Also
- [`ArimaSpec`](@ref)
- [`fit`](@ref)
"""
abstract type AbstractModelSpec end

# ============================================================================
# Fitted Model Types
# ============================================================================

"""
    AbstractFittedModel

Abstract base type for fitted models.

A fitted model contains:
1. The original specification
2. The fitted parameters/state
3. Metadata about the fitting process
4. Information needed for forecasting

# Interface Requirements

Subtypes must implement:
- `forecast(fitted::YourFitted, h::Int; kwargs...)` → forecast results

Recommended fields:
- `spec::AbstractModelSpec` - Original specification
- `target_col::Symbol` - Name of target variable
- `xreg_cols::Vector{Symbol}` - Names of exogenous variables (if any)
- `data_schema::Dict{Symbol, Type}` - For validation

# Examples
```julia
# Get fitted model
fitted = fit(spec, data, m = 12)

# Generate forecasts
fc = forecast(fitted, h = 12)
```

# See Also
- [`FittedArima`](@ref)
- [`forecast`](@ref)
"""
abstract type AbstractFittedModel end

# ============================================================================
# Model Collection Type
# ============================================================================

"""
    ModelCollection

Container for multiple model specifications.

Used for model comparison, selection, and ensembling. Created via `model()`
function when multiple specs are provided.

# Fields
- `specs::Vector{AbstractModelSpec}` - Model specifications
- `names::Vector{String}` - Names for each model

# Examples
```julia
# Create collection
models = model(
    ArimaSpec(@formula(y = p() + q())),
    ArimaSpec(@formula(y = p(1) + d(1) + q(1))),
    names = ["auto", "fixed"]
)

# Fit all models
fitted = fit(models, data, m = 12)

# Compare and select best
best = select_best(fitted, metric = :aic)
```

# See Also
- [`model`](@ref)
- [`FittedModelCollection`](@ref)
"""
struct ModelCollection
    specs::Vector{AbstractModelSpec}
    names::Vector{String}

    function ModelCollection(specs::Vector{<:AbstractModelSpec}, names::Vector{String})
        length(specs) == length(names) ||
            error("Number of specs ($(length(specs))) must equal number of names ($(length(names)))")
        length(specs) > 0 ||
            error("ModelCollection must contain at least one model specification")
        new(specs, names)
    end
end

"""
    FittedModelCollection

Multiple fitted models (for comparison/ensembling).

# Fields
- `models::Vector{AbstractFittedModel}` - Fitted models
- `names::Vector{String}` - Names for each model
- `metrics::Dict{String, Dict{Symbol, Float64}}` - Fit metrics per model

# Examples
```julia
# Fit collection
models = model(spec1, spec2, spec3, names = ["m1", "m2", "m3"])
fitted = fit(models, data, m = 12)

# Access metrics
fitted.metrics["m1"]  # Dict(:aic => ..., :bic => ...)

# Forecast with best model
fc = forecast(fitted, h = 12, method = :best)

# Ensemble forecast
fc_ensemble = forecast(fitted, h = 12, method = :mean)
```

# See Also
- [`ModelCollection`](@ref)
- [`forecast`](@ref)
"""
struct FittedModelCollection
    models::Vector{AbstractFittedModel}
    names::Vector{String}
    metrics::Dict{String, Dict{Symbol, Float64}}

    function FittedModelCollection(models::Vector{<:AbstractFittedModel},
                                   names::Vector{String})
        length(models) == length(names) ||
            error("Number of models must equal number of names")

        # Extract metrics from each model
        metrics = Dict{String, Dict{Symbol, Float64}}()
        for (name, model) in zip(names, models)
            metrics[name] = extract_metrics(model)
        end

        new(models, names, metrics)
    end
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    extract_metrics(model::AbstractFittedModel) -> Dict{Symbol, Float64}

Extract fit quality metrics from a fitted model.

Default implementation returns empty dict. Subtypes should override.

# Returns
Dictionary with metrics like:
- `:aic` - Akaike Information Criterion
- `:aicc` - Corrected AIC
- `:bic` - Bayesian Information Criterion
- `:rmse` - Root Mean Squared Error (if available)
"""
function extract_metrics(model::AbstractFittedModel)
    # Default: return empty dict
    # Subtypes override this
    return Dict{Symbol, Float64}()
end

# ============================================================================
# Pretty Printing
# ============================================================================

function Base.show(io::IO, collection::ModelCollection)
    n = length(collection.specs)
    print(io, "ModelCollection with $n model")
    n > 1 && print(io, "s")
    if n <= 5
        print(io, ": ")
        print(io, join(collection.names, ", "))
    end
end

function Base.show(io::IO, fitted::FittedModelCollection)
    n = length(fitted.models)
    print(io, "FittedModelCollection with $n fitted model")
    n > 1 && print(io, "s")

    if n <= 5
        println(io, ":")
        for name in fitted.names
            metrics = fitted.metrics[name]
            aic = get(metrics, :aic, NaN)
            print(io, "  $name: ")
            if !isnan(aic)
                print(io, "AIC = ", round(aic, digits=2))
            end
            println(io)
        end
    end
end
