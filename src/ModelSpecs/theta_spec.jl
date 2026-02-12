"""
    Theta Model Specification and Fitted Models

Provides `ThetaSpec` for specifying Theta models using the grammar interface,
and `FittedTheta` for fitted Theta models.

The Theta method decomposes a time series into "theta lines" capturing long-term
trend and short-term dynamics, then combines their forecasts. Supports four variants:
STM, OTM, DSTM, DOTM.
"""


"""
    ThetaSpec(formula::ModelFormula; m=nothing, kwargs...)

Specify a Theta forecasting model using Durbyn's forecasting grammar.

The Theta method decomposes a series into theta lines capturing long-term trend
and short-term dynamics, then combines their forecasts.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - Theta options: `theta()` with optional parameters

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (can also be specified in fit())
- Additional kwargs passed to `theta` or `auto_theta` during fitting

# Theta Specification

**Auto-select best model** (default):
```julia
@formula(sales = theta())
```

**Specific model variant**:
```julia
@formula(sales = theta(model=:STM))   # Simple Theta (θ=2 fixed, α optimized)
@formula(sales = theta(model=:OTM))   # Optimized Theta (both optimized)
@formula(sales = theta(model=:DSTM))  # Dynamic Simple Theta
@formula(sales = theta(model=:DOTM))  # Dynamic Optimized Theta
```

**With fixed parameters**:
```julia
@formula(sales = theta(model=:OTM, alpha=0.3))
@formula(sales = theta(model=:OTM, theta_param=2.5))
```

**Seasonal decomposition**:
```julia
@formula(sales = theta(decomposition="multiplicative"))
@formula(sales = theta(decomposition="additive"))
```

**Multi-step MSE**:
```julia
@formula(sales = theta(nmse=5))  # 1-30, default is 3
```

# Examples
```julia
# Theta with auto model selection
spec = ThetaSpec(@formula(sales = theta()))

# Theta with specific model variant
spec = ThetaSpec(@formula(sales = theta(model=:OTM)))

# Theta with fixed smoothing parameter
spec = ThetaSpec(@formula(sales = theta(model=:OTM, alpha=0.2)))

# Theta with seasonal decomposition
spec = ThetaSpec(@formula(sales = theta(decomposition="additive")))

# Full specification
spec = ThetaSpec(@formula(sales = theta(model=:DOTM, decomposition="multiplicative", nmse=5)))

# Fit to data
fitted = fit(spec, data, m=12)

# Generate forecasts
fc = forecast(fitted, h=12)

# Panel data support
fitted = fit(spec, data, m=12, groupby=[:product, :region])
```

"""
struct ThetaSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    options::Dict{Symbol, Any}

    function ThetaSpec(formula::ModelFormula; m::Union{Int, Nothing}=nothing, kwargs...)
        new(formula, m, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedTheta

A fitted Theta model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::ThetaSpec` - Original specification
- `fit::Any` - Fitted Theta model (ThetaFit)
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation
- `m::Int` - Seasonal period used for fitting

# Examples
```julia
spec = ThetaSpec(@formula(sales = theta()))
fitted = fit(spec, data, m=12)

# Access underlying Theta fit
fitted.fit.model_type        # STM, OTM, DSTM, or DOTM
fitted.fit.alpha             # Smoothing parameter
fitted.fit.theta             # Theta parameter
fitted.fit.initial_level     # Initial level
fitted.fit.mse               # Mean squared error
fitted.fit.fitted            # Fitted values
fitted.fit.residuals         # In-sample residuals
fitted.fit.decompose         # Whether seasonal decomposition was applied
fitted.fit.decomposition_type  # "multiplicative", "additive", or "none"

# Generate forecasts
fc = forecast(fitted, h=12)
```

"""
struct FittedTheta <: AbstractFittedModel
    spec::ThetaSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedTheta(spec::ThetaSpec,
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

function extract_metrics(model::FittedTheta)
    metrics = Dict{Symbol, Float64}()
    if isfinite(model.fit.mse)
        metrics[:mse] = model.fit.mse
    end
    return metrics
end

function Base.show(io::IO, spec::ThetaSpec)
    print(io, "ThetaSpec: ", spec.formula.target)
    if !isnothing(spec.m)
        print(io, ", m=", spec.m)
    end
end

function Base.show(io::IO, fitted::FittedTheta)
    print(io, "FittedTheta: ", fitted.fit.model_type)
    print(io, ", MSE = ", round(fitted.fit.mse, digits=4))
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedTheta)
    println(io, "FittedTheta")
    println(io, "  Target: ", fitted.target_col)
    println(io, "  Model: ", fitted.fit.model_type)
    println(io, "  m: ", fitted.m)
    println(io, "  Alpha: ", round(fitted.fit.alpha, digits=4))
    println(io, "  Theta: ", round(fitted.fit.theta, digits=4))
    if fitted.fit.decompose
        println(io, "  Decomposition: ", fitted.fit.decomposition_type)
    end
    println(io, "  MSE: ", round(fitted.fit.mse, digits=6))
    println(io, "  n: ", length(fitted.fit.y))
end
