"""
    Diffusion Model Specification and Fitted Models

Provides `DiffusionSpec` for specifying diffusion models using the grammar interface,
and `FittedDiffusion` for fitted diffusion models.

Diffusion models capture technology adoption and market penetration patterns using
classical S-curve models: Bass, Gompertz, GSGompertz, and Weibull.
"""


"""
    DiffusionSpec(formula::ModelFormula; kwargs...)

Specify a diffusion forecasting model using Durbyn's forecasting grammar.

Diffusion models capture technology adoption and market penetration patterns,
modeling how innovations spread through a population over time.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - Diffusion options: `diffusion()` with optional parameters

# Keyword Arguments
- Additional kwargs passed to `diffusion` during fitting

# Diffusion Specification

**Auto-select Bass model** (default):
```julia
@formula(sales = diffusion())
```

**Specific model type**:
```julia
@formula(sales = diffusion(model=:Bass))       # Bass diffusion model
@formula(sales = diffusion(model=:Gompertz))   # Gompertz growth curve
@formula(sales = diffusion(model=:GSGompertz)) # Gamma/Shifted Gompertz
@formula(sales = diffusion(model=:Weibull))    # Weibull distribution
```

**With fixed parameters** (Bass example):
```julia
@formula(sales = diffusion(model=:Bass, m=1000))     # Fix market potential
@formula(sales = diffusion(model=:Bass, p=0.03))     # Fix innovation coefficient
@formula(sales = diffusion(model=:Bass, q=0.38))     # Fix imitation coefficient
```

**Optimization options**:
```julia
@formula(sales = diffusion(loss=1))                  # Use L1 loss (MAE)
@formula(sales = diffusion(cumulative=false))        # Optimize on adoption, not cumulative
```

# Examples
```julia
# Basic diffusion specification
spec = DiffusionSpec(@formula(adoption = diffusion()))

# Bass model specification
spec = DiffusionSpec(@formula(adoption = diffusion(model=:Bass)))

# Gompertz with fixed market potential
spec = DiffusionSpec(@formula(adoption = diffusion(model=:Gompertz, m=5000)))

# Fit to data
fitted = fit(spec, data)

# Generate forecasts
fc = forecast(fitted, h=12)

# Panel data support
fitted = fit(spec, data, groupby=[:product, :region])
```

"""
struct DiffusionSpec <: AbstractModelSpec
    formula::ModelFormula
    options::Dict{Symbol, Any}

    function DiffusionSpec(formula::ModelFormula; kwargs...)
        new(formula, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedDiffusion

A fitted diffusion model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::DiffusionSpec` - Original specification
- `fit::Any` - Fitted diffusion model (DiffusionFit)
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation

# Examples
```julia
spec = DiffusionSpec(@formula(adoption = diffusion()))
fitted = fit(spec, data)

# Access underlying diffusion fit
fitted.fit.model_type        # Bass, Gompertz, GSGompertz, or Weibull
fitted.fit.params            # Model parameters (m, p, q for Bass)
fitted.fit.mse               # Mean squared error
fitted.fit.fitted            # Fitted values
fitted.fit.residuals         # In-sample residuals

# Generate forecasts
fc = forecast(fitted, h=12)
```

"""
struct FittedDiffusion <: AbstractFittedModel
    spec::DiffusionSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}

    function FittedDiffusion(spec::DiffusionSpec,
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

function extract_metrics(model::FittedDiffusion)
    metrics = Dict{Symbol, Float64}()
    if hasproperty(model.fit, :mse) && !isnothing(model.fit.mse)
        metrics[:mse] = Float64(model.fit.mse)
    end
    return metrics
end

function Base.show(io::IO, spec::DiffusionSpec)
    print(io, "DiffusionSpec: ", spec.formula.target)
end

function Base.show(io::IO, fitted::FittedDiffusion)
    print(io, "FittedDiffusion: ", fitted.fit.model_type)
    print(io, ", MSE = ", round(fitted.fit.mse, digits=4))
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedDiffusion)
    println(io, "FittedDiffusion")
    println(io, "  Target: ", fitted.target_col)
    println(io, "  Model: ", fitted.fit.model_type)
    println(io, "  MSE: ", round(fitted.fit.mse, digits=6))
    println(io, "  Parameters:")
    for (k, v) in pairs(fitted.fit.params)
        println(io, "    ", k, ": ", round(v, digits=6))
    end
end
