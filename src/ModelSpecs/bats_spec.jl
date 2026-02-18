"""
    BATS Model Specification and Fitted Models

Provides `BatsSpec` for specifying BATS models using the grammar interface,
and `FittedBats` for fitted BATS models.
"""


"""
    BatsSpec(formula::ModelFormula; kwargs...)

Specify a BATS (Box-Cox transformation, ARMA errors, Trend and Seasonal components)
model structure using Durbyn's forecasting grammar.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - BATS options: `bats()` with optional parameters

# Keyword Arguments
- Additional kwargs passed to `bats` during fitting

# BATS Specification

**Default parameters** (automatic component selection):
```julia
@formula(sales = bats())
```

**Seasonal periods**:
```julia
# Single seasonal period
@formula(sales = bats(m=12))

# Multiple seasonal periods
@formula(sales = bats(m=[24, 168]))
```

**Component selection**:
```julia
# Specify Box-Cox transformation
@formula(sales = bats(m=12, use_box_cox=true))

# Specify trend component
@formula(sales = bats(m=12, use_trend=true))

# Specify damped trend
@formula(sales = bats(m=12, use_trend=true, use_damped_trend=true))

# Include ARMA errors
@formula(sales = bats(m=12, use_arma_errors=true))
```

# Examples
```julia
# BATS with defaults (automatic selection)
spec = BatsSpec(@formula(sales = bats()))

# BATS with monthly seasonality
spec = BatsSpec(@formula(sales = bats(m=12)))

# BATS with multiple seasonal periods (e.g., daily and weekly)
spec = BatsSpec(@formula(sales = bats(m=[24, 168])))

# BATS with Box-Cox and trend
spec = BatsSpec(@formula(sales = bats(m=12, use_box_cox=true, use_trend=true)))

# BATS with all options
spec = BatsSpec(@formula(sales = bats(
    m=12,
    use_box_cox=true,
    use_trend=true,
    use_damped_trend=false,
    use_arma_errors=true
)))

# Fit to data
fitted = fit(spec, data)

# Generate forecasts
fc = forecast(fitted, h = 12)
```

# See Also
- [`@formula`](@ref)
- [`bats`](@ref)
- [`fit`](@ref)
"""
struct BatsSpec <: AbstractModelSpec
    formula::ModelFormula
    options::Dict{Symbol, Any}

    function BatsSpec(formula::ModelFormula; kwargs...)
        new(formula, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedBats

A fitted BATS model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::BatsSpec` - Original specification
- `fit::Any` - Fitted BATS model (BATSModel)
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation

# Examples
```julia
spec = BatsSpec(@formula(sales = bats(m=12)))
fitted = fit(spec, data)

# Access underlying BATS fit
fitted.fit.lambda              # Box-Cox lambda
fitted.fit.alpha               # Level smoothing parameter
fitted.fit.beta                # Trend smoothing parameter
fitted.fit.seasonal_periods    # Seasonal periods
fitted.fit.ar_coefficients     # AR coefficients (if present)
fitted.fit.ma_coefficients     # MA coefficients (if present)
fitted.fit.fitted_values       # Fitted values
fitted.fit.variance            # Residual variance
fitted.fit.AIC                 # AIC

# Generate forecasts
fc = forecast(fitted, h = 12)
```

# See Also
- [`BatsSpec`](@ref)
- [`fit`](@ref)
- [`forecast`](@ref)
"""
struct FittedBats <: AbstractFittedModel
    spec::BatsSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}

    function FittedBats(spec::BatsSpec,
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

function extract_metrics(model::FittedBats)
    metrics = Dict{Symbol, Float64}()
    if isfinite(model.fit.AIC)
        metrics[:aic] = model.fit.AIC
    end
    if isfinite(model.fit.likelihood)
        metrics[:loglik] = model.fit.likelihood
    end
    metrics[:sigma2] = model.fit.variance
    return metrics
end

function Base.show(io::IO, spec::BatsSpec)
    print(io, "BatsSpec: ", spec.formula.target)
end

function Base.show(io::IO, fitted::FittedBats)
    print(io, "FittedBats: ", fitted.fit.method)
    if isfinite(fitted.fit.AIC)
        print(io, ", AIC = ", round(fitted.fit.AIC, digits=2))
    end
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedBats)
    println(io, "FittedBats")
    println(io, "  Target: ", fitted.target_col)
    println(io, "  Model: ", fitted.fit.method)
    if !isnothing(fitted.fit.lambda)
        println(io, "  Lambda: ", round(fitted.fit.lambda, digits=4))
    end
    println(io, "  Alpha: ", round(fitted.fit.alpha, digits=4))
    if !isnothing(fitted.fit.beta)
        println(io, "  Beta: ", round(fitted.fit.beta, digits=4))
    end
    if !isnothing(fitted.fit.damping_parameter)
        println(io, "  Damping: ", round(fitted.fit.damping_parameter, digits=4))
    end
    if isfinite(fitted.fit.AIC)
        println(io, "  AIC: ", round(fitted.fit.AIC, digits=4))
    end
    println(io, "  σ²: ", round(fitted.fit.variance, digits=6))
    println(io, "  n: ", length(fitted.fit.y))
end
