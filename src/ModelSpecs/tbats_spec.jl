"""
    TBATS Model Specification and Fitted Models

Provides `TbatsSpec` for specifying TBATS models using the grammar interface,
and `FittedTbats` for fitted TBATS models.

TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and
Seasonal components) extends BATS by using Fourier representation for seasonal
components, enabling non-integer seasonal periods and efficient handling of long cycles.
"""


"""
    TbatsSpec(formula::ModelFormula; kwargs...)

Specify a TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors,
Trend and Seasonal components) model structure using Durbyn's forecasting grammar.

TBATS extends BATS with Fourier-based seasonality, enabling:
- Non-integer seasonal periods (e.g., 52.18 weeks per year)
- Very long seasonal cycles (hundreds or thousands of periods)
- Multiple complex seasonalities (daily + weekly + yearly)
- Dual calendar effects (e.g., Gregorian + Hijri calendars)

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - TBATS options: `tbats()` with optional parameters

# Keyword Arguments
- Additional kwargs passed to `tbats` during fitting

# TBATS Specification

**Default parameters** (automatic component selection):
```julia
@formula(sales = tbats())
```

**Seasonal periods** (can be non-integer):
```julia
# Single seasonal period (non-integer allowed)
@formula(sales = tbats(seasonal_periods=52.18))

# Multiple seasonal periods
@formula(sales = tbats(seasonal_periods=[7, 365.25]))

# Dual calendar (Gregorian + Hijri)
@formula(sales = tbats(seasonal_periods=[365.25, 354.37]))
```

**Fourier orders** (k):
```julia
# Explicit Fourier orders per seasonal period
@formula(sales = tbats(seasonal_periods=[7, 365.25], k=[3, 10]))
```

**Component selection**:
```julia
# Specify Box-Cox transformation
@formula(sales = tbats(seasonal_periods=52.18, use_box_cox=true))

# Specify trend component
@formula(sales = tbats(seasonal_periods=52.18, use_trend=true))

# Specify damped trend
@formula(sales = tbats(seasonal_periods=52.18, use_trend=true, use_damped_trend=true))

# Include ARMA errors
@formula(sales = tbats(seasonal_periods=52.18, use_arma_errors=true))
```

# Examples
```julia
# TBATS with defaults (automatic selection)
spec = TbatsSpec(@formula(sales = tbats()))

# TBATS with non-integer seasonal period (weekly data, yearly pattern)
spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=52.18)))

# TBATS with multiple seasonal periods (daily + yearly)
spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=[7, 365.25])))

# TBATS with explicit Fourier orders
spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=[7, 365.25], k=[3, 10])))

# TBATS with Box-Cox and damped trend
spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=52.18, use_box_cox=true, use_damped_trend=true)))

# TBATS with all options
spec = TbatsSpec(@formula(sales = tbats(
    seasonal_periods=[7, 365.25],
    k=[3, 10],
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
- [`tbats`](@ref)
- [`fit`](@ref)
- [`BatsSpec`](@ref) - BATS specification (integer seasonal periods only)
"""
struct TbatsSpec <: AbstractModelSpec
    formula::ModelFormula
    options::Dict{Symbol, Any}

    function TbatsSpec(formula::ModelFormula; kwargs...)
        new(formula, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedTbats

A fitted TBATS model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::TbatsSpec` - Original specification
- `fit::Any` - Fitted TBATS model (TBATSModel)
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation

# Examples
```julia
spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=52.18)))
fitted = fit(spec, data)

# Access underlying TBATS fit
fitted.fit.lambda              # Box-Cox lambda
fitted.fit.alpha               # Level smoothing parameter
fitted.fit.beta                # Trend smoothing parameter
fitted.fit.damping_parameter   # Damping parameter phi
fitted.fit.gamma_one_values    # First seasonal smoothing parameters
fitted.fit.gamma_two_values    # Second seasonal smoothing parameters
fitted.fit.seasonal_periods    # Seasonal periods
fitted.fit.k_vector            # Fourier orders per seasonal period
fitted.fit.ar_coefficients     # AR coefficients (if present)
fitted.fit.ma_coefficients     # MA coefficients (if present)
fitted.fit.fitted_values       # Fitted values
fitted.fit.variance            # Residual variance
fitted.fit.AIC                 # AIC

# Generate forecasts
fc = forecast(fitted, h = 12)
```

# See Also
- [`TbatsSpec`](@ref)
- [`fit`](@ref)
- [`forecast`](@ref)
"""
struct FittedTbats <: AbstractFittedModel
    spec::TbatsSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}

    function FittedTbats(spec::TbatsSpec,
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

function extract_metrics(model::FittedTbats)
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

function Base.show(io::IO, spec::TbatsSpec)
    print(io, "TbatsSpec: ", spec.formula.target)
end

function Base.show(io::IO, fitted::FittedTbats)
    print(io, "FittedTbats: ", fitted.fit.method)
    if isfinite(fitted.fit.AIC)
        print(io, ", AIC = ", round(fitted.fit.AIC, digits=2))
    end
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedTbats)
    println(io, "FittedTbats")
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
    if !isnothing(fitted.fit.k_vector) && !isnothing(fitted.fit.seasonal_periods)
        periods = fitted.fit.seasonal_periods
        ks = fitted.fit.k_vector
        for (p, k) in zip(periods, ks)
            println(io, "  Seasonal: period=$p, k=$k")
        end
    end
    if isfinite(fitted.fit.AIC)
        println(io, "  AIC: ", round(fitted.fit.AIC, digits=4))
    end
    println(io, "  σ²: ", round(fitted.fit.variance, digits=6))
    println(io, "  n: ", length(fitted.fit.y))
end
