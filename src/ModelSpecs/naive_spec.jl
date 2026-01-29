"""
    Naive Model Specifications and Fitted Models

Provides `NaiveSpec`, `SnaiveSpec`, and `RwSpec` for specifying naive forecasting
models using the grammar interface, and their corresponding fitted model types.

Naive methods serve as simple benchmarks for more complex forecasting methods:
- Naive: uses last observation as forecast
- Seasonal Naive: uses observation from m periods ago
- Random Walk: naive with optional drift
"""

import ..Grammar: _extract_single_term, NaiveTerm, SnaiveTerm, RwTerm, MeanfTerm

"""
    NaiveSpec(formula::ModelFormula; m=nothing, kwargs...)

Specify a naive forecasting model using Durbyn's forecasting grammar.

The naive method uses the last observed value as the forecast for all future periods.
Equivalent to a random walk without drift.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - `naive_term()` on RHS

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (stored for reference)
- `lambda::Union{Nothing, Float64}=nothing` - Box-Cox transformation parameter
- `biasadj::Bool=false` - Bias adjustment for Box-Cox back-transformation

# Examples
```julia
# Naive specification
spec = NaiveSpec(@formula(sales = naive_term()))

# Fit to data
fitted = fit(spec, data)

# Generate forecasts
fc = forecast(fitted, h=12)

# Panel data support
fitted = fit(spec, data, groupby=[:product, :region])
```

# See Also
- [`SnaiveSpec`](@ref) - Seasonal naive specification
- [`RwSpec`](@ref) - Random walk specification
"""
struct NaiveSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    lambda::Union{Nothing, Float64}
    biasadj::Bool
    options::Dict{Symbol, Any}

    function NaiveSpec(formula::ModelFormula;
                       m::Union{Int, Nothing}=nothing,
                       lambda::Union{Nothing, Float64}=nothing,
                       biasadj::Bool=false,
                       kwargs...)
        # Validate formula contains NaiveTerm
        _extract_single_term(formula, NaiveTerm)
        new(formula, m, lambda, biasadj, Dict{Symbol, Any}(kwargs))
    end
end

"""
    SnaiveSpec(formula::ModelFormula; m=nothing, kwargs...)

Specify a seasonal naive forecasting model using Durbyn's forecasting grammar.

The seasonal naive method uses the observation from m periods ago as the forecast.
This is equivalent to an ARIMA(0,0,0)(0,1,0)_m model.
The seasonal period `m` is required either at spec construction or at fit time.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - `snaive_term()` on RHS

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (required at spec or fit time)
- `lambda::Union{Nothing, Float64}=nothing` - Box-Cox transformation parameter
- `biasadj::Bool=false` - Bias adjustment for Box-Cox back-transformation

# Examples
```julia
# Seasonal naive specification (monthly data with yearly seasonality)
spec = SnaiveSpec(@formula(sales = snaive_term()), m=12)

# Or specify m at fit time
spec = SnaiveSpec(@formula(sales = snaive_term()))
fitted = fit(spec, data, m=12)

# Generate forecasts
fc = forecast(fitted, h=24)

# Panel data support
fitted = fit(spec, data, m=12, groupby=[:product, :region])
```

# See Also
- [`NaiveSpec`](@ref) - Non-seasonal naive specification
- [`RwSpec`](@ref) - Random walk specification
"""
struct SnaiveSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    lambda::Union{Nothing, Float64}
    biasadj::Bool
    options::Dict{Symbol, Any}

    function SnaiveSpec(formula::ModelFormula;
                        m::Union{Int, Nothing}=nothing,
                        lambda::Union{Nothing, Float64}=nothing,
                        biasadj::Bool=false,
                        kwargs...)
        # Validate formula contains SnaiveTerm
        _extract_single_term(formula, SnaiveTerm)
        new(formula, m, lambda, biasadj, Dict{Symbol, Any}(kwargs))
    end
end

"""
    RwSpec(formula::ModelFormula; m=nothing, drift=false, kwargs...)

Specify a random walk forecasting model using Durbyn's forecasting grammar.

Without drift, this is equivalent to the naive method. With drift, the forecast
includes a linear trend based on the average change in the historical data.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - `rw_term()` or `rw_term(drift=true)` on RHS

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (stored for reference)
- `drift::Bool=false` - Include drift term (can also be specified in formula)
- `lambda::Union{Nothing, Float64}=nothing` - Box-Cox transformation parameter
- `biasadj::Bool=false` - Bias adjustment for Box-Cox back-transformation

# Examples
```julia
# Random walk without drift
spec = RwSpec(@formula(sales = rw_term()))

# Random walk with drift (formula-level)
spec = RwSpec(@formula(sales = rw_term(drift=true)))

# Random walk with drift (spec-level)
spec = RwSpec(@formula(sales = rw_term()), drift=true)

# Fit and forecast
fitted = fit(spec, data)
fc = forecast(fitted, h=12)

# Panel data support
fitted = fit(spec, data, groupby=[:product, :region])
```

# See Also
- [`NaiveSpec`](@ref) - Naive specification
- [`SnaiveSpec`](@ref) - Seasonal naive specification
"""
struct RwSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    drift::Bool
    lambda::Union{Nothing, Float64}
    biasadj::Bool
    options::Dict{Symbol, Any}

    function RwSpec(formula::ModelFormula;
                    m::Union{Int, Nothing}=nothing,
                    drift::Bool=false,
                    lambda::Union{Nothing, Float64}=nothing,
                    biasadj::Bool=false,
                    kwargs...)
        # Validate formula contains RwTerm and extract it
        rw_term = _extract_single_term(formula, RwTerm)
        # Use formula drift if not explicitly specified via kwarg (kwarg takes precedence)
        final_drift = drift || rw_term.drift
        new(formula, m, final_drift, lambda, biasadj, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedNaive

A fitted naive model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::NaiveSpec` - Original specification
- `fit::Any` - Fitted NaiveFit object
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation
- `m::Int` - Seasonal period used

# Examples
```julia
spec = NaiveSpec(@formula(sales = naive_term()))
fitted = fit(spec, data)

# Access underlying fit
fitted.fit.fitted      # Fitted values
fitted.fit.residuals   # Residuals
fitted.fit.sigma2      # Residual variance

# Generate forecasts
fc = forecast(fitted, h=12)
```
"""
struct FittedNaive <: AbstractFittedModel
    spec::NaiveSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedNaive(spec::NaiveSpec,
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

"""
    FittedSnaive

A fitted seasonal naive model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::SnaiveSpec` - Original specification
- `fit::Any` - Fitted NaiveFit object
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation
- `m::Int` - Seasonal period used

# Examples
```julia
spec = SnaiveSpec(@formula(sales = snaive_term()), m=12)
fitted = fit(spec, data)

# Access underlying fit
fitted.fit.lag         # Lag used (equals m)
fitted.fit.sigma2      # Residual variance

# Generate forecasts
fc = forecast(fitted, h=24)
```
"""
struct FittedSnaive <: AbstractFittedModel
    spec::SnaiveSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedSnaive(spec::SnaiveSpec,
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

"""
    FittedRw

A fitted random walk model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::RwSpec` - Original specification
- `fit::Any` - Fitted NaiveFit object
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation
- `m::Int` - Seasonal period used

# Examples
```julia
spec = RwSpec(@formula(sales = rw_term(drift=true)))
fitted = fit(spec, data)

# Access underlying fit
fitted.fit.drift       # Drift coefficient
fitted.fit.drift_se    # Drift standard error
fitted.fit.sigma2      # Residual variance

# Generate forecasts
fc = forecast(fitted, h=12)
```
"""
struct FittedRw <: AbstractFittedModel
    spec::RwSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedRw(spec::RwSpec,
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

"""
    MeanfSpec(formula::ModelFormula; m=nothing, kwargs...)

Specify a mean forecasting model using Durbyn's forecasting grammar.

The mean method uses the sample mean as the forecast for all future periods.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - `meanf_term()` on RHS

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (stored for reference)
- `lambda::Union{Nothing, Float64}=nothing` - Box-Cox transformation parameter
- `biasadj::Bool=false` - Bias adjustment for Box-Cox back-transformation

# Examples
```julia
# Mean specification
spec = MeanfSpec(@formula(sales = meanf_term()))

# Fit to data
fitted = fit(spec, data, m=12)

# Generate forecasts
fc = forecast(fitted, h=12)

# Panel data support
fitted = fit(spec, data, m=12, groupby=[:product, :region])
```

# See Also
- [`NaiveSpec`](@ref) - Naive specification
- [`SnaiveSpec`](@ref) - Seasonal naive specification
- [`RwSpec`](@ref) - Random walk specification
"""
struct MeanfSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    lambda::Union{Nothing, Float64}
    biasadj::Bool
    options::Dict{Symbol, Any}

    function MeanfSpec(formula::ModelFormula;
                       m::Union{Int, Nothing}=nothing,
                       lambda::Union{Nothing, Float64}=nothing,
                       biasadj::Bool=false,
                       kwargs...)
        # Validate formula contains MeanfTerm
        _extract_single_term(formula, MeanfTerm)
        new(formula, m, lambda, biasadj, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedMeanf

A fitted mean model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::MeanfSpec` - Original specification
- `fit::Any` - Fitted MeanFit object
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation
- `m::Int` - Seasonal period used

# Examples
```julia
spec = MeanfSpec(@formula(sales = meanf_term()))
fitted = fit(spec, data, m=12)

# Access underlying fit
fitted.fit.mu           # Mean (transformed scale)
fitted.fit.mu_original  # Mean (original scale)
fitted.fit.sd           # Standard deviation

# Generate forecasts
fc = forecast(fitted, h=12)
```
"""
struct FittedMeanf <: AbstractFittedModel
    spec::MeanfSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedMeanf(spec::MeanfSpec,
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

# Extract metrics - naive methods don't have standard information criteria
function extract_metrics(model::FittedNaive)
    return Dict{Symbol, Float64}()
end

function extract_metrics(model::FittedSnaive)
    return Dict{Symbol, Float64}()
end

function extract_metrics(model::FittedRw)
    return Dict{Symbol, Float64}()
end

function extract_metrics(model::FittedMeanf)
    return Dict{Symbol, Float64}()
end

# Show methods
function Base.show(io::IO, spec::NaiveSpec)
    print(io, "NaiveSpec(")
    print(io, spec.formula)
    if !isnothing(spec.m)
        print(io, ", m=", spec.m)
    end
    if !isnothing(spec.lambda)
        print(io, ", lambda=", spec.lambda)
    end
    print(io, ")")
end

function Base.show(io::IO, spec::SnaiveSpec)
    print(io, "SnaiveSpec(")
    print(io, spec.formula)
    if !isnothing(spec.m)
        print(io, ", m=", spec.m)
    end
    if !isnothing(spec.lambda)
        print(io, ", lambda=", spec.lambda)
    end
    print(io, ")")
end

function Base.show(io::IO, spec::RwSpec)
    print(io, "RwSpec(")
    print(io, spec.formula)
    if !isnothing(spec.m)
        print(io, ", m=", spec.m)
    end
    if spec.drift
        print(io, ", drift=true")
    end
    if !isnothing(spec.lambda)
        print(io, ", lambda=", spec.lambda)
    end
    print(io, ")")
end

function Base.show(io::IO, fitted::FittedNaive)
    println(io, "FittedNaive")
    println(io, "  Method: ", fitted.fit.method)
    println(io, "  Target: ", fitted.target_col)
    println(io, "  Series length: ", length(fitted.fit.x))
    println(io, "  Residual variance: ", round(fitted.fit.sigma2, digits=6))
end

function Base.show(io::IO, fitted::FittedSnaive)
    println(io, "FittedSnaive")
    println(io, "  Method: ", fitted.fit.method)
    println(io, "  Target: ", fitted.target_col)
    println(io, "  Seasonal period (m): ", fitted.m)
    println(io, "  Series length: ", length(fitted.fit.x))
    println(io, "  Residual variance: ", round(fitted.fit.sigma2, digits=6))
end

function Base.show(io::IO, fitted::FittedRw)
    println(io, "FittedRw")
    println(io, "  Method: ", fitted.fit.method)
    println(io, "  Target: ", fitted.target_col)
    if !isnothing(fitted.fit.drift)
        println(io, "  Drift: ", round(fitted.fit.drift, digits=6))
        println(io, "  Drift SE: ", round(fitted.fit.drift_se, digits=6))
    end
    println(io, "  Series length: ", length(fitted.fit.x))
    println(io, "  Residual variance: ", round(fitted.fit.sigma2, digits=6))
end

function Base.show(io::IO, spec::MeanfSpec)
    print(io, "MeanfSpec(")
    print(io, spec.formula)
    if !isnothing(spec.m)
        print(io, ", m=", spec.m)
    end
    if !isnothing(spec.lambda)
        print(io, ", lambda=", spec.lambda)
    end
    print(io, ")")
end

function Base.show(io::IO, fitted::FittedMeanf)
    println(io, "FittedMeanf")
    println(io, "  Target: ", fitted.target_col)
    println(io, "  Seasonal period (m): ", fitted.m)
    println(io, "  Series length: ", fitted.fit.n)
    println(io, "  Mean (original scale): ", round(fitted.fit.mu_original, digits=6))
    println(io, "  Standard deviation: ", round(fitted.fit.sd, digits=6))
end
