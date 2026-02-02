"""
    ARIMA Model Specification and Fitted Models

Provides `ArimaSpec` for specifying ARIMA models using the grammar interface,
and `FittedArima` for fitted ARIMA models.
"""


"""
    ArimaSpec(formula::ModelFormula; m=nothing, xreg_formula=nothing, kwargs...)

Specify an ARIMA model structure using Durbyn's forecasting grammar.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - ARIMA orders: `p()`, `d()`, `q()`, `P()`, `D()`, `Q()`
  - Exogenous variables (optional): variable names

# Keyword Arguments
- `m::Union{Int, Nothing}` - Seasonal period (e.g., 12 for monthly, 7 for daily weekly)
- `xreg_formula::Union{Formula, Nothing}` - Optional Formula for complex xreg design
- Additional kwargs passed to `auto_arima` during fitting

# ARIMA Order Specification

**Search ranges** (uses `auto_arima` for selection):
- `p()`, `q()` - Use defaults (p: 2-5, q: 2-5)
- `p(min, max)` - Search range for AR order
- `q(min, max)` - Search range for MA order

**Fixed values** (uses `arima` directly):
- `p(value)` - Fixed AR order
- `q(value)` - Fixed MA order

**Differencing**:
- `d()` - Auto-determine via unit root tests
- `d(value)` - Fixed differencing order

**Seasonal components**:
- `P()`, `Q()` - Use defaults (P: 1-2, Q: 1-2)
- `P(value)`, `Q(value)` - Fixed seasonal orders
- `D()` - Auto-determine seasonal differencing
- `D(value)` - Fixed seasonal differencing

# Exogenous Variables

**Simple inclusion** (via ModelFormula):
```julia
@formula(sales = p() + q() + temperature + promotion)
```

**Complex interactions** (via xreg_formula):
```julia
ArimaSpec(
    @formula(sales = p() + q()),
    xreg_formula = Formula("~ temperature * promotion + price")
)
```

# Examples
```julia
# Auto ARIMA (search for best p and q)
spec = ArimaSpec(@formula(sales = p() + q()))

# Fixed ARIMA(1,1,1)
spec = ArimaSpec(@formula(sales = p(1) + d(1) + q(1)))

# SARIMA with search
spec = ArimaSpec(
    @formula(sales = p(0,2) + d(1) + q(0,2) + P(0,1) + D(1) + Q(0,1)),
    m = 12
)

# ARIMAX with exogenous variables
spec = ArimaSpec(@formula(sales = p(1,3) + q(1,3) + temperature + promotion))

# Automatic exogenous selection (all numeric columns except target/group/date)
spec = ArimaSpec(@formula(sales = auto()))

# Complex interactions for xreg
spec = ArimaSpec(
    @formula(sales = p() + q()),
    m = 7,
    xreg_formula = Formula("~ temperature * promotion + price")
)

# With additional options
spec = ArimaSpec(
    @formula(sales = p() + q()),
    m = 12,
    method = "CSS-ML",
    lambda = :auto
)
```

# See Also
- [`@formula`](@ref)
- [`p`](@ref), [`d`](@ref), [`q`](@ref)
- [`P`](@ref), [`D`](@ref), [`Q`](@ref)
- [`fit`](@ref)
"""
struct ArimaSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    xreg_formula::Union{Formula, Nothing}
    auto_xreg::Bool
    options::Dict{Symbol, Any}

    function ArimaSpec(formula::ModelFormula;
                      m::Union{Int, Nothing} = nothing,
                      xreg_formula::Union{Formula, Nothing} = nothing,
                      kwargs...)
        auto_terms = filter(t -> isa(t, AutoVarTerm), formula.terms)
        length(auto_terms) <= 1 ||
            throw(ArgumentError("Formula may include at most one '.' sentinel for automatic exogenous selection."))
        auto_flag = !isempty(auto_terms)
        if auto_flag
            if any(t -> isa(t, VarTerm), formula.terms)
                throw(ArgumentError("Cannot mix explicit exogenous variables with '.' sentinel; please choose one approach."))
            end
            if !isnothing(xreg_formula)
                throw(ArgumentError("Cannot use xreg_formula when formula includes '.'."))
            end
        end
        new(formula, m, xreg_formula, auto_flag, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedArima

A fitted ARIMA model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::ArimaSpec` - Original specification
- `fit::Any` - Fitted ARIMA model (ArimaFit from auto_arima or arima)
- `target_col::Symbol` - Name of target variable
- `xreg_cols::Vector{Symbol}` - Names of exogenous variables (empty if none)
- `data_schema::Dict{Symbol, Type}` - Column types for validation
- `m::Int` - Seasonal period used

# Examples
```julia
spec = ArimaSpec(@formula(sales = p() + q()))
fitted = fit(spec, data, m = 12)

# Access underlying ARIMA fit
fitted.fit.arma  # ARIMA orders
fitted.fit.aic   # Model AIC
fitted.fit.coef  # Coefficients

# Generate forecasts
fc = forecast(fitted, h = 12)
```

# See Also
- [`ArimaSpec`](@ref)
- [`fit`](@ref)
- [`forecast`](@ref)
"""
struct FittedArima <: AbstractFittedModel
    spec::ArimaSpec
    fit::Any
    target_col::Symbol
    xreg_cols::Vector{Symbol}
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedArima(spec::ArimaSpec,
                        fit,
                        target_col::Symbol,
                        xreg_cols::Vector{Symbol},
                        data,
                        m::Int)
        schema = Dict{Symbol, Type}()
        for (k, v) in pairs(data)
            schema[k] = eltype(v)
        end

        new(spec, fit, target_col, xreg_cols, schema, m)
    end
end

"""
    extract_metrics(model::FittedArima)

Extract fit quality metrics from a fitted ARIMA model.
"""
function extract_metrics(model::FittedArima)
    metrics = Dict{Symbol, Float64}()

    if !isnothing(model.fit.aic)
        metrics[:aic] = model.fit.aic
    end
    if !isnothing(model.fit.aicc)
        metrics[:aicc] = model.fit.aicc
    end
    if !isnothing(model.fit.bic)
        metrics[:bic] = model.fit.bic
    end
    if !isnothing(model.fit.sigma2)
        metrics[:sigma2] = model.fit.sigma2
    end

    return metrics
end


function Base.show(io::IO, spec::ArimaSpec)
    print(io, "ArimaSpec: ")

    print(io, spec.formula)

    if !isnothing(spec.m)
        print(io, ", m = ", spec.m)
    end

    if !isnothing(spec.xreg_formula)
        print(io, ", xreg = Formula(...)")
    end
end

function Base.show(io::IO, fitted::FittedArima)
    
    p, q, P, Q, m, d, D = fitted.fit.arma

    model_str = "ARIMA($(Int(p)),$(Int(d)),$(Int(q)))"
    if m > 1
        model_str *= "($(Int(P)),$(Int(D)),$(Int(Q)))[$(Int(m))]"
    end

    if !isempty(fitted.xreg_cols)
        model_str *= " with xreg"
    end

    print(io, "FittedArima: ", model_str)

    if !isnothing(fitted.fit.aic)
        print(io, ", AIC = ", round(fitted.fit.aic, digits=2))
    end
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedArima)
    p, q, P, Q, m, d, D = fitted.fit.arma

    println(io, "FittedArima")
    println(io, "  Model: ARIMA($(Int(p)),$(Int(d)),$(Int(q)))")
    if m > 1
        println(io, "         ($(Int(P)),$(Int(D)),$(Int(Q)))[$(Int(m))]")
    end

    if !isempty(fitted.xreg_cols)
        println(io, "  Exogenous: ", join(fitted.xreg_cols, ", "))
    end

    if !isnothing(fitted.fit.aic)
        println(io, "  AIC:  ", round(fitted.fit.aic, digits=4))
    end
    if !isnothing(fitted.fit.aicc)
        println(io, "  AICc: ", round(fitted.fit.aicc, digits=4))
    end
    if !isnothing(fitted.fit.bic)
        println(io, "  BIC:  ", round(fitted.fit.bic, digits=4))
    end
    if !isnothing(fitted.fit.sigma2)
        println(io, "  σ²:   ", round(fitted.fit.sigma2, digits=6))
    end

    println(io, "  n:    ", fitted.fit.nobs)
end
