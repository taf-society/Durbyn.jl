"""
    ARARMA Model Specification and Fitted Models

Provides `ArarmaSpec` for specifying ARARMA models using the grammar interface,
and `FittedArarma` for fitted ARARMA models.
"""


"""
    ArarmaSpec(formula::ModelFormula; max_ar_depth=26, max_lag=40, kwargs...)

Specify an ARARMA (ARAR + short-memory ARMA) model structure using Durbyn's forecasting grammar.

The ARARMA model first applies an adaptive AR prefilter (ARAR stage) to shorten memory,
then fits a short-memory ARMA(p,q) model on the prefiltered residuals.

# Arguments
- `formula::ModelFormula` - Formula from `@formula` macro specifying:
  - Target variable (LHS)
  - ARMA orders: `p()` and `q()`

# Keyword Arguments
**ARAR Stage Parameters:**
- `max_ar_depth::Int` - Maximum lag to consider when selecting best AR model (default: 26)
- `max_lag::Int` - Maximum lag for computing autocovariance sequence (default: 40)

**Additional kwargs:**
- For `auto_ararma` mode: `max_p`, `max_q`, `crit` (`:aic` or `:bic`)
- For `ararma` mode: `options` (NelderMeadOptions)

# ARMA Order Specification

**Search ranges** (uses `auto_ararma` for selection):
- `p()`, `q()` - Use defaults (p: 0-4, q: 0-2)
- `p(min, max)` - Search range for AR order
- `q(min, max)` - Search range for MA order

**Fixed values** (uses `ararma` directly):
- `p(value)` - Fixed AR order
- `q(value)` - Fixed MA order

# Examples
```julia
# Auto ARARMA with default search ranges
spec = ArarmaSpec(@formula(sales = p() + q()))

# Fixed ARARMA(1,2)
spec = ArarmaSpec(@formula(sales = p(1) + q(2)))

# Auto ARARMA with custom search ranges
spec = ArarmaSpec(@formula(sales = p(0,3) + q(0,2)))

# With custom ARAR parameters
spec = ArarmaSpec(
    @formula(sales = p(2) + q(1)),
    max_ar_depth = 20,
    max_lag = 30
)

# With additional options
spec = ArarmaSpec(
    @formula(sales = p() + q()),
    max_ar_depth = 20,
    max_lag = 40,
    crit = :bic
)
```

# See Also
- [`@formula`](@ref)
- [`p`](@ref), [`q`](@ref)
- [`fit`](@ref)
- [`ararma`](@ref)
- [`auto_ararma`](@ref)
"""
struct ArarmaSpec <: AbstractModelSpec
    formula::ModelFormula
    max_ar_depth::Int
    max_lag::Int
    options::Dict{Symbol, Any}

    function ArarmaSpec(formula::ModelFormula;
                        max_ar_depth::Int = 26,
                        max_lag::Int = 40,
                        kwargs...)

        for term in formula.terms
            if isa(term, ArimaOrderTerm)
                if term.term ∉ (:p, :q)
                    throw(ArgumentError(
                        "ARARMA only supports p() and q() terms. " *
                        "Found: $(term.term)(). " *
                        "ARARMA handles differencing internally through the ARAR stage."
                    ))
                end
            elseif isa(term, VarTerm)
                throw(ArgumentError(
                    "ARARMA does not support exogenous regressors. " *
                    "Found variable: :$(term.name). " *
                    "Use ArimaSpec for ARIMAX models with exogenous variables."
                ))
            elseif isa(term, AutoVarTerm)
                throw(ArgumentError(
                    "ARARMA does not support automatic exogenous variable selection. " *
                    "Use ArimaSpec with auto() for ARIMAX models."
                ))
            elseif !(term isa ArimaOrderTerm)
                throw(ArgumentError(
                    "Unsupported term type in ARARMA formula: $(typeof(term)). " *
                    "ARARMA only supports p() and q() terms."
                ))
            end
        end

        max_ar_depth >= 4 ||
            throw(ArgumentError("max_ar_depth must be at least 4, got $(max_ar_depth)"))
        max_lag >= 0 ||
            throw(ArgumentError("max_lag must be non-negative, got $(max_lag)"))

        new(formula, max_ar_depth, max_lag, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedArarma

A fitted ARARMA model containing the specification, fitted parameters,
and metadata needed for forecasting.

# Fields
- `spec::ArarmaSpec` - Original specification
- `fit::Any` - Fitted ARARMA model (ArarmaModel)
- `target_col::Symbol` - Name of target variable
- `data_schema::Dict{Symbol, Type}` - Column types for validation

# Examples
```julia
spec = ArarmaSpec(@formula(sales = p() + q()))
fitted = fit(spec, data)

# Access underlying ARARMA fit
fitted.fit.ar_order       # AR order
fitted.fit.ma_order       # MA order
fitted.fit.best_lag       # Selected AR lags
fitted.fit.lag_phi        # Lag-selection AR coefficients (4 terms)
fitted.fit.arma_phi       # ARMA-stage AR coefficients (p terms)
fitted.fit.arma_theta     # ARMA-stage MA coefficients (q terms)
fitted.fit.sigma2         # Residual variance
fitted.fit.aic            # AIC
fitted.fit.bic            # BIC

# Generate forecasts
fc = forecast(fitted, h = 12)
```

# See Also
- [`ArarmaSpec`](@ref)
- [`fit`](@ref)
- [`forecast`](@ref)
"""
struct FittedArarma <: AbstractFittedModel
    spec::ArarmaSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}

    function FittedArarma(spec::ArarmaSpec,
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

"""
    extract_metrics(model::FittedArarma)

Extract fit quality metrics from a fitted ARARMA model.
"""
function extract_metrics(model::FittedArarma)
    metrics = Dict{Symbol, Float64}()

    if !isnothing(model.fit.aic)
        metrics[:aic] = model.fit.aic
    end
    if !isnothing(model.fit.bic)
        metrics[:bic] = model.fit.bic
    end
    if !isnothing(model.fit.sigma2)
        metrics[:sigma2] = model.fit.sigma2
    end
    if !isnothing(model.fit.loglik)
        metrics[:loglik] = model.fit.loglik
    end

    return metrics
end

function Base.show(io::IO, spec::ArarmaSpec)
    print(io, "ArarmaSpec: ")
    print(io, spec.formula)
    print(io, ", max_ar_depth = ", spec.max_ar_depth)
    print(io, ", max_lag = ", spec.max_lag)
end

function Base.show(io::IO, fitted::FittedArarma)
    p = fitted.fit.ar_order
    q = fitted.fit.ma_order

    model_str = "ARARMA($(p),$(q))"
    print(io, "FittedArarma: ", model_str)

    if !isnothing(fitted.fit.aic)
        print(io, ", AIC = ", round(fitted.fit.aic, digits=2))
    end
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedArarma)
    p = fitted.fit.ar_order
    q = fitted.fit.ma_order

    println(io, "FittedArarma")
    println(io, "  Model: ARARMA($(p),$(q))")
    println(io, "  Selected AR lags: ", fitted.fit.best_lag)
    println(io, "  Lag-selection AR coefficients: ", round.(fitted.fit.lag_phi, digits=4))

    if !isempty(fitted.fit.arma_phi)
        println(io, "  ARMA AR coefficients (ϕ): ", round.(fitted.fit.arma_phi, digits=4))
    end
    if !isempty(fitted.fit.arma_theta)
        println(io, "  ARMA MA coefficients (θ): ", round.(fitted.fit.arma_theta, digits=4))
    end

    if !isnothing(fitted.fit.aic)
        println(io, "  AIC:  ", round(fitted.fit.aic, digits=4))
    end
    if !isnothing(fitted.fit.bic)
        println(io, "  BIC:  ", round(fitted.fit.bic, digits=4))
    end
    if !isnothing(fitted.fit.sigma2)
        println(io, "  σ²:   ", round(fitted.fit.sigma2, digits=6))
    end
    if !isnothing(fitted.fit.loglik)
        println(io, "  Log-likelihood: ", round(fitted.fit.loglik, digits=4))
    end

    println(io, "  n:    ", length(fitted.fit.y_original))
end
