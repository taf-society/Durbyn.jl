"""
    Durbyn Forecasting Grammar

This module provides a Domain-Specific Language (DSL) for specifying forecasting models
in a declarative way. The grammar is designed to be extensible and support various
forecasting methods including ARIMA, exponential smoothing, and ML models.

# Current Implementation
- ARIMA order specification: p(), q(), P(), Q(), d(), D()

# Planned Extensions
- Feature engineering: dow(), month(), woy(), lag(), ma(), rollsum(), etc.
- ML model specification
- Custom transformations
"""

# ============================================================================
# Base Grammar Types
# ============================================================================

"""
    AbstractTerm

Base type for all grammar terms in the Durbyn forecasting DSL.
All specific term types should inherit from this.
"""
abstract type AbstractTerm end

"""
    ArimaOrderTerm <: AbstractTerm

Represents an ARIMA model order term (p, d, q, P, D, or Q) with search range.

# Fields
- `term::Symbol` - The term type (:p, :d, :q, :P, :D, or :Q)
- `min::Int` - Minimum order to search
- `max::Int` - Maximum order to search

# Constructor
    ArimaOrderTerm(term::Symbol, min::Int, max::Int)

Creates an ARIMA order term with validation.

# Examples
```julia
# These are typically created via the p(), d(), q(), P(), D(), Q() functions
ArimaOrderTerm(:p, 1, 2)  # Search p from 1 to 2
ArimaOrderTerm(:d, 1, 1)  # Fixed d = 1
ArimaOrderTerm(:q, 2, 2)  # Fixed q = 2
```
"""
struct ArimaOrderTerm <: AbstractTerm
    term::Symbol
    min::Int
    max::Int

    function ArimaOrderTerm(term::Symbol, min::Int, max::Int)
        term ∈ (:p, :d, :q, :P, :D, :Q) ||
            throw(ArgumentError("ARIMA term must be one of :p, :d, :q, :P, :D, :Q, got :$(term)"))
        min >= 0 ||
            throw(ArgumentError("min must be non-negative, got $(min)"))
        max >= min ||
            throw(ArgumentError("max must be >= min, got max=$(max) < min=$(min)"))
        new(term, min, max)
    end
end

# ============================================================================
# ETS Grammar Terms
# ============================================================================

"""
    EtsComponentTerm <: AbstractTerm

Represents a component (error, trend, seasonal) in an ETS specification.

# Fields
- `component::Symbol` - One of `:error`, `:trend`, `:seasonal`
- `code::String` - Component code (`"A"`, `"M"`, `"N"`, `"Z"`)
"""
struct EtsComponentTerm <: AbstractTerm
    component::Symbol
    code::String

    function EtsComponentTerm(component::Symbol, code::AbstractString)
        component ∈ (:error, :trend, :seasonal) ||
            throw(ArgumentError("ETS component must be :error, :trend, or :seasonal, got :$(component)"))
        normalized = uppercase(code)
        new(component, normalized)
    end
end

"""
    EtsDriftTerm <: AbstractTerm

Represents the damping/drift setting for an ETS specification.
"""
struct EtsDriftTerm <: AbstractTerm
    damped::Union{Bool, Nothing}
end

"""
    SesTerm <: AbstractTerm

Sentinel term for Simple Exponential Smoothing specifications.
"""
struct SesTerm <: AbstractTerm
end

"""
    HoltTerm <: AbstractTerm

Represents Holt's method options within a formula (`holt()`).
"""
struct HoltTerm <: AbstractTerm
    damped::Union{Bool, Nothing}
    exponential::Bool
end

"""
    HoltWintersTerm <: AbstractTerm

Represents Holt-Winters seasonal exponential smoothing options (`hw()`/`holt_winters()`).
"""
struct HoltWintersTerm <: AbstractTerm
    seasonal::String
    damped::Union{Bool, Nothing}
    exponential::Bool
end

"""
    CrostonTerm <: AbstractTerm

Sentinel term for Croston's intermittent-demand model.
"""
struct CrostonTerm <: AbstractTerm
end

"""
    ArarTerm <: AbstractTerm

Represents ARAR (AutoRegressive with Adaptive Reduction) model options within a formula.

# Fields
- `max_ar_depth::Union{Int, Nothing}` - Maximum AR depth to consider
- `max_lag::Union{Int, Nothing}` - Maximum lag for autocovariance computation
"""
struct ArarTerm <: AbstractTerm
    max_ar_depth::Union{Int, Nothing}
    max_lag::Union{Int, Nothing}
end

"""
    VarTerm <: AbstractTerm

Represents an exogenous variable to be used as a regressor in the model.

# Fields
- `name::Symbol` - Name of the variable (column) to include

# Examples
```julia
# Typically created by using Symbol directly in formula
VarTerm(:temperature)
VarTerm(:promotion)

# Used in formulas like:
# :sales = p(1,2) + q(2,3) + :temperature + :promotion
```
"""
struct VarTerm <: AbstractTerm
    name::Symbol
end

"""
    AutoVarTerm <: AbstractTerm

Sentinel term indicating that all eligible columns (excluding date, group, and target)
should be treated as exogenous regressors automatically.

Created when a formula uses `.` on the right-hand side, e.g. `@formula(value = .)`.
"""
struct AutoVarTerm <: AbstractTerm
end

"""
    auto()

Shorthand for automatic exogenous-variable selection (see [`AutoVarTerm`](@ref)).
Use in formulas as `@formula(y = auto())` (optionally combined with ARIMA terms).
"""
auto() = AutoVarTerm()

# ============================================================================
# ARIMA Grammar Functions
# ============================================================================

"""
    p()
    p(value::Int)
    p(min::Int, max::Int)

Specify non-seasonal AR order(s) for ARIMA model selection.

# Arguments
- No arguments - Use auto_arima defaults (start_p=2, max_p=5)
- `value::Int` - Fixed AR order (searches only this value)
- `min::Int, max::Int` - Range of AR orders to search [min, max]

# Returns
`ArimaOrderTerm` representing the non-seasonal AR component

# Examples
```julia
p()        # Use defaults: search p ∈ {2, 3, 4, 5}
p(1)       # Fix p = 1
p(0, 3)    # Search p ∈ {0, 1, 2, 3}
p(2, 5)    # Search p ∈ {2, 3, 4, 5}
```

# Notes
In ARIMA(p,d,q) notation, `p` is the order of the autoregressive component.
"""
p() = ArimaOrderTerm(:p, 2, 5)  # auto_arima defaults
p(value::Int) = ArimaOrderTerm(:p, value, value)
p(min::Int, max::Int) = ArimaOrderTerm(:p, min, max)

"""
    q()
    q(value::Int)
    q(min::Int, max::Int)

Specify non-seasonal MA order(s) for ARIMA model selection.

# Arguments
- No arguments - Use auto_arima defaults (start_q=2, max_q=5)
- `value::Int` - Fixed MA order (searches only this value)
- `min::Int, max::Int` - Range of MA orders to search [min, max]

# Returns
`ArimaOrderTerm` representing the non-seasonal MA component

# Examples
```julia
q()        # Use defaults: search q ∈ {2, 3, 4, 5}
q(2)       # Fix q = 2
q(1, 3)    # Search q ∈ {1, 2, 3}
q(0, 2)    # Search q ∈ {0, 1, 2}
```

# Notes
In ARIMA(p,d,q) notation, `q` is the order of the moving average component.
"""
q() = ArimaOrderTerm(:q, 2, 5)  # auto_arima defaults
q(value::Int) = ArimaOrderTerm(:q, value, value)
q(min::Int, max::Int) = ArimaOrderTerm(:q, min, max)

"""
    d()
    d(value::Int)
    d(min::Int, max::Int)

Specify non-seasonal differencing order(s) for ARIMA model selection.

# Arguments
- No arguments - Let auto_arima determine via unit root tests (returns nothing)
- `value::Int` - Fixed differencing order
- `min::Int, max::Int` - Range of differencing orders (not recommended)

# Returns
- `nothing` when called with no arguments (let auto_arima determine)
- `ArimaOrderTerm` representing the non-seasonal differencing component otherwise

# Examples
```julia
d()        # Auto-determine via tests (KPSS, ADF, PP) - returns nothing
d(1)       # Fix d = 1 (first differencing)
d(0, 2)    # Search d ∈ {0, 1, 2} (not typical)
```

# Notes
In ARIMA(p,d,q) notation, `d` is the degree of non-seasonal differencing.
Typically d ∈ {0, 1, 2}. When d() is called without arguments or omitted entirely,
auto_arima uses unit root tests to determine the appropriate order.
"""
d() = nothing
d(value::Int) = ArimaOrderTerm(:d, value, value)
d(min::Int, max::Int) = ArimaOrderTerm(:d, min, max)

"""
    P()
    P(value::Int)
    P(min::Int, max::Int)

Specify seasonal AR order(s) for SARIMA model selection.

# Arguments
- No arguments - Use auto_arima defaults (start_P=1, max_P=2)
- `value::Int` - Fixed seasonal AR order (searches only this value)
- `min::Int, max::Int` - Range of seasonal AR orders to search [min, max]

# Returns
`ArimaOrderTerm` representing the seasonal AR component

# Examples
```julia
P()        # Use defaults: search P ∈ {1, 2}
P(1)       # Fix P = 1
P(0, 2)    # Search P ∈ {0, 1, 2}
```

# Notes
In ARIMA(p,d,q)(P,D,Q)[m] notation, `P` is the order of the seasonal autoregressive component.
"""
P() = ArimaOrderTerm(:P, 1, 2)  # auto_arima defaults
P(value::Int) = ArimaOrderTerm(:P, value, value)
P(min::Int, max::Int) = ArimaOrderTerm(:P, min, max)

"""
    Q()
    Q(value::Int)
    Q(min::Int, max::Int)

Specify seasonal MA order(s) for SARIMA model selection.

# Arguments
- No arguments - Use auto_arima defaults (start_Q=1, max_Q=2)
- `value::Int` - Fixed seasonal MA order (searches only this value)
- `min::Int, max::Int` - Range of seasonal MA orders to search [min, max]

# Returns
`ArimaOrderTerm` representing the seasonal MA component

# Examples
```julia
Q()        # Use defaults: search Q ∈ {1, 2}
Q(1)       # Fix Q = 1
Q(0, 2)    # Search Q ∈ {0, 1, 2}
```

# Notes
In ARIMA(p,d,q)(P,D,Q)[m] notation, `Q` is the order of the seasonal moving average component.
"""
Q() = ArimaOrderTerm(:Q, 1, 2)  # auto_arima defaults
Q(value::Int) = ArimaOrderTerm(:Q, value, value)
Q(min::Int, max::Int) = ArimaOrderTerm(:Q, min, max)

"""
    D()
    D(value::Int)
    D(min::Int, max::Int)

Specify seasonal differencing order(s) for SARIMA model selection.

# Arguments
- No arguments - Let auto_arima determine via seasonal strength tests (returns nothing)
- `value::Int` - Fixed seasonal differencing order
- `min::Int, max::Int` - Range of seasonal differencing orders (not recommended)

# Returns
- `nothing` when called with no arguments (let auto_arima determine)
- `ArimaOrderTerm` representing the seasonal differencing component otherwise

# Examples
```julia
D()        # Auto-determine via seasonal strength tests - returns nothing
D(1)       # Fix D = 1 (seasonal differencing)
D(0, 1)    # Search D ∈ {0, 1}
```

# Notes
In ARIMA(p,d,q)(P,D,Q)[m] notation, `D` is the degree of seasonal differencing.
Typically D ∈ {0, 1}. D > 1 is rarely recommended.
"""
D() = nothing
D(value::Int) = ArimaOrderTerm(:D, value, value)
D(min::Int, max::Int) = ArimaOrderTerm(:D, min, max)

# ============================================================================
# ETS Grammar Functions
# ============================================================================

const _ETS_ERROR_CODES = Set(["A", "M", "Z"])
const _ETS_TREND_CODES = Set(["N", "A", "M", "Z"])
const _ETS_SEASON_CODES = Set(["N", "A", "M", "Z"])

function _validate_ets_code(component::Symbol, code::AbstractString)
    normalized = uppercase(code)
    if component === :error
        normalized ∈ _ETS_ERROR_CODES ||
            throw(ArgumentError("Invalid ETS error code '$(code)'. Use \"A\", \"M\", or \"Z\"."))
    elseif component === :trend
        normalized ∈ _ETS_TREND_CODES ||
            throw(ArgumentError("Invalid ETS trend code '$(code)'. Use \"N\", \"A\", \"M\", or \"Z\"."))
    elseif component === :seasonal
        normalized ∈ _ETS_SEASON_CODES ||
            throw(ArgumentError("Invalid ETS seasonal code '$(code)'. Use \"N\", \"A\", \"M\", or \"Z\"."))
    else
        throw(ArgumentError("Unknown ETS component :$(component)."))
    end
    return normalized
end

"""
    ses()

Specify Simple Exponential Smoothing in a model formula.
"""
ses() = SesTerm()

function _validate_bool_or_auto(name::Symbol, value)
    if !(value === nothing || value isa Bool)
        throw(ArgumentError("Keyword $(name) must be Bool or nothing, got $(typeof(value))"))
    end
    return value
end

"""
    holt(; damped=nothing, exponential=false)

Specify Holt's linear trend method. `damped` may be `true`, `false`, or `nothing`
(use default). `exponential=true` requests an exponential trend.
"""
function holt(; damped=nothing, exponential::Bool=false)
    _validate_bool_or_auto(:damped, damped)
    return HoltTerm(damped, exponential)
end

_normalize_hw_seasonal(x::Symbol) = _normalize_hw_seasonal(string(x))
function _normalize_hw_seasonal(x::AbstractString)
    val = lowercase(x)
    val in ("additive", "multiplicative") ||
        throw(ArgumentError("seasonal must be \"additive\" or \"multiplicative\", got $(x)"))
    return val
end

"""
    hw(; seasonal="additive", damped=nothing, exponential=false)
    holt_winters(; seasonal="additive", damped=nothing, exponential=false)

Specify Holt-Winters seasonal exponential smoothing within a formula.
"""
function hw(; seasonal::Union{AbstractString,Symbol}="additive",
             damped=nothing,
             exponential::Bool=false)
    _validate_bool_or_auto(:damped, damped)
    seasonal_norm = _normalize_hw_seasonal(seasonal)
    if exponential && seasonal_norm == "additive"
        throw(ArgumentError("exponential trend is only supported with multiplicative seasonality in Holt-Winters."))
    end
    return HoltWintersTerm(seasonal_norm, damped, exponential)
end
holt_winters(; kwargs...) = hw(; kwargs...)

"""
    croston()

Specify Croston's intermittent demand model in a formula.
"""
croston() = CrostonTerm()

"""
    arar(; max_ar_depth=nothing, max_lag=nothing)

Specify ARAR (AutoRegressive with Adaptive Reduction) model in a formula.

# Arguments
- `max_ar_depth::Union{Int, Nothing}=nothing` - Maximum lag to consider when selecting the best 4-lag AR model (must be at least 4)
- `max_lag::Union{Int, Nothing}=nothing` - Maximum lag for computing autocovariance sequence

# Examples
```julia
# Basic ARAR with defaults
@formula(y = arar())

# ARAR with custom max_ar_depth
@formula(y = arar(max_ar_depth=15))

# ARAR with both parameters
@formula(y = arar(max_ar_depth=20, max_lag=20))
```
"""
function arar(; max_ar_depth::Union{Int, Nothing}=nothing, max_lag::Union{Int, Nothing}=nothing)
    if !isnothing(max_ar_depth) && max_ar_depth < 4
        throw(ArgumentError("max_ar_depth must be at least 4, got $(max_ar_depth)"))
    end
    if !isnothing(max_lag) && max_lag < 0
        throw(ArgumentError("max_lag must be non-negative, got $(max_lag)"))
    end
    return ArarTerm(max_ar_depth, max_lag)
end

"""
    e(code::AbstractString = "Z")

Specify the error component in an ETS model.

- `"A"`: additive error
- `"M"`: multiplicative error
- `"Z"`: automatically select
"""
e(code::AbstractString = "Z") = EtsComponentTerm(:error, _validate_ets_code(:error, code))

"""
    t(code::AbstractString = "Z")

Specify the trend component in an ETS model.

- `"N"`: no trend
- `"A"`: additive trend
- `"M"`: multiplicative trend
- `"Z"`: automatically select
"""
t(code::AbstractString = "Z") = EtsComponentTerm(:trend, _validate_ets_code(:trend, code))

"""
    s(code::AbstractString = "Z")

Specify the seasonal component in an ETS model.

- `"N"`: no seasonality
- `"A"`: additive seasonality
- `"M"`: multiplicative seasonality
- `"Z"`: automatically select
"""
s(code::AbstractString = "Z") = EtsComponentTerm(:seasonal, _validate_ets_code(:seasonal, code))

"""
    drift(v=true)

Control the damping/drift behaviour of an ETS trend component.

- `true`: include a damped trend (φ estimated)
- `false`: forbid damping (standard trend)
- `nothing`/`drift(:auto)`: allow automatic selection
"""
drift() = EtsDriftTerm(true)
drift(flag::Bool) = EtsDriftTerm(flag)
drift(::Nothing) = EtsDriftTerm(nothing)
function drift(mode::Symbol)
    mode === :auto ||
        throw(ArgumentError("Unsupported drift mode ':$(mode)'. Use true, false, nothing, or :auto."))
    return EtsDriftTerm(nothing)
end
function drift(mode::AbstractString)
    upper = uppercase(mode)
    if upper == "AUTO"
        return EtsDriftTerm(nothing)
    elseif upper == "TRUE"
        return EtsDriftTerm(true)
    elseif upper == "FALSE"
        return EtsDriftTerm(false)
    else
        throw(ArgumentError("Unsupported drift mode \"$(mode)\". Use \"auto\", \"true\", or \"false\"."))
    end
end

# ============================================================================
# Formula Structure
# ============================================================================

"""
    ModelFormula

Represents a complete model formula in Durbyn's forecasting grammar.

# Fields
- `target::Symbol` - Target variable name (left-hand side)
- `terms::Vector{AbstractTerm}` - Model specification terms (right-hand side)

# Examples
```julia
# Typically created via the = operator:
# target = p(1,2) + q(2,3)
ModelFormula(:y, [ArimaOrderTerm(:p, 1, 2), ArimaOrderTerm(:q, 2, 3)])
```
"""
struct ModelFormula
    target::Symbol
    terms::Vector{AbstractTerm}
end

# ============================================================================
# Operator Overloading for Formula Construction
# ============================================================================

"""
    +(term1::AbstractTerm, term2::AbstractTerm)
    +(term::AbstractTerm, var::Symbol)
    +(var::Symbol, term::AbstractTerm)

Combine grammar terms and/or variable symbols with the + operator.

# Returns
Vector of terms that can be further combined

# Examples
```julia
# ARIMA terms only
p(1,2) + q(2,3)                    # [ArimaOrderTerm(:p,...), ArimaOrderTerm(:q,...)]

# ARIMAX with exogenous variables
p(1,2) + q(2,3) + :temperature     # Adds VarTerm(:temperature)
p(1,2) + :temp + q(2,3)            # Variable can be anywhere
:promotion + p(1,2) + :temperature # Multiple variables

# Complex formulas
p(1,2) + d(1) + q(2,3) + P(0,1) + D(1) + Q(0,1) + :temperature + :promotion
```
"""
# Combine two terms
Base.:+(t1::AbstractTerm, t2::AbstractTerm) = AbstractTerm[t1, t2]

# Add term to existing vector
Base.:+(terms::Vector{<:AbstractTerm}, t::AbstractTerm) = push!(copy(terms), t)
Base.:+(t::AbstractTerm, terms::Vector{<:AbstractTerm}) = pushfirst!(copy(terms), t)

# Combine term with Symbol (create VarTerm)
Base.:+(t::AbstractTerm, s::Symbol) = AbstractTerm[t, VarTerm(s)]
Base.:+(s::Symbol, t::AbstractTerm) = AbstractTerm[VarTerm(s), t]

# Add Symbol to existing vector
Base.:+(terms::Vector{<:AbstractTerm}, s::Symbol) = push!(copy(terms), VarTerm(s))
Base.:+(s::Symbol, terms::Vector{<:AbstractTerm}) = pushfirst!(copy(terms), VarTerm(s))

# Combine two Symbols
Base.:+(s1::Symbol, s2::Symbol) = AbstractTerm[VarTerm(s1), VarTerm(s2)]

# Handle nothing (from d() and D() with no arguments)
# nothing + term → just the term
Base.:+(::Nothing, t::AbstractTerm) = t
Base.:+(t::AbstractTerm, ::Nothing) = t
Base.:+(::Nothing, terms::Vector{<:AbstractTerm}) = terms
Base.:+(terms::Vector{<:AbstractTerm}, ::Nothing) = terms
Base.:+(::Nothing, ::Nothing) = nothing
Base.:+(::Nothing, s::Symbol) = VarTerm(s)
Base.:+(s::Symbol, ::Nothing) = VarTerm(s)

"""
    @formula(expr)

Create a model formula from an expression using the `=` syntax.

# Syntax
```julia
@formula(target = terms)
```

where `terms` can be any combination of ARIMA order terms and variables joined with `+`.

# Examples
```julia
# Single term
formula = @formula(y = p(1, 2))

# Multiple terms
formula = @formula(y = p(1, 2) + q(2, 3))

# SARIMA specification
formula = @formula(sales = p(1, 2) + q(2, 3) + P(0, 1) + Q(0, 1))

# With exogenous variables
formula = @formula(sales = p(1, 2) + q(2, 3) + temperature + promotion)
```
"""
macro formula(expr)
    if expr.head != :(=)
        error("@formula requires an assignment expression: @formula(target = terms)")
    end

    target = expr.args[1]
    rhs = expr.args[2]

    # Convert variable names to Symbols with : prefix for VarTerm
    function process_rhs(ex)
        if ex isa Symbol
            if ex === Symbol(".")
                return :($(esc(:AutoVarTerm))())
            end
            # Plain symbol becomes VarTerm - must escape it
            return :($(esc(:VarTerm))($(QuoteNode(ex))))
        elseif ex isa Expr && ex.head == :call
            if ex.args[1] == :+
                # Process + operator recursively
                return Expr(:call, :+, [process_rhs(arg) for arg in ex.args[2:end]]...)
            else
                # Function calls (p(), q(), etc.) pass through - escape them
                return esc(ex)
            end
        else
            return esc(ex)
        end
    end

    processed_rhs = process_rhs(rhs)

    # Always wrap in a vector and create ModelFormula
    # The + operator already creates vectors, so handle both cases
    # Filter out nothing values (from d() and D() with no arguments)
    result_expr = quote
        local rhs_result = $processed_rhs
        local terms_vec = if rhs_result isa Vector
            rhs_result
        elseif rhs_result === nothing
            $(esc(:AbstractTerm))[]  # Empty vector if only nothing
        else
            [rhs_result]
        end
        $(esc(:ModelFormula))($(QuoteNode(target)), terms_vec)
    end

    return result_expr
end

# ============================================================================
# Formula Compilation
# ============================================================================

"""
    compile_arima_formula(formula::ModelFormula) -> Dict{Symbol, Tuple{Int,Int}}

Compile a model formula into ARIMA-specific parameter ranges.

# Arguments
- `formula::ModelFormula` - Formula containing ARIMA order terms

# Returns
Dictionary mapping term symbols to (min, max) tuples:
- `:p => (min_p, max_p)` - Non-seasonal AR
- `:d => (d, d)` - Non-seasonal differencing (typically fixed)
- `:q => (min_q, max_q)` - Non-seasonal MA
- `:P => (min_P, max_P)` - Seasonal AR
- `:D => (D, D)` - Seasonal differencing (typically fixed)
- `:Q => (min_Q, max_Q)` - Seasonal MA

# Errors
- Throws `ArgumentError` if duplicate terms are found
- Throws `ArgumentError` if non-ARIMA terms are present

# Examples
```julia
formula = :y = p(1, 2) + d(1) + q(2, 3)
compiled = compile_arima_formula(formula)
# Returns: Dict(:p => (1, 2), :d => (1, 1), :q => (2, 3))
```
"""
function compile_arima_formula(formula::ModelFormula)
    result = Dict{Symbol, Tuple{Int, Int}}()

    for term in formula.terms
        if !isa(term, ArimaOrderTerm)
            throw(ArgumentError("Expected ArimaOrderTerm, got $(typeof(term)). " *
                              "Only p(), d(), q(), P(), D(), Q() terms are supported for ARIMA models."))
        end

        if haskey(result, term.term)
            throw(ArgumentError("Duplicate term '$(term.term)' in formula. " *
                              "Each of p, d, q, P, D, Q can appear only once."))
        end

        result[term.term] = (term.min, term.max)
    end

    return result
end

"""
    compile_ets_formula(formula::ModelFormula)

Extract ETS components (error, trend, seasonal, drift) from a model formula.

# Returns
Named tuple `(error, trend, seasonal, damped)` with component codes and
optional damping directive.
"""
function compile_ets_formula(formula::ModelFormula)
    error_code = "Z"
    trend_code = "Z"
    seasonal_code = "Z"
    damped::Union{Bool, Nothing} = nothing
    error_set = false
    trend_set = false
    seasonal_set = false

    for term in formula.terms
        if term isa EtsComponentTerm
            if term.component === :error
                error_set &&
                    throw(ArgumentError("Multiple error terms detected in ETS formula."))
                error_code = _validate_ets_code(:error, term.code)
                error_set = true
            elseif term.component === :trend
                trend_set &&
                    throw(ArgumentError("Multiple trend terms detected in ETS formula."))
                trend_code = _validate_ets_code(:trend, term.code)
                trend_set = true
            elseif term.component === :seasonal
                seasonal_set &&
                    throw(ArgumentError("Multiple seasonal terms detected in ETS formula."))
                seasonal_code = _validate_ets_code(:seasonal, term.code)
                seasonal_set = true
            else
                throw(ArgumentError("Unsupported ETS component :$(term.component)."))
            end
        elseif term isa EtsDriftTerm
            !isnothing(damped) &&
                throw(ArgumentError("Multiple drift() directives detected in ETS formula."))
            damped = term.damped
        elseif term isa VarTerm || term isa AutoVarTerm || term isa ArimaOrderTerm
            throw(ArgumentError("ETS formulas cannot include ARIMA terms or exogenous regressors."))
        elseif term === nothing
            continue
        else
            throw(ArgumentError("Unsupported term type $(typeof(term)) in ETS formula."))
        end
    end

    return (error = error_code,
            trend = trend_code,
            seasonal = seasonal_code,
            damped = damped)
end

function _extract_single_term(formula::ModelFormula, ::Type{T}) where {T<:AbstractTerm}
    selected = nothing
    for term in formula.terms
        if term isa T
            isnothing(selected) ||
                throw(ArgumentError("Formula may contain only one $(T) term."))
            selected = term
        elseif term isa EtsComponentTerm || term isa EtsDriftTerm || term isa ArimaOrderTerm ||
               term isa VarTerm || term isa AutoVarTerm || term isa ArarTerm
            throw(ArgumentError("Formula term $(term) is not compatible with $(T)."))
        elseif term !== nothing
            throw(ArgumentError("Unsupported term $(term) in formula for $(T)."))
        end
    end
    isnothing(selected) &&
        throw(ArgumentError("Formula must include $(T) to build this specification."))
    return selected
end

# ============================================================================
# Pretty Printing
# ============================================================================

function Base.show(io::IO, term::ArimaOrderTerm)
    if term.min == term.max
        print(io, "$(term.term)($(term.min))")
    else
        print(io, "$(term.term)($(term.min), $(term.max))")
    end
end

function Base.show(io::IO, term::VarTerm)
    print(io, ":$(term.name)")
end

function Base.show(io::IO, ::AutoVarTerm)
    print(io, ".")
end

function Base.show(io::IO, term::EtsComponentTerm)
    component_label = term.component === :error ? "e" :
                      term.component === :trend ? "t" : "s"
    print(io, "$(component_label)(\"$(term.code)\")")
end

function Base.show(io::IO, term::EtsDriftTerm)
    if term.damped === true
        print(io, "drift()")
    elseif term.damped === false
        print(io, "drift(false)")
    else
        print(io, "drift(:auto)")
    end
end

function Base.show(io::IO, ::SesTerm)
    print(io, "ses()")
end

function Base.show(io::IO, term::HoltTerm)
    args = String[]
    if !isnothing(term.damped)
        push!(args, "damped=$(term.damped)")
    end
    if term.exponential
        push!(args, "exponential=true")
    end
    if isempty(args)
        print(io, "holt()")
    else
        print(io, "holt(", join(args, ", "), ")")
    end
end

function Base.show(io::IO, term::HoltWintersTerm)
    args = ["seasonal=\"$(term.seasonal)\""]
    if !isnothing(term.damped)
        push!(args, "damped=$(term.damped)")
    end
    if term.exponential
        push!(args, "exponential=true")
    end
    print(io, "hw(", join(args, ", "), ")")
end

function Base.show(io::IO, ::CrostonTerm)
    print(io, "croston()")
end

function Base.show(io::IO, term::ArarTerm)
    args = String[]
    if !isnothing(term.max_ar_depth)
        push!(args, "max_ar_depth=$(term.max_ar_depth)")
    end
    if !isnothing(term.max_lag)
        push!(args, "max_lag=$(term.max_lag)")
    end
    if isempty(args)
        print(io, "arar()")
    else
        print(io, "arar(", join(args, ", "), ")")
    end
end

function Base.show(io::IO, formula::ModelFormula)
    print(io, "$(formula.target) = ")
    for (i, term) in enumerate(formula.terms)
        print(io, term)
        if i < length(formula.terms)
            print(io, " + ")
        end
    end
end
