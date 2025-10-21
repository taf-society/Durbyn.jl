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

function Base.show(io::IO, formula::ModelFormula)
    print(io, "$(formula.target) = ")
    for (i, term) in enumerate(formula.terms)
        print(io, term)
        if i < length(formula.terms)
            print(io, " + ")
        end
    end
end
