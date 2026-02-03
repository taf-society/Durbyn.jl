"""
    ARARMA formula interface

This file provides the formula-based interface for ARARMA, allowing users to specify
ARMA model orders using Durbyn's forecasting grammar.

# Examples
```julia
# Data as NamedTuple
data = (
    sales = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
)

# Fixed ARARMA(1,2)
fit = ararma(:sales = p(1) + q(2), data)

# Auto ARARMA with search ranges
fit = ararma(:sales = p(0,3) + q(0,2), data)

# Auto ARARMA with defaults
fit = ararma(:sales = p() + q(), data)

# Using a Tables.jl table source
using DataFrames
df = DataFrame(data)
fit = ararma(:sales = p(1) + q(1), df)
```
"""

"""
    ararma(formula::ModelFormula, data; max_ar_depth=26, max_lag=40, kwargs...)

Fit an ARARMA model using Durbyn's forecasting grammar.

**Automatic Model Selection:**
- If ANY order term specifies a range (e.g., `p(1,3)` or `q()`), uses `auto_ararma` for model selection
- If ALL orders are fixed (e.g., `p(1) + q(2)`), uses `ararma` directly for efficiency

This method allows you to specify ARMA model orders using a declarative formula syntax.
The ARARMA model first applies an adaptive AR prefilter (ARAR stage) to shorten memory,
then fits a short-memory ARMA(p,q) model.

# Arguments
- `formula::ModelFormula` - Formula created with grammar syntax (e.g., `@formula(y = p(1,3) + q(1,2))`)
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.) containing target variable

# Formula Terms
**ARMA Order Terms:**
- `p(min, max)` or `p(value)` - AR order (range → search, fixed → use value)
- `q(min, max)` or `q(value)` - MA order (range → search, fixed → use value)
- `p()` - Use default search range (0-4)
- `q()` - Use default search range (0-2)

# Keyword Arguments
**ARAR Stage Parameters:**
- `max_ar_depth::Int` - Maximum lag to consider when selecting best AR model (default: 26)
- `max_lag::Int` - Maximum lag for computing autocovariance sequence (default: 40)

**For `auto_ararma` mode (when ranges specified):**
- `max_p::Int` - Maximum AR order to search
- `max_q::Int` - Maximum MA order to search
- `crit::Symbol` - Information criterion (`:aic` or `:bic`, default: `:aic`)

**For `ararma` mode (when orders fixed):**
- `p::Int` - AR order (overridden by formula)
- `q::Int` - MA order (overridden by formula)
- `options::NelderMeadOptions` - Optimizer options

# Returns
`ArarmaModel` object containing the fitted model

# Examples
```julia
using Durbyn
using Durbyn.Ararma

# Create sample data
data = (sales = randn(120),)

# MODEL SELECTION (uses auto_ararma)
# Any term with range triggers auto_ararma
fit = @formula(sales = p(0, 3) + q(0, 2)) |> f -> ararma(f, data)

# With default search ranges
fit = @formula(sales = p() + q()) |> f -> ararma(f, data)

# FIXED MODEL (uses ararma directly - faster!)
# All terms fixed → direct fit
fit = @formula(sales = p(1) + q(2)) |> f -> ararma(f, data)

# With custom ARAR parameters
fit = @formula(sales = p(2) + q(1)) |> f -> ararma(f, data, max_ar_depth=20, max_lag=30)

# Mixed: some ranges, some fixed (uses auto_ararma)
fit = @formula(sales = p(1,3) + q(2)) |> f -> ararma(f, data)

# Auto selection with custom criterion
fit = @formula(sales = p() + q()) |> f -> ararma(f, data, crit=:bic)
```

# See Also
- [`ararma`](@ref) - Direct ARARMA fitting with specified orders
- [`auto_ararma`](@ref) - Traditional parameter-based interface for model selection
- [`p`](@ref), [`q`](@ref) - Grammar functions
- [`@formula`](@ref) - Macro for creating formulas
"""
function ararma(formula::ModelFormula, data; max_ar_depth::Int=26, max_lag::Int=40, kwargs...)

    if !Tables.istable(data)
        throw(ArgumentError("Input must be a Tables.jl-compatible table"))
    end
    tbl = Tables.columntable(data)

    target = formula.target

    if !haskey(tbl, target)
        available_cols = join(keys(tbl), ", ")
        throw(ArgumentError(
            "Target variable ':$(target)' not found in data. " *
            "Available columns: $(available_cols)"
        ))
    end

    y = tbl[target]

    if !(y isa AbstractVector)
        throw(ArgumentError("Target variable ':$(target)' must be a vector, got $(typeof(y))"))
    end

    arma_terms = filter(t -> isa(t, ArimaOrderTerm) && t.term ∈ (:p, :q), formula.terms)

    for term in formula.terms
        if isa(term, ArimaOrderTerm)
            if term.term ∉ (:p, :q)
                throw(ArgumentError(
                    "ARARMA only supports p() and q() terms. " *
                    "Found unsupported term: $(term.term)(). " *
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

    use_auto = isempty(arma_terms) || any(t -> t.min != t.max, arma_terms)

    compiled = Dict{Symbol, Tuple{Int, Int}}()
    for term in arma_terms
        if haskey(compiled, term.term)
            throw(ArgumentError("Duplicate term '$(term.term)' in formula. " *
                              "Each of p and q can appear only once."))
        end
        compiled[term.term] = (term.min, term.max)
    end

    # Track which terms were explicitly specified in formula
    p_in_formula = haskey(compiled, :p)
    q_in_formula = haskey(compiled, :q)

    kwargs_dict = Dict{Symbol, Any}(kwargs)
    kwargs_dict[:max_ar_depth] = max_ar_depth
    kwargs_dict[:max_lag] = max_lag

    # Determine bounds: formula takes precedence, then kwargs, then defaults
    if p_in_formula
        min_p, max_p = compiled[:p]
        # Warn if kwargs try to override formula-specified p bounds
        for key in [:p, :min_p, :max_p]
            if haskey(kwargs_dict, key)
                @warn "Keyword argument '$(key)' is ignored because p() is specified in formula."
                delete!(kwargs_dict, key)
            end
        end
    else
        # No p() in formula - use kwargs if provided, else defaults
        min_p = get(kwargs_dict, :min_p, 0)
        max_p = get(kwargs_dict, :max_p, 4)
        delete!(kwargs_dict, :min_p)
        delete!(kwargs_dict, :max_p)
        # Warn if :p kwarg is present (it's ignored in auto mode; use min_p/max_p instead)
        if haskey(kwargs_dict, :p)
            @warn "Keyword argument 'p' is ignored in auto mode. Use 'min_p'/'max_p' or add p() to formula."
            delete!(kwargs_dict, :p)
        end
    end

    if q_in_formula
        min_q, max_q = compiled[:q]
        # Warn if kwargs try to override formula-specified q bounds
        for key in [:q, :min_q, :max_q]
            if haskey(kwargs_dict, key)
                @warn "Keyword argument '$(key)' is ignored because q() is specified in formula."
                delete!(kwargs_dict, key)
            end
        end
    else
        # No q() in formula - use kwargs if provided, else defaults
        min_q = get(kwargs_dict, :min_q, 0)
        max_q = get(kwargs_dict, :max_q, 2)
        delete!(kwargs_dict, :min_q)
        delete!(kwargs_dict, :max_q)
        # Warn if :q kwarg is present (it's ignored in auto mode; use min_q/max_q instead)
        if haskey(kwargs_dict, :q)
            @warn "Keyword argument 'q' is ignored in auto mode. Use 'min_q'/'max_q' or add q() to formula."
            delete!(kwargs_dict, :q)
        end
    end

    if use_auto

        ararma_args = Dict{Symbol, Any}(
            :min_p => min_p,
            :max_p => max_p,
            :min_q => min_q,
            :max_q => max_q,
        )

        merge!(ararma_args, kwargs_dict)

        return auto_ararma(y; pairs(ararma_args)...)

    else

        p_val = min_p
        q_val = min_q

        # Remove kwargs not accepted by ararma() in fixed-order mode
        for key in [:min_p, :max_p, :min_q, :max_q, :crit]
            if haskey(kwargs_dict, key)
                @warn "Keyword argument '$(key)' is for auto_ararma and is ignored when all orders are fixed."
                delete!(kwargs_dict, key)
            end
        end

        kwargs_dict[:p] = p_val
        kwargs_dict[:q] = q_val

        return ararma(y; pairs(kwargs_dict)...)
    end
end
