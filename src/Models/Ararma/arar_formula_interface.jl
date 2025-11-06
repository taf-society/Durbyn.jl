"""
    ARAR formula interface

This file provides the formula-based interface for ARAR, allowing users to specify
ARAR model parameters using Durbyn's forecasting grammar.

# Examples
```julia
# Data as NamedTuple
data = (
    sales = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
)

# Basic ARAR with defaults
fit = @formula(sales = arar()) |> f -> arar(f, data)

# ARAR with custom max_ar_depth
fit = @formula(sales = arar(max_ar_depth=15)) |> f -> arar(f, data)

# ARAR with both parameters
fit = @formula(sales = arar(max_ar_depth=20, max_lag=20)) |> f -> arar(f, data)

# Using a Tables.jl table source
using DataFrames
df = DataFrame(data)
fit = @formula(sales = arar()) |> f -> arar(f, df)
```
"""

"""
    arar(formula::ModelFormula, data; kwargs...)

Fit an ARAR model using Durbyn's forecasting grammar.

This method allows you to specify ARAR model parameters using a declarative formula syntax.

# Arguments
- `formula::ModelFormula` - Formula created with grammar syntax (e.g., `@formula(y = arar())`)
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.) containing the target variable

# Formula Terms
**ARAR Specification:**
- `arar()` - Use default parameters
- `arar(max_ar_depth=value)` - Specify maximum AR depth
- `arar(max_lag=value)` - Specify maximum lag for autocovariance
- `arar(max_ar_depth=value, max_lag=value)` - Specify both parameters

# Keyword Arguments
All standard `arar` kwargs are supported and will be passed through to the underlying function.

# Returns
`ARAR` object containing the fitted model

# Examples
```julia
using Durbyn

# Create sample data
data = (sales = randn(120),)

# Basic ARAR with defaults
fit = @formula(sales = arar()) |> f -> arar(f, data)

# ARAR with custom max_ar_depth
fit = @formula(sales = arar(max_ar_depth=15)) |> f -> arar(f, data)

# ARAR with both parameters
fit = @formula(sales = arar(max_ar_depth=20, max_lag=20)) |> f -> arar(f, data)

# Generate forecasts
fc = forecast(fit, h=12)
```

# See Also
- [`arar`](@ref) - Traditional parameter-based interface
- [`@formula`](@ref) - Macro for creating formulas
- [`forecast`](@ref) - Generate forecasts from fitted model
"""
function arar(formula::ModelFormula, data; kwargs...)
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
    
    arar_term = _extract_single_term(formula, ArarTerm)

    arar_args = Dict{Symbol, Any}()
    
    if !isnothing(arar_term.max_ar_depth)
        arar_args[:max_ar_depth] = arar_term.max_ar_depth
    end

    if !isnothing(arar_term.max_lag)
        arar_args[:max_lag] = arar_term.max_lag
    end
    
    merge!(arar_args, Dict{Symbol, Any}(kwargs))
    
    return arar(y; pairs(arar_args)...)
end
