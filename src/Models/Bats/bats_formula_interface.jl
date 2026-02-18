"""
    BATS formula interface

This file provides the formula-based interface for BATS, allowing users to specify
BATS model parameters using Durbyn's forecasting grammar.

# Examples
```julia
# Data as NamedTuple
data = (
    sales = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
)

# Basic BATS with defaults (automatic component selection)
fit = @formula(sales = bats()) |> f -> bats(f, data)

# BATS with monthly seasonality
fit = @formula(sales = bats(m=12)) |> f -> bats(f, data)

# BATS with multiple seasonal periods
fit = @formula(sales = bats(m=[24, 168])) |> f -> bats(f, data)

# BATS with specific components
fit = @formula(sales = bats(m=12, use_box_cox=true, use_trend=true)) |> f -> bats(f, data)

# Using a Tables.jl table source
using DataFrames
df = DataFrame(data)
fit = @formula(sales = bats(m=12)) |> f -> bats(f, df)
```
"""

"""
    bats(formula::ModelFormula, data; kwargs...)

Fit a BATS model using Durbyn's forecasting grammar.

This method allows you to specify BATS model parameters using a declarative formula syntax.

# Arguments
- `formula::ModelFormula` - Formula created with grammar syntax (e.g., `@formula(y = bats())`)
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.) containing the target variable

# Formula Terms
**BATS Specification:**
- `bats()` - Use default parameters (automatic component selection)
- `bats(m=value)` - Specify seasonal period (Int) or periods (Vector{Int}). `seasonal_periods` also accepted as alias.
- `bats(use_box_cox=value)` - Whether to use Box-Cox transformation (true/false/nothing for auto)
- `bats(use_trend=value)` - Whether to include trend component (true/false/nothing for auto)
- `bats(use_damped_trend=value)` - Whether to use damped trend (true/false/nothing for auto)
- `bats(use_arma_errors=value)` - Whether to include ARMA errors (true/false/nothing for default)

# Keyword Arguments
All standard `bats` kwargs are supported and will be passed through to the underlying function:
- `bc_lower::Real=0.0` - Lower bound for Box-Cox lambda search
- `bc_upper::Real=1.0` - Upper bound for Box-Cox lambda search
- `biasadj::Bool=false` - Request bias-adjusted inverse Box-Cox transformation
- `model=nothing` - Previous BATS model to reuse specification from

# Returns
`BATSModel` object containing the fitted model

# Examples
```julia
using Durbyn

# Create sample data
data = (sales = randn(120),)

# Basic BATS with defaults
fit = @formula(sales = bats()) |> f -> bats(f, data)

# BATS with monthly seasonality
fit = @formula(sales = bats(m=12)) |> f -> bats(f, data)

# BATS with multiple seasonal periods (e.g., hourly data with daily and weekly patterns)
fit = @formula(sales = bats(m=[24, 168])) |> f -> bats(f, data)

# BATS with Box-Cox and trend
fit = @formula(sales = bats(m=12, use_box_cox=true, use_trend=true)) |> f -> bats(f, data)

# BATS with all options specified
fit = @formula(sales = bats(m=12, use_box_cox=true, use_trend=true,
                           use_damped_trend=false, use_arma_errors=true)) |> f -> bats(f, data)

# With additional kwargs
fit = @formula(sales = bats(m=12)) |> f -> bats(f, data, bc_lower=0.0, bc_upper=1.5)

# Generate forecasts
fc = forecast(fit, h=12)
```

# See Also
- [`bats`](@ref) - Traditional parameter-based interface
- [`@formula`](@ref) - Macro for creating formulas
- [`forecast`](@ref) - Generate forecasts from fitted model
"""
function bats(formula::ModelFormula, data; kwargs...)
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

    bats_term = _extract_single_term(formula, BatsTerm)

    bats_args = Dict{Symbol, Any}()

    if !isnothing(bats_term.seasonal_periods)
        if bats_term.seasonal_periods isa Int
            m = [bats_term.seasonal_periods]
        else
            m = bats_term.seasonal_periods
        end
    else
        m = nothing
    end

    if !isnothing(bats_term.use_box_cox)
        bats_args[:use_box_cox] = bats_term.use_box_cox
    end

    if !isnothing(bats_term.use_trend)
        bats_args[:use_trend] = bats_term.use_trend
    end

    if !isnothing(bats_term.use_damped_trend)
        bats_args[:use_damped_trend] = bats_term.use_damped_trend
    end

    if !isnothing(bats_term.use_arma_errors)
        bats_args[:use_arma_errors] = bats_term.use_arma_errors
    end

    merge!(bats_args, Dict{Symbol, Any}(kwargs))

    return bats(y, m; pairs(bats_args)...)
end
