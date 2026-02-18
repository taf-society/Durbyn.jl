"""
    TBATS formula interface

This file provides the formula-based interface for TBATS, allowing users to specify
TBATS model parameters using Durbyn's forecasting grammar.

TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and
Seasonal components) extends BATS by using Fourier representation for seasonal
components, enabling non-integer seasonal periods and efficient handling of long cycles.

# Examples
```julia
# Data as NamedTuple
data = (
    sales = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
)

# Basic TBATS with defaults (automatic component selection)
fit = @formula(sales = tbats()) |> f -> tbats(f, data)

# TBATS with non-integer seasonal period (52.18 weeks per year)
fit = @formula(sales = tbats(m=52.18)) |> f -> tbats(f, data)

# TBATS with multiple seasonal periods
fit = @formula(sales = tbats(m=[7, 365.25])) |> f -> tbats(f, data)

# TBATS with explicit Fourier orders
fit = @formula(sales = tbats(m=[7, 365.25], k=[3, 10])) |> f -> tbats(f, data)

# TBATS with specific components
fit = @formula(sales = tbats(m=52.18, use_box_cox=true, use_trend=true)) |> f -> tbats(f, data)

# Using a Tables.jl table source
using DataFrames
df = DataFrame(data)
fit = @formula(sales = tbats(m=52.18)) |> f -> tbats(f, df)
```
"""

"""
    tbats(formula::ModelFormula, data; kwargs...)

Fit a TBATS model using Durbyn's forecasting grammar.

This method allows you to specify TBATS model parameters using a declarative formula syntax.
TBATS uses Fourier-based seasonal representation, enabling non-integer seasonal periods
and efficient handling of very long seasonal cycles.

# Arguments
- `formula::ModelFormula` - Formula created with grammar syntax (e.g., `@formula(y = tbats())`)
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.) containing the target variable

# Formula Terms
**TBATS Specification:**
- `tbats()` - Use default parameters (automatic component selection)
- `tbats(m=value)` - Specify seasonal period (Real) or periods (Vector{<:Real}). `seasonal_periods` also accepted as alias.
  - Can be non-integer, e.g., `52.18` for weekly data with yearly seasonality
- `tbats(k=value)` - Specify Fourier order(s) per seasonal period (Int or Vector{Int})
  - Higher k captures more complex seasonal patterns
- `tbats(use_box_cox=value)` - Whether to use Box-Cox transformation (true/false/nothing for auto)
- `tbats(use_trend=value)` - Whether to include trend component (true/false/nothing for auto)
- `tbats(use_damped_trend=value)` - Whether to use damped trend (true/false/nothing for auto)
- `tbats(use_arma_errors=value)` - Whether to include ARMA errors (true/false/nothing for default)

# Keyword Arguments
All standard `tbats` kwargs are supported and will be passed through to the underlying function:
- `bc_lower::Real=0.0` - Lower bound for Box-Cox lambda search
- `bc_upper::Real=1.0` - Upper bound for Box-Cox lambda search
- `biasadj::Bool=false` - Request bias-adjusted inverse Box-Cox transformation
- `model=nothing` - Previous TBATS model to reuse specification from

# Returns
`TBATSModel` object containing the fitted model

# Examples
```julia
using Durbyn

# Create sample data
data = (sales = randn(200),)

# Basic TBATS with defaults
fit = @formula(sales = tbats()) |> f -> tbats(f, data)

# TBATS with non-integer seasonal period (weekly data, yearly seasonality)
fit = @formula(sales = tbats(m=52.18)) |> f -> tbats(f, data)

# TBATS with multiple seasonal periods (daily + yearly)
fit = @formula(sales = tbats(m=[7, 365.25])) |> f -> tbats(f, data)

# TBATS with explicit Fourier orders
fit = @formula(sales = tbats(m=[7, 365.25], k=[3, 10])) |> f -> tbats(f, data)

# Dual calendar effects (Gregorian + Hijri)
fit = @formula(sales = tbats(m=[365.25, 354.37])) |> f -> tbats(f, data)

# TBATS with Box-Cox and trend
fit = @formula(sales = tbats(m=52.18, use_box_cox=true, use_trend=true)) |> f -> tbats(f, data)

# TBATS with all options specified
fit = @formula(sales = tbats(m=[7, 365.25], k=[3, 10],
                             use_box_cox=true, use_trend=true,
                             use_damped_trend=false, use_arma_errors=true)) |> f -> tbats(f, data)

# With additional kwargs
fit = @formula(sales = tbats(m=52.18)) |> f -> tbats(f, data, bc_lower=0.0, bc_upper=1.5)

# Generate forecasts
fc = forecast(fit, h=12)
```

# See Also
- [`tbats`](@ref) - Traditional parameter-based interface
- [`bats`](@ref) - BATS model (integer seasonal periods only)
- [`@formula`](@ref) - Macro for creating formulas
- [`forecast`](@ref) - Generate forecasts from fitted model
"""
function tbats(formula::ModelFormula, data; kwargs...)
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

    tbats_term = _extract_single_term(formula, TbatsTerm)

    tbats_args = Dict{Symbol, Any}()

    # Convert seasonal_periods to the format expected by tbats()
    if !isnothing(tbats_term.seasonal_periods)
        if tbats_term.seasonal_periods isa Real
            m = [tbats_term.seasonal_periods]
        else
            m = collect(Float64, tbats_term.seasonal_periods)
        end
    else
        m = nothing
    end

    # Handle k (Fourier orders) - stored in tbats_args if provided
    if !isnothing(tbats_term.k)
        if tbats_term.k isa Int
            tbats_args[:k] = [tbats_term.k]
        else
            tbats_args[:k] = collect(tbats_term.k)
        end
    end

    if !isnothing(tbats_term.use_box_cox)
        tbats_args[:use_box_cox] = tbats_term.use_box_cox
    end

    if !isnothing(tbats_term.use_trend)
        tbats_args[:use_trend] = tbats_term.use_trend
    end

    if !isnothing(tbats_term.use_damped_trend)
        tbats_args[:use_damped_trend] = tbats_term.use_damped_trend
    end

    if !isnothing(tbats_term.use_arma_errors)
        tbats_args[:use_arma_errors] = tbats_term.use_arma_errors
    end

    merge!(tbats_args, Dict{Symbol, Any}(kwargs))

    return tbats(y, m; pairs(tbats_args)...)
end
