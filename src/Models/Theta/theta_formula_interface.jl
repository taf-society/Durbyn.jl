"""
    Theta formula interface

This file provides the formula-based interface for Theta models, allowing users to specify
Theta model parameters using Durbyn's forecasting grammar.

# Examples
```julia
# Data as NamedTuple
data = (
    sales = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, ...],
)

# Auto-select best Theta variant (tries STM, OTM, DSTM, DOTM)
fit = @formula(sales = theta()) |> f -> theta(f, data, m=12)

# Specific model variant
fit = @formula(sales = theta(model=:OTM)) |> f -> theta(f, data, m=12)

# With fixed alpha
fit = @formula(sales = theta(model=:OTM, alpha=0.3)) |> f -> theta(f, data, m=12)

# Force additive decomposition
fit = @formula(sales = theta(decomposition="additive")) |> f -> theta(f, data, m=12)

fit = @formula(sales = theta()) |> f -> theta(f, df, m=12)
```
"""


"""
    theta(formula::ModelFormula, data; m::Int=1, kwargs...)

Fit a Theta model using Durbyn's forecasting grammar.

This method allows you to specify Theta model parameters using a declarative formula syntax.
The function automatically chooses between `auto_theta` (tries all variants) and `theta`
(specific variant) based on the formula specification.

# Arguments
- `formula::ModelFormula` - Formula created with grammar syntax (e.g., `@formula(y = theta())`)
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.) containing the target variable
- `m::Int=1` - Seasonal period (1 for non-seasonal data)

# Formula Terms
**Theta Specification:**
- `theta()` - Auto-select best model variant (tries STM, OTM, DSTM, DOTM)
- `theta(model=:STM)` - Simple Theta Model (θ=2 fixed, α optimized)
- `theta(model=:OTM)` - Optimized Theta Model (both θ and α optimized)
- `theta(model=:DSTM)` - Dynamic Simple Theta Model (θ=2, dynamic trend)
- `theta(model=:DOTM)` - Dynamic Optimized Theta Model (dynamic + optimized)
- `theta(alpha=value)` - Fix smoothing parameter (0 < α < 1)
- `theta(theta_param=value)` - Fix theta parameter (≥ 1, ignored for STM/DSTM)
- `theta(decomposition="multiplicative"|"additive")` - Seasonal decomposition type
- `theta(nmse=value)` - Steps for multi-step MSE calculation (1-30)

# Behavior
- If `model=nothing` (default) → calls `auto_theta` which tries all 4 variants and selects best by MSE
- If `model=:STM/:OTM/:DSTM/:DOTM` → calls `theta` with the specific model type

# Keyword Arguments
All standard `theta`/`auto_theta` kwargs are supported and will be passed through.

# Returns
`ThetaFit` object containing the fitted model

# Examples
```julia
using Durbyn

data = (sales = sin.(2π .* (1:120) ./ 12) .+ collect(1:120) .* 0.05 .+ randn(120) .* 0.1,)

fit = @formula(sales = theta()) |> f -> theta(f, data, m=12)

fit = @formula(sales = theta(model=:OTM)) |> f -> theta(f, data, m=12)
fit = @formula(sales = theta(model=:STM)) |> f -> theta(f, data, m=12)

fit = @formula(sales = theta(model=:OTM, alpha=0.3)) |> f -> theta(f, data, m=12)

fit = @formula(sales = theta(decomposition="additive")) |> f -> theta(f, data, m=12)

fit = @formula(sales = theta(model=:DOTM, decomposition="multiplicative", nmse=5)) |>
      f -> theta(f, data, m=12)

fc = forecast(fit, 24)
```

# See Also
- [`theta`](@ref) - Traditional parameter-based interface for specific model
- [`auto_theta`](@ref) - Automatic model selection interface
- [`@formula`](@ref) - Macro for creating formulas
- [`forecast`](@ref) - Generate forecasts from fitted model
"""
function theta(formula::ModelFormula, data; m::Int=1, kwargs...)
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

    theta_term = _extract_single_term(formula, ThetaTerm)

    theta_args = Dict{Symbol, Any}()

    if !isnothing(theta_term.alpha)
        theta_args[:alpha] = theta_term.alpha
    end

    if !isnothing(theta_term.theta)
        theta_args[:theta_param] = theta_term.theta
    end

    if !isnothing(theta_term.decomposition_type)
        theta_args[:decomposition_type] = theta_term.decomposition_type
    end

    if !isnothing(theta_term.nmse)
        theta_args[:nmse] = theta_term.nmse
    end

    merge!(theta_args, Dict{Symbol, Any}(kwargs))

    if isnothing(theta_term.model_type)
        return auto_theta(y, m; theta_args...)
    else
        model_enum = _symbol_to_theta_model_type(theta_term.model_type)
        return theta(y, m; model_type=model_enum, theta_args...)
    end
end

"""
    _symbol_to_theta_model_type(sym::Symbol) -> ThetaModelType

Convert a Symbol (:STM, :OTM, :DSTM, :DOTM) to the corresponding ThetaModelType enum value.
"""
function _symbol_to_theta_model_type(sym::Symbol)
    if sym === :STM
        return STM
    elseif sym === :OTM
        return OTM
    elseif sym === :DSTM
        return DSTM
    elseif sym === :DOTM
        return DOTM
    else
        throw(ArgumentError("Unknown Theta model type: :$(sym). Valid types: :STM, :OTM, :DSTM, :DOTM"))
    end
end
