"""
    auto_arima formula interface

This file provides the formula-based interface for auto_arima, allowing users to specify
ARIMA model orders and exogenous variables using Durbyn's forecasting grammar.

# Examples
```julia
# Data as NamedTuple
data = (
    sales = [100.0, 110.0, 120.0, 130.0],
    temperature = [20.0, 25.0, 22.0, 24.0],
    promotion = [0, 1, 0, 1]
)

# Pure ARIMA
fit = auto_arima(:sales = p(1,2) + d(1) + q(2,3), data, 12)

# SARIMA
fit = auto_arima(:sales = p(1,2) + d(1) + q(2,3) + P(0,1) + D(1) + Q(0,1), data, 12)

# ARIMAX with exogenous variables
fit = auto_arima(:sales = p(1,2) + q(2,3) + :temperature + :promotion, data, 12)

# Using a Tables.jl table source
tbl = Tables.table(data)
fit = auto_arima(:sales = p(1) + d(1) + q(1), tbl, 12)
```
"""

"""
    auto_arima(formula::ModelFormula, data, m::Int; xreg=nothing, kwargs...)

Fit an ARIMA/SARIMA model using Durbyn's forecasting grammar.

**Automatic Model Selection:**
- If ANY order term specifies a range (e.g., `p(1,3)`), uses `auto_arima` for model selection
- If ALL orders are fixed (e.g., `p(1) + q(2)`), uses `arima` directly for efficiency

This method allows you to specify ARIMA model orders and exogenous variables
using a declarative formula syntax.

# Arguments
- `formula::ModelFormula` - Formula created with grammar syntax (e.g., `@formula(y = p(1,2) + q(2,3))`)
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.) containing target and exogenous variables
- `m::Int` - Seasonal period (e.g., 12 for monthly, 4 for quarterly, 1 for non-seasonal)

# Formula Terms
**ARIMA Order Terms:**
- `p(min, max)` or `p(value)` - Non-seasonal AR order (range → search, fixed → use value)
- `d(min, max)` or `d(value)` - Non-seasonal differencing order
- `q(min, max)` or `q(value)` - Non-seasonal MA order (range → search, fixed → use value)
- `P(min, max)` or `P(value)` - Seasonal AR order (range → search, fixed → use value)
- `D(min, max)` or `D(value)` - Seasonal differencing order
- `Q(min, max)` or `Q(value)` - Seasonal MA order (range → search, fixed → use value)

**Exogenous Variables:**
- `varname` - Include variable from data as exogenous regressor (no `:` needed in @formula)

# Keyword Arguments
- For `auto_arima` mode: All standard `auto_arima` kwargs supported
- For `arima` mode: All standard `arima` kwargs supported
- `xreg` - Built from formula variables (or use kwarg for backward compatibility)

# Returns
`ArimaFit` object containing the fitted model

# Examples
```julia
using Durbyn

# Create sample data
data = (
    sales = randn(120),
    temperature = randn(120),
    promotion = rand(0:1, 120)
)

# MODEL SELECTION (uses auto_arima)
# Any term with range triggers auto_arima
fit = @formula(sales = p(1, 2) + d(1) + q(2, 3)) |> f -> auto_arima(f, data, 12)

# FIXED MODEL (uses arima directly - faster!)
# All terms fixed → direct fit
fit = @formula(sales = p(1) + d(1) + q(2)) |> f -> auto_arima(f, data, 12)

# SARIMAX with fixed orders (uses arima)
fit = @formula(sales = p(2) + d(1) + q(1) + P(1) + D(1) + Q(1) + temperature) |>
      f -> auto_arima(f, data, 12)

# Mixed: some ranges, some fixed (uses auto_arima)
fit = @formula(sales = p(1,3) + d(1) + q(2)) |> f -> auto_arima(f, data, 12)

# Let auto_arima determine d and D automatically
fit = @formula(sales = p(1, 3) + q(1, 2)) |> f -> auto_arima(f, data, 12)
```

# See Also
- [`auto_arima`](@ref) - Traditional parameter-based interface for model selection
- [`arima`](@ref) - Direct ARIMA fitting with specified orders
- [`p`](@ref), [`d`](@ref), [`q`](@ref), [`P`](@ref), [`D`](@ref), [`Q`](@ref) - Grammar functions
- [`@formula`](@ref) - Macro for creating formulas
"""
function auto_arima(formula::ModelFormula, data, m::Int; xreg=nothing, kwargs...)
    
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

    
    arima_terms = filter(t -> isa(t, ArimaOrderTerm), formula.terms)
    var_terms = filter(t -> isa(t, VarTerm), formula.terms)

    
    use_auto_arima = isempty(arima_terms) || any(t -> t.min != t.max, arima_terms)

    
    formula_xreg = nothing
    if !isempty(var_terms)
        
        if !isnothing(xreg)
            @warn "Both formula variables and xreg= kwarg specified. Formula variables take precedence."
        end

        
        for vt in var_terms
            if !haskey(tbl, vt.name)
                available_cols = join(keys(tbl), ", ")
                throw(ArgumentError(
                    "Exogenous variable ':$(vt.name)' not found in data. " *
                    "Available columns: $(available_cols)"
                ))
            end
        end

        
        var_names = [String(vt.name) for vt in var_terms]
        var_data = [tbl[vt.name] for vt in var_terms]

        
        for (i, vd) in enumerate(var_data)
            if !(vd isa AbstractVector)
                throw(ArgumentError(
                    "Exogenous variable ':$(var_terms[i].name)' must be a vector, got $(typeof(vd))"
                ))
            end
        end

        
        n = length(y)
        for (i, vd) in enumerate(var_data)
            if length(vd) != n
                throw(ArgumentError(
                    "Exogenous variable ':$(var_terms[i].name)' has length $(length(vd)), " *
                    "but target has length $(n). All variables must have the same length."
                ))
            end
        end

        
        xreg_matrix = hcat(var_data...)
        formula_xreg = NamedMatrix(xreg_matrix, var_names)
    else
        
        formula_xreg = xreg
    end

    
    compiled = compile_arima_formula(ModelFormula(target, arima_terms))
    
    min_p, max_p = get(compiled, :p, (2, 5))
    min_q, max_q = get(compiled, :q, (2, 5))
    min_P, max_P = get(compiled, :P, (1, 2))
    min_Q, max_Q = get(compiled, :Q, (1, 2))

    
    d_spec = get(compiled, :d, nothing)
    D_spec = get(compiled, :D, nothing)

    
    d_value = nothing
    if !isnothing(d_spec)
        min_d, max_d = d_spec
        if min_d == max_d
            d_value = min_d
        else
            @warn "Differencing order d should typically be fixed, not a range. " *
                  "Using d=$(min_d). Specify d=$(min_d) for clarity."
            d_value = min_d
        end
    end

    
    D_value = nothing
    if !isnothing(D_spec)
        min_D, max_D = D_spec
        if min_D == max_D
            D_value = min_D
        else
            @warn "Seasonal differencing order D should typically be fixed, not a range. " *
                  "Using D=$(min_D). Specify D=$(min_D) for clarity."
            D_value = min_D
        end
    end

    
    kwargs_dict = Dict{Symbol, Any}(kwargs)

    if use_auto_arima
        
        for key in [:max_p, :max_q, :max_P, :max_Q, :start_p, :start_q, :start_P, :start_Q, :d, :D]
            if haskey(kwargs_dict, key)
                @warn "Keyword argument '$(key)' is ignored when using formula interface. " *
                      "The formula specification takes precedence."
                delete!(kwargs_dict, key)
            end
        end

        
        arima_args = Dict{Symbol, Any}(
            :max_p => max_p,
            :max_q => max_q,
            :max_P => max_P,
            :max_Q => max_Q,
            :start_p => min_p,
            :start_q => min_q,
            :start_P => min_P,
            :start_Q => min_Q,
        )

        
        if !isnothing(d_value)
            arima_args[:d] = d_value
        end

        
        if !isnothing(D_value)
            arima_args[:D] = D_value
        end
        
        if !isnothing(formula_xreg)
            arima_args[:xreg] = formula_xreg
        end

        
        merge!(arima_args, kwargs_dict)

        
        return auto_arima(y, m; pairs(arima_args)...)

    else
        
        p_val = min_p
        q_val = min_q
        P_val = min_P
        Q_val = min_Q

        
        d_val = something(d_value, 0)
        D_val = something(D_value, 0)

        
        for key in [:max_p, :max_q, :max_P, :max_Q, :start_p, :start_q, :start_P, :start_Q,
                    :stationary, :seasonal_test, :test, :test_args, :seasonal_test_args,
                    :allowdrift, :allowmean, :stepwise, :nmodels, :trace, :approximation,
                    :max_order, :max_d, :max_D]
            if haskey(kwargs_dict, key)
                @warn "Keyword argument '$(key)' is for auto_arima and is ignored when all orders are fixed."
                delete!(kwargs_dict, key)
            end
        end

        
        order = PDQ(p_val, d_val, q_val)
        seasonal_order = PDQ(P_val, D_val, Q_val)

        
        if !isnothing(formula_xreg)
            kwargs_dict[:xreg] = formula_xreg
        end

        return arima_rjh(y, m; order=order, seasonal=seasonal_order, pairs(kwargs_dict)...)
    end
end
