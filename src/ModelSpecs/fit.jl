"""
    fit.jl

Implementation of fit() methods for model specifications.

This file provides the concrete implementations that connect model specs
to the underlying fitting functions (auto_arima, arima, etc.).
"""

import ..Generics: fit, forecast, fitted

import Tables

"""
    fit(spec::ArimaSpec, data; m=nothing, groupby=nothing, parallel=true, fail_fast=false, kwargs...)

Fit an ARIMA model specification to data (single series or grouped).

This method connects the declarative `ArimaSpec` to the underlying `auto_arima`
or `arima` fitting functions, handling data extraction and xreg preparation.

# Arguments
- `spec::ArimaSpec` - ARIMA model specification created with `@formula`
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.)

# Keyword Arguments
- `m::Union{Int, Nothing}` - Seasonal period (required if not specified in spec)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by for panel data
- `datecol::Union{Symbol, Nothing}` - Date/time column (excluded from automatic exogenous selection)
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)
- Additional kwargs passed to underlying `auto_arima` or `arima` function

# Returns
- If `groupby=nothing`: `FittedArima` - Single fitted model
- If `groupby` specified: `GroupedFittedModels` - Fitted models for each group

# Behavior
- If `m` not provided in spec or kwargs, raises error
- Calls existing `auto_arima(formula, data, m; ...)` which handles:
  - Smart routing (search vs fixed orders)
  - Exogenous variable extraction
  - Model selection or direct fitting
- Wraps result in `FittedArima` with metadata for forecasting

# Examples
```julia
# Single series fit
spec = ArimaSpec(@formula(sales = p() + q()))
fitted = fit(spec, data, m = 12)

# Grouped data fit
spec = ArimaSpec(@formula(sales = p(1) + d(1) + q(1)))
fitted = fit(spec, data, m = 7, groupby = [:product, :location])

# Sequential processing (no parallel)
fitted = fit(spec, data, m = 7, groupby = [:product], parallel = false)

# Stop on first error
fitted = fit(spec, data, m = 7, groupby = [:product], fail_fast = true)
```

# Error Handling
- Validates that `m` is specified (either in spec or kwargs)
- Validates target variable exists in data
- Validates exogenous variables exist in data
- For grouped data: catches errors per group and continues (unless fail_fast=true)

# See Also
- [`ArimaSpec`](@ref)
- [`GroupedFittedModels`](@ref)
- [`auto_arima`](@ref)
- [`forecast`](@ref)
"""
function fit(spec::ArimaSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)
             
    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m=m,
                           groupby=groupby,
                           datecol=datecol,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           kwargs...)
    end

    seasonal_period = if !isnothing(m)
        m
    elseif !isnothing(spec.m)
        spec.m
    else
        throw(ArgumentError(
            "Seasonal period 'm' must be specified either in ArimaSpec or as kwarg to fit(). " *
            "Example: fit(spec, data, m = 12) or ArimaSpec(..., m = 12)"
        ))
    end

    if seasonal_period < 1
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))
    end

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    if !haskey(tbl, target_col)
        available_cols = join(string.(keys(tbl)), ", ")
        throw(ArgumentError(
            "Target variable ':$(target_col)' not found in data. " *
            "Available columns: $(available_cols)"
        ))
    end

    target_data = tbl[target_col]
    if !(target_data isa AbstractVector)
        throw(ArgumentError(
            "Target variable ':$(target_col)' must be a vector, got $(typeof(target_data))"
        ))
    end
    n_rows = length(target_data)

    parent_mod = parentmodule(@__MODULE__)
    Utils_mod = getfield(parent_mod, :Utils)

    xreg_formula_cols = Symbol[]
    design_matrix, design_cols = _build_xreg_formula_matrix(spec, tbl, n_rows, Utils_mod)
    if !isnothing(design_matrix) && size(design_matrix.data, 2) > 0
        fit_options[:xreg] = design_matrix
        xreg_formula_cols = design_cols
    end

    auto_xreg_cols = Symbol[]
    if spec.auto_xreg
        haskey(fit_options, :xreg) && throw(ArgumentError(
            "Cannot supply xreg kwargs or xreg_formula when formula uses '.' for automatic exogenous selection."))

        exclude = _auto_exclusion_set(target_col, Symbol[], datecol)
        auto_xreg_cols, skipped = _collect_auto_xreg_columns(tbl, exclude, n_rows)
        _warn_skipped_auto(skipped)

        if isempty(auto_xreg_cols)
            @warn "Automatic exogenous selection requested ('.') but no eligible numeric columns were found after exclusions."
        else
            auto_matrix = _build_auto_xreg_matrix(tbl, auto_xreg_cols, n_rows, Utils_mod)
            fit_options[:xreg] = auto_matrix
        end
    end
    
    Arima_mod = getfield(parent_mod, :Arima)
    arima_fit = Arima_mod.auto_arima(spec.formula, data, seasonal_period; pairs(fit_options)...)

    var_terms = filter(t -> isa(t, VarTerm), spec.formula.terms)
    xreg_cols = Symbol[vt.name for vt in var_terms]
    if !isempty(xreg_formula_cols)
        append!(xreg_cols, xreg_formula_cols)
    end
    if spec.auto_xreg && !isempty(auto_xreg_cols)
        append!(xreg_cols, auto_xreg_cols)
    end
    if !isempty(xreg_cols)
        unique!(xreg_cols)
    end

    return FittedArima(
        spec,
        arima_fit,
        target_col,
        xreg_cols,
        tbl,
        seasonal_period
    )
end

"""
    fit(spec::EtsSpec, data; m=nothing, groupby=nothing, parallel=true, fail_fast=false, kwargs...)

Fit an ETS specification to data (single series or grouped).
"""
function fit(spec::EtsSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)
    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m = m,
                           groupby = groupby,
                           datecol = datecol,
                           parallel = parallel,
                           fail_fast = fail_fast,
                           kwargs...)
    end

    seasonal_period = isnothing(m) ? (isnothing(spec.m) ? 1 : spec.m) : m
    seasonal_period >= 1 ||
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    haskey(tbl, target_col) ||
        throw(ArgumentError("Target variable ':$(target_col)' not found in data."))

    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError("Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"))

    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"))

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))
    damped = haskey(fit_options, :damped) ? pop!(fit_options, :damped) : spec.damped
    if !(damped === nothing || damped isa Bool)
        throw(ArgumentError("damped option must be Bool or nothing, got $(typeof(damped))"))
    end

    parent_mod = parentmodule(@__MODULE__)
    Exp_mod = getfield(parent_mod, :ExponentialSmoothing)

    model_code = string(spec.components.error,
                        spec.components.trend,
                        spec.components.seasonal)

    ets_fit = Exp_mod.ets(target_vector,
                          seasonal_period,
                          model_code;
                          damped = damped,
                          pairs(fit_options)...)

    return FittedEts(spec, ets_fit, target_col, tbl, seasonal_period)
end

"""
    fit(spec::SesSpec, data; m=nothing, groupby=nothing, kwargs...)
"""
function fit(spec::SesSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)
    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m = m,
                           groupby = groupby,
                           datecol = datecol,
                           parallel = parallel,
                           fail_fast = fail_fast,
                           kwargs...)
    end

    seasonal_period = isnothing(m) ? (isnothing(spec.m) ? 1 : spec.m) : m
    seasonal_period >= 1 ||
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))

    tbl = Tables.columntable(data)
    target_col = spec.formula.target
    haskey(tbl, target_col) ||
        throw(ArgumentError("Target variable ':$(target_col)' not found in data."))
    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError("Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"))
    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"))

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))

    parent_mod = parentmodule(@__MODULE__)
    Exp_mod = getfield(parent_mod, :ExponentialSmoothing)

    ses_fit = Exp_mod.ses(target_vector, seasonal_period; pairs(fit_options)...)

    return FittedSes(spec, ses_fit, target_col, tbl, seasonal_period)
end

"""
    fit(spec::HoltSpec, data; ...)
"""
function fit(spec::HoltSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)
    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m = m,
                           groupby = groupby,
                           datecol = datecol,
                           parallel = parallel,
                           fail_fast = fail_fast,
                           kwargs...)
    end

    seasonal_period = isnothing(m) ? (isnothing(spec.m) ? 1 : spec.m) : m
    seasonal_period >= 1 ||
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))

    tbl = Tables.columntable(data)
    target_col = spec.formula.target
    haskey(tbl, target_col) ||
        throw(ArgumentError("Target variable ':$(target_col)' not found in data."))
    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError("Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"))
    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"))

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))
    damped = haskey(fit_options, :damped) ? pop!(fit_options, :damped) : spec.damped
    if !(damped === nothing || damped isa Bool)
        throw(ArgumentError("damped must be Bool or nothing, got $(typeof(damped))"))
    end
    exponential = haskey(fit_options, :exponential) ? pop!(fit_options, :exponential) : spec.exponential
    exponential isa Bool ||
        throw(ArgumentError("exponential must be Bool, got $(typeof(exponential))"))

    parent_mod = parentmodule(@__MODULE__)
    Exp_mod = getfield(parent_mod, :ExponentialSmoothing)

    holt_fit = if isnothing(damped)
        Exp_mod.holt(target_vector, seasonal_period;
                     exponential = exponential,
                     pairs(fit_options)...)
    else
        Exp_mod.holt(target_vector, seasonal_period;
                     damped = damped,
                     exponential = exponential,
                     pairs(fit_options)...)
    end

    return FittedHolt(spec, holt_fit, target_col, tbl, seasonal_period)
end

"""
    fit(spec::HoltWintersSpec, data; ...)
"""
function fit(spec::HoltWintersSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)
    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m = m,
                           groupby = groupby,
                           datecol = datecol,
                           parallel = parallel,
                           fail_fast = fail_fast,
                           kwargs...)
    end

    seasonal_period = isnothing(m) ? spec.m : m
    isnothing(seasonal_period) &&
        throw(ArgumentError(
            "Seasonal period 'm' must be specified for Holt-Winters. Provide it in the spec or as a kwarg."
        ))
    seasonal_period >= 1 ||
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))

    tbl = Tables.columntable(data)
    target_col = spec.formula.target
    haskey(tbl, target_col) ||
        throw(ArgumentError("Target variable ':$(target_col)' not found in data."))
    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError("Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"))
    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"))

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))
    seasonal = haskey(fit_options, :seasonal) ? pop!(fit_options, :seasonal) : spec.seasonal
    seasonal_str = lowercase(String(seasonal))
    seasonal_str in ("additive", "multiplicative") ||
        throw(ArgumentError("seasonal must be \"additive\" or \"multiplicative\", got $(seasonal)"))

    damped = haskey(fit_options, :damped) ? pop!(fit_options, :damped) : spec.damped
    if !(damped === nothing || damped isa Bool)
        throw(ArgumentError("damped must be Bool or nothing, got $(typeof(damped))"))
    end
    exponential = haskey(fit_options, :exponential) ? pop!(fit_options, :exponential) : spec.exponential
    exponential isa Bool ||
        throw(ArgumentError("exponential must be Bool, got $(typeof(exponential))"))
    if exponential && seasonal_str == "additive"
        throw(ArgumentError("exponential trend cannot be combined with additive seasonality."))
    end

    parent_mod = parentmodule(@__MODULE__)
    Exp_mod = getfield(parent_mod, :ExponentialSmoothing)

    hw_kwargs = Dict{Symbol, Any}(fit_options)
    hw_kwargs[:seasonal] = seasonal_str
    hw_kwargs[:exponential] = exponential
    if !isnothing(damped)
        hw_kwargs[:damped] = damped
    end

    hw_fit = Exp_mod.holt_winters(target_vector, seasonal_period; pairs(hw_kwargs)...)

    return FittedHoltWinters(spec, hw_fit, target_col, tbl, seasonal_period)
end

"""
    fit(spec::CrostonSpec, data; ...)
"""
function fit(spec::CrostonSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)
    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m = m,
                           groupby = groupby,
                           datecol = datecol,
                           parallel = parallel,
                           fail_fast = fail_fast,
                           kwargs...)
    end

    seasonal_period = isnothing(m) ? (isnothing(spec.m) ? 1 : spec.m) : m
    seasonal_period >= 1 ||
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))

    tbl = Tables.columntable(data)
    target_col = spec.formula.target
    haskey(tbl, target_col) ||
        throw(ArgumentError("Target variable ':$(target_col)' not found in data."))
    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError("Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"))
    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"))

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))

    parent_mod = parentmodule(@__MODULE__)

    croston_fit = if spec.method == "hyndman"
        Exp_mod = getfield(parent_mod, :ExponentialSmoothing)
        Exp_mod.croston(target_vector, seasonal_period; pairs(fit_options)...)
    else
        ID_mod = getfield(parent_mod, :IntermittentDemand)

        id_options = Dict{Symbol, Any}()
        id_options[:init_strategy] = something(spec.init_strategy, "mean")
        id_options[:number_of_params] = something(spec.number_of_params, 2)
        id_options[:cost_metric] = something(spec.cost_metric, "mar")
        id_options[:optimize_init] = something(spec.optimize_init, true)
        id_options[:rm_missing] = something(spec.rm_missing, false)

        merge!(id_options, fit_options)

        if spec.method == "classic"
            ID_mod.croston_classic(target_vector; pairs(id_options)...)
        elseif spec.method == "sba"
            ID_mod.croston_sba(target_vector; pairs(id_options)...)
        elseif spec.method == "sbj"
            ID_mod.croston_sbj(target_vector; pairs(id_options)...)
        else
            throw(ArgumentError("Unknown Croston method: $(spec.method)"))
        end
    end

    return FittedCroston(spec, croston_fit, target_col, tbl, seasonal_period)
end

"""
    forecast(fitted::FittedArima; h, level=[80, 95], newdata=nothing, kwargs...)

Generate forecasts from a fitted ARIMA model.

# Keyword Arguments
- `h::Int` - Forecast horizon (number of periods ahead)
- `level::Vector{<:Real}` - Confidence levels for prediction intervals (default [80, 95])
- `newdata` - Tables.jl-compatible data with exogenous variables for forecast period (optional)
- Additional kwargs passed to underlying `forecast()` (fan, bootstrap, etc.)

# Returns
`Forecast` object with mean, lower, and upper prediction intervals

# Examples
```julia
# Basic forecast
spec = ArimaSpec(@formula(sales = p() + q()))
fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 12)

# Forecast with exogenous variables
spec = ArimaSpec(@formula(sales = p() + q() + temperature))
fitted = fit(spec, data, m = 12)
newdata = (temperature = rand(12),)
fc = forecast(fitted, h = 12, newdata = newdata)

# Custom confidence levels
fc = forecast(fitted, h = 12, level = [90, 99])

# Fan plot
fc = forecast(fitted, h = 12, fan = true)
```

# See Also
- [`ArimaSpec`](@ref)
- [`fit`](@ref)
- [`Forecast`](@ref)
"""
function forecast(fitted::FittedArima; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    parent_mod = parentmodule(@__MODULE__)
    Utils_mod = getfield(parent_mod, :Utils)
    Arima_mod = getfield(parent_mod, :Arima)

    var_terms = filter(t -> isa(t, VarTerm), fitted.spec.formula.terms)
    var_cols = Symbol[vt.name for vt in var_terms]
    formula_cols = _xreg_formula_symbols(fitted.spec.xreg_formula)

    train_xreg = fitted.fit.xreg
    train_named = train_xreg isa Utils_mod.NamedMatrix ? train_xreg : nothing
    train_colnames = isnothing(train_named) ? String[] : copy(train_named.colnames)
    uses_train_xreg = !isnothing(train_named) && size(train_named.data, 2) > 0

    auto_cols = fitted.spec.auto_xreg ? (isempty(train_colnames) ? fitted.xreg_cols : Symbol.(train_colnames)) : Symbol[]
    has_auto = fitted.spec.auto_xreg && !isempty(auto_cols)
    explicit_xreg = !isempty(var_cols) || !isempty(formula_cols)
    require_xreg = explicit_xreg

    xreg_new = nothing
    if !isnothing(newdata)
        tbl_new = Tables.columntable(newdata)

        if !(explicit_xreg || has_auto)
            @warn "newdata provided but model has no exogenous variables. Ignoring newdata."
        else
            numeric_types = DataType[]
            var_series = Dict{String, AbstractVector}()
            auto_series = Dict{String, AbstractVector}()

            for col in var_cols
                if !haskey(tbl_new, col)
                    available_cols = join(string.(keys(tbl_new)), ", ")
                    throw(ArgumentError(
                        "Required exogenous variable ':$(col)' not found in newdata. " *
                        "Available columns: $(available_cols)"
                    ))
                end
                vec = tbl_new[col]
                if !(vec isa AbstractVector)
                    throw(ArgumentError(
                        "Exogenous variable ':$(col)' must be a vector, got $(typeof(vec))"
                    ))
                end
                if length(vec) != h
                    throw(ArgumentError(
                        "Exogenous variable ':$(col)' has length $(length(vec)), " *
                        "but forecast horizon is $(h). All xreg variables must have length h."
                    ))
                end
                core_type = Base.nonmissingtype(eltype(vec))
                if !(core_type <: Number)
                    throw(ArgumentError(
                        "Exogenous variable ':$(col)' must be numeric, got element type $(eltype(vec))."
                    ))
                end
                var_series[string(col)] = vec
                push!(numeric_types, core_type)
            end

            design_matrix, design_cols = _build_xreg_formula_matrix(fitted.spec, tbl_new, h, Utils_mod)
            design_series = Dict{String, AbstractVector}()
            if !isnothing(design_matrix) && size(design_matrix.data, 2) > 0
                for (j, name) in enumerate(design_matrix.colnames)
                    column = design_matrix.data[:, j]
                    core_type = Base.nonmissingtype(eltype(column))
                    if !(core_type <: Number)
                        throw(ArgumentError(
                            "Design column '$(name)' derived from xreg_formula must be numeric."
                        ))
                    end
                    design_series[name] = column
                    push!(numeric_types, core_type)
                end
            end

            combined_series = Dict{String, AbstractVector}()
            for (name, vec) in var_series
                combined_series[name] = vec
            end
            for (name, vec) in design_series
                combined_series[name] = vec
            end
            if has_auto
                for col in auto_cols
                    if !haskey(tbl_new, col)
                        available_cols = join(string.(keys(tbl_new)), ", ")
                        throw(ArgumentError(
                            "Required exogenous variable ':$(col)' not found in newdata. " *
                            "Available columns: $(available_cols)"
                        ))
                    end
                    vec = tbl_new[col]
                    if !(vec isa AbstractVector)
                        throw(ArgumentError(
                            "Exogenous variable ':$(col)' must be a vector, got $(typeof(vec))"
                        ))
                    end
                    if length(vec) != h
                        throw(ArgumentError(
                            "Exogenous variable ':$(col)' has length $(length(vec)), " *
                            "but forecast horizon is $(h). All xreg variables must have length h."
                        ))
                    end
                    core_type = Base.nonmissingtype(eltype(vec))
                    if !(core_type <: Number)
                        throw(ArgumentError(
                            "Exogenous variable ':$(col)' must be numeric, got element type $(eltype(vec))."
                        ))
                    end
                    auto_series[String(col)] = vec
                    push!(numeric_types, core_type)
                end
                for (name, vec) in auto_series
                    combined_series[name] = vec
                end
            end

            if isempty(combined_series)
                @warn "newdata provided but no exogenous variables detected for forecasting. Ignoring newdata."
            else
                final_names = if !isempty(train_colnames)
                    train_colnames
                else
                    names = String[string(col) for col in var_cols]
                    if !isnothing(design_matrix) && size(design_matrix.data, 2) > 0
                        append!(names, design_matrix.colnames)
                    elseif !isempty(design_cols)
                        append!(names, String.(design_cols))
                    end
                    if has_auto
                        append!(names, string.(auto_cols))
                    end
                    unique!(names)
                end

                internal_cols = Set(["drift", "intercept"])
                if any(name -> name in internal_cols, final_names)
                    push!(numeric_types, Float64)
                end
                promoted_type = isempty(numeric_types) ? Float64 : promote_type(numeric_types...)
                train_length = hasproperty(fitted.fit, :y) ? length(fitted.fit.y) : getproperty(fitted.fit, :nobs)

                internal_series = Dict{String, Vector{promoted_type}}()
                if "drift" in final_names && !haskey(combined_series, "drift")
                    drift_vals = collect((train_length + 1):(train_length + h))
                    internal_series["drift"] = convert.(promoted_type, drift_vals)
                end
                if "intercept" in final_names && !haskey(combined_series, "intercept")
                    internal_series["intercept"] = fill(one(promoted_type), h)
                end

                matrix = Matrix{promoted_type}(undef, h, length(final_names))
                for (j, name) in enumerate(final_names)
                    col_data = get(combined_series, name, nothing)
                    if isnothing(col_data)
                        col_data = get(internal_series, name, nothing)
                    else
                        col_data = convert.(promoted_type, col_data)
                    end
                    if isnothing(col_data)
                        available = join(sort(collect(keys(combined_series))), ", ")
                        throw(ArgumentError(
                            "Exogenous variable '$(name)' required by the fitted model not found in newdata. " *
                            "Available columns: $(available)"
                        ))
                    end
                    matrix[:, j] = col_data
                end
                xreg_new = Utils_mod.NamedMatrix(matrix, final_names)
            end
        end
    elseif has_auto && !isnothing(train_named) && size(train_named.data, 2) > 0
        fallback = Matrix{eltype(train_named.data)}(undef, h, size(train_named.data, 2))
        last_row = train_named.data[end, :]
        for j in 1:size(fallback, 2)
            fallback[:, j] .= last_row[j]
        end
        xreg_new = Utils_mod.NamedMatrix(fallback, train_named.colnames)
        internal_cols = Set(["drift", "mean", "intercept"])
        informative = [name for name in train_named.colnames if !(name in internal_cols)]
        if !isempty(informative)
            @info "No newdata supplied; reusing last observed automatic exogenous values for forecasting." columns=informative
        end
    elseif has_auto
        @warn "Automatic exogenous selection requested but fitted model lacks stored regressors for fallback; proceeding without xreg. Forecast accuracy may degrade."
    elseif require_xreg
        throw(ArgumentError(
            "Model has exogenous variables but no newdata provided. " *
            "Please provide newdata with the required columns for forecasting."
        ))
    end

    return Arima_mod.forecast(fitted.fit; h = h, level = level, xreg = xreg_new, pairs(kwargs)...)
end

"""
    forecast(fitted::FittedEts; h, level=[80,95], kwargs...)

Generate forecasts from a fitted ETS model.
"""
function forecast(fitted::FittedEts; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    parent_mod = parentmodule(@__MODULE__)
    Exp_mod = getfield(parent_mod, :ExponentialSmoothing)

    if !isnothing(newdata)
        @warn "newdata ignored for ETS forecasts; ETS models do not support exogenous regressors."
    end

    return Exp_mod.forecast(fitted.fit;
        h = h,
        level = level,
        kwargs...)
end

function forecast(fitted::FittedSes; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for SES forecasts."
    end
    parent_mod = parentmodule(@__MODULE__)
    Exp_mod = getfield(parent_mod, :ExponentialSmoothing)
    return Exp_mod.forecast(fitted.fit;
        h = h,
        level = level,
        kwargs...)
end

function forecast(fitted::FittedHolt; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for Holt forecasts."
    end
    parent_mod = parentmodule(@__MODULE__)
    Exp_mod = getfield(parent_mod, :ExponentialSmoothing)
    return Exp_mod.forecast(fitted.fit;
        h = h,
        level = level,
        kwargs...)
end

function forecast(fitted::FittedHoltWinters; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for Holt-Winters forecasts."
    end
    parent_mod = parentmodule(@__MODULE__)
    Exp_mod = getfield(parent_mod, :ExponentialSmoothing)
    return Exp_mod.forecast(fitted.fit;
        h = h,
        level = level,
        kwargs...)
end

function forecast(fitted::FittedCroston; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isempty(kwargs)
        @warn "Croston forecast ignores additional keywords."
    end
    if !isnothing(newdata)
        @warn "newdata ignored for Croston forecasts."
    end
    parent_mod = parentmodule(@__MODULE__)
    croston_fc = if fitted.spec.method == "hyndman"
        Exp_mod = getfield(parent_mod, :ExponentialSmoothing)
        Exp_mod.forecast(fitted.fit, h)
    else
        ID_mod = getfield(parent_mod, :IntermittentDemand)
        ID_mod.forecast(fitted.fit; h = h)
    end
    return _wrap_croston_forecast(fitted, croston_fc)
end

function _wrap_croston_forecast(fitted::FittedCroston, croston_fc)
    mean_vec = Float64.(croston_fc.mean)
    levels = Float64[]
    x_data = croston_fc.model.x
    upper = Matrix{Float64}(undef, length(mean_vec), 0)
    lower = Matrix{Float64}(undef, length(mean_vec), 0)

    fitted_vals = try
        fitted(croston_fc.model)
    catch
        Float64[]
    end

    residuals = begin
        if length(fitted_vals) == length(x_data)
            try
                Float64.(x_data) .- Float64.(fitted_vals)
            catch
                Float64[]
            end
        else
            Float64[]
        end
    end

    return Forecast(
        fitted,
        croston_fc.method,
        mean_vec,
        levels,
        x_data,
        upper,
        lower,
        fitted_vals,
        residuals
    )
end

"""
    forecast(collection::FittedModelCollection; kwargs...)

Generate forecasts for every fitted model in a collection.

Returns a `Dict{String, Any}` keyed by model name. Each value is whatever the
underlying model returns (e.g. `Forecast`, `GroupedForecasts`). Any keyword
arguments are forwarded to the individual `forecast` calls.
"""
function forecast(collection::FittedModelCollection; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    haskey(kwdict, :h) ||
        throw(ArgumentError("forecast(collection) requires keyword argument h = horizon."))

    results = Dict{String, Any}()
    for (name, model) in zip(collection.names, collection.models)
        results[name] = forecast(model; kwargs...)
    end

    return ForecastModelCollection(copy(collection.names), results, kwdict)
end

"""
    forecast(collection::ModelCollection, data; kwargs...)

Fit multiple model specifications to the same data.

# Arguments
- `collection::ModelCollection` - Collection of model specs from `model(...)`
- `data` - Tables.jl-compatible data

# Keyword Arguments
- Passed to each individual `fit()` call

# Returns
`FittedModelCollection` - Collection of fitted models with metrics

# Examples
```julia
models = model(
    ArimaSpec(@formula(sales = p() + q())),
    ArimaSpec(@formula(sales = p(1) + d(1) + q(1))),
    names = ["auto", "fixed"]
)

fitted = fit(models, data, m = 12)

# Access individual models
fitted.models[1]  # First model
fitted.names      # ["auto", "fixed"]
fitted.metrics    # Dict of metrics by model name
```
"""
function fit(collection::ModelCollection, data; kwargs...)
    fitted_models = [fit(spec, data; kwargs...) for spec in collection.specs]

    return FittedModelCollection(fitted_models, collection.names)
end

function fit(spec::ArimaSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

function fit(spec::EtsSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

function fit(spec::SesSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

function fit(spec::HoltSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

function fit(spec::HoltWintersSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

function fit(spec::CrostonSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

"""
    fit(spec::ArarSpec, data; m=nothing, groupby=nothing, parallel=true, fail_fast=false, kwargs...)

Fit an ARAR model specification to data (single series or grouped).

# Arguments
- `spec::ArarSpec` - ARAR model specification created with `@formula`
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.)

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (accepted for ModelCollection compatibility, not used by ARAR)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by for panel data
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)
- Additional kwargs passed to underlying `arar` function

# Returns
- If `groupby=nothing`: `FittedArar` - Single fitted model
- If `groupby` specified: `GroupedFittedModels` - Fitted models for each group

# Examples
```julia
# Single series fit
spec = ArarSpec(@formula(sales = arar()))
fitted = fit(spec, data)

# With custom parameters
spec = ArarSpec(@formula(sales = arar(max_ar_depth=20)))
fitted = fit(spec, data)

# Grouped data fit
spec = ArarSpec(@formula(sales = arar()))
fitted = fit(spec, data, groupby = [:product, :location])
```

# See Also
- [`ArarSpec`](@ref)
- [`forecast`](@ref)
"""
function fit(spec::ArarSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)

    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           groupby=groupby,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           kwargs...)
    end

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    if !haskey(tbl, target_col)
        available_cols = join(string.(keys(tbl)), ", ")
        throw(ArgumentError(
            "Target variable ':$(target_col)' not found in data. " *
            "Available columns: $(available_cols)"
        ))
    end

    target_data = tbl[target_col]
    if !(target_data isa AbstractVector)
        throw(ArgumentError(
            "Target variable ':$(target_col)' must be a vector, got $(typeof(target_data))"
        ))
    end

    parent_mod = parentmodule(@__MODULE__)
    Ararma_mod = getfield(parent_mod, :Ararma)
    arar_fit = Ararma_mod.arar(spec.formula, data; pairs(fit_options)...)

    return FittedArar(
        spec,
        arar_fit,
        target_col,
        tbl
    )
end

"""
    forecast(fitted::FittedArar; h::Int, level::Vector{<:Real} = [80, 95], kwargs...)

Generate forecasts from a fitted ARAR model.

# Arguments
- `fitted::FittedArar` - Fitted ARAR model from `fit(ArarSpec, data)`

# Keyword Arguments
- `h::Int` - Forecast horizon (number of periods ahead)
- `level::Vector{<:Real}` - Confidence levels for prediction intervals (default [80, 95])
- Additional kwargs passed to underlying `forecast` function

# Returns
`Forecast` object containing point forecasts and prediction intervals

# Examples
```julia
spec = ArarSpec(@formula(sales = arar()))
fitted = fit(spec, data)

# 12-period ahead forecast
fc = forecast(fitted, h = 12)

# Custom confidence levels
fc = forecast(fitted, h = 12, level = [90, 95, 99])
```

# See Also
- [`ArarSpec`](@ref)
- [`fit`](@ref)
"""
function forecast(fitted::FittedArar; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for ARAR forecasts."
    end
    parent_mod = parentmodule(@__MODULE__)
    Generics_mod = getfield(parent_mod, :Generics)

    return Generics_mod.forecast(fitted.fit; h=h, level=level, kwargs...)
end

function fit(spec::ArarSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    
    return fit(spec, panel.data; pairs(kwdict)...)
end

"""
    fit(spec::ArarmaSpec, data; m=nothing, groupby=nothing, parallel=true, fail_fast=false, kwargs...)

Fit an ARARMA model specification to data (single series or grouped).

# Arguments
- `spec::ArarmaSpec` - ARARMA model specification created with `@formula`
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.)

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (accepted for ModelCollection compatibility, not used by ARARMA)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by for panel data
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)
- Additional kwargs passed to underlying `ararma` or `auto_ararma` function

# Returns
- If `groupby=nothing`: `FittedArarma` - Single fitted model
- If `groupby` specified: `GroupedFittedModels` - Fitted models for each group

# Examples
```julia
# Single series fit - fixed orders
spec = ArarmaSpec(@formula(sales = p(1) + q(2)))
fitted = fit(spec, data)

# Single series fit - auto selection
spec = ArarmaSpec(@formula(sales = p() + q()))
fitted = fit(spec, data)

# With custom ARAR parameters
spec = ArarmaSpec(@formula(sales = p(2) + q(1)), max_ar_depth=20, max_lag=30)
fitted = fit(spec, data)

# Grouped data fit
spec = ArarmaSpec(@formula(sales = p() + q()))
fitted = fit(spec, data, groupby = [:product, :location])
```

# See Also
- [`ArarmaSpec`](@ref)
- [`forecast`](@ref)
"""
function fit(spec::ArarmaSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)

    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           groupby=groupby,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           kwargs...)
    end

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))

    fit_options[:max_ar_depth] = spec.max_ar_depth
    fit_options[:max_lag] = spec.max_lag

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    if !haskey(tbl, target_col)
        available_cols = join(string.(keys(tbl)), ", ")
        throw(ArgumentError(
            "Target variable ':$(target_col)' not found in data. " *
            "Available columns: $(available_cols)"
        ))
    end

    target_data = tbl[target_col]
    if !(target_data isa AbstractVector)
        throw(ArgumentError(
            "Target variable ':$(target_col)' must be a vector, got $(typeof(target_data))"
        ))
    end

    parent_mod = parentmodule(@__MODULE__)
    Ararma_mod = getfield(parent_mod, :Ararma)

    ararma_fit = Ararma_mod.ararma(spec.formula, data; pairs(fit_options)...)

    return FittedArarma(
        spec,
        ararma_fit,
        target_col,
        tbl
    )
end

"""
    forecast(fitted::FittedArarma; h::Int, level::Vector{<:Real} = [80, 95], kwargs...)

Generate forecasts from a fitted ARARMA model.

# Arguments
- `fitted::FittedArarma` - Fitted ARARMA model from `fit(ArarmaSpec, data)`

# Keyword Arguments
- `h::Int` - Forecast horizon (number of periods ahead)
- `level::Vector{<:Real}` - Confidence levels for prediction intervals (default [80, 95])
- Additional kwargs passed to underlying `forecast` function

# Returns
`Forecast` object containing point forecasts and prediction intervals

# Examples
```julia
spec = ArarmaSpec(@formula(sales = p(1) + q(2)))
fitted = fit(spec, data)

# 12-period ahead forecast
fc = forecast(fitted, h = 12)

# Custom confidence levels
fc = forecast(fitted, h = 12, level = [90, 95, 99])
```

# See Also
- [`ArarmaSpec`](@ref)
- [`fit`](@ref)
"""
function forecast(fitted::FittedArarma; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for ARARMA forecasts."
    end
    parent_mod = parentmodule(@__MODULE__)
    Generics_mod = getfield(parent_mod, :Generics)

    return Generics_mod.forecast(fitted.fit; h=h, level=level, kwargs...)
end

function fit(spec::ArarmaSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end

    return fit(spec, panel.data; pairs(kwdict)...)
end

"""
    fit(spec::BatsSpec, data; m=nothing, groupby=nothing, parallel=true, fail_fast=false, kwargs...)

Fit a BATS model specification to data (single series or grouped).

# Arguments
- `spec::BatsSpec` - BATS model specification created with `@formula`
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.)

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (consumed but not used; BATS gets `m` from the formula's `seasonal_periods`/`m` term)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by for panel data
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)
- Additional kwargs passed to underlying `bats` function

# Returns
- If `groupby=nothing`: `FittedBats` - Single fitted model
- If `groupby` specified: `GroupedFittedModels` - Fitted models for each group

# Examples
```julia
# Single series fit with defaults
spec = BatsSpec(@formula(sales = bats()))
fitted = fit(spec, data)

# With seasonal period
spec = BatsSpec(@formula(sales = bats(m=12)))
fitted = fit(spec, data)

# With multiple seasonal periods
spec = BatsSpec(@formula(sales = bats(m=[24, 168])))
fitted = fit(spec, data)

# With custom parameters
spec = BatsSpec(@formula(sales = bats(m=12, use_box_cox=true)))
fitted = fit(spec, data, bc_lower=0.0, bc_upper=1.5)

# Grouped data fit
spec = BatsSpec(@formula(sales = bats(m=12)))
fitted = fit(spec, data, groupby = [:product, :location])
```

# See Also
- [`BatsSpec`](@ref)
- [`forecast`](@ref)
"""
function fit(spec::BatsSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)

    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           groupby=groupby,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           kwargs...)
    end

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    if !haskey(tbl, target_col)
        available_cols = join(string.(keys(tbl)), ", ")
        throw(ArgumentError(
            "Target variable ':$(target_col)' not found in data. " *
            "Available columns: $(available_cols)"
        ))
    end

    target_data = tbl[target_col]
    if !(target_data isa AbstractVector)
        throw(ArgumentError(
            "Target variable ':$(target_col)' must be a vector, got $(typeof(target_data))"
        ))
    end

    parent_mod = parentmodule(@__MODULE__)
    Bats_mod = getfield(parent_mod, :Bats)
    bats_fit = Bats_mod.bats(spec.formula, data; pairs(fit_options)...)

    return FittedBats(
        spec,
        bats_fit,
        target_col,
        tbl
    )
end

"""
    forecast(fitted::FittedBats; h::Int, level::Vector{<:Real} = [80, 95], kwargs...)

Generate forecasts from a fitted BATS model.

# Arguments
- `fitted::FittedBats` - Fitted BATS model from `fit(BatsSpec, data)`

# Keyword Arguments
- `h::Int` - Forecast horizon (number of periods ahead)
- `level::Vector{<:Real}` - Confidence levels for prediction intervals (default [80, 95])
- `fan::Bool` - Generate fan chart with multiple levels (default false)
- `biasadj::Union{Bool, Nothing}` - Bias adjustment for Box-Cox back-transformation
- Additional kwargs passed to underlying `forecast` function

# Returns
`Forecast` object containing point forecasts and prediction intervals

# Examples
```julia
spec = BatsSpec(@formula(sales = bats(m=12)))
fitted = fit(spec, data)

# 12-period ahead forecast
fc = forecast(fitted, h = 12)

# Custom confidence levels
fc = forecast(fitted, h = 12, level = [90, 95, 99])

# Fan chart
fc = forecast(fitted, h = 12, fan = true)

# With bias adjustment
fc = forecast(fitted, h = 12, biasadj = true)
```

# See Also
- [`BatsSpec`](@ref)
- [`fit`](@ref)
"""
function forecast(fitted::FittedBats; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for BATS forecasts."
    end
    parent_mod = parentmodule(@__MODULE__)
    Generics_mod = getfield(parent_mod, :Generics)

    return Generics_mod.forecast(fitted.fit; h=h, level=level, kwargs...)
end

function fit(spec::BatsSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end

    return fit(spec, panel.data; pairs(kwdict)...)
end

"""
    fit(spec::TbatsSpec, data; m=nothing, groupby=nothing, parallel=true, fail_fast=false, kwargs...)

Fit a TBATS model specification to data (single series or grouped).

TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and
Seasonal components) uses Fourier-based seasonal representation, enabling non-integer
seasonal periods and efficient handling of very long seasonal cycles.

# Arguments
- `spec::TbatsSpec` - TBATS model specification created with `@formula`
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.)

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (consumed but not used; TBATS gets `m` from the formula's `seasonal_periods`/`m` term)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by for panel data
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)
- Additional kwargs passed to underlying `tbats` function:
  - `bc_lower::Real=0.0` - Lower bound for Box-Cox lambda search
  - `bc_upper::Real=1.0` - Upper bound for Box-Cox lambda search
  - `biasadj::Bool=false` - Bias-adjusted inverse Box-Cox transformation
  - `model=nothing` - Previous TBATS model to refit

# Returns
- If `groupby=nothing`: `FittedTbats` - Single fitted model
- If `groupby` specified: `GroupedFittedModels` - Fitted models for each group

# Examples
```julia
# Single series fit with defaults
spec = TbatsSpec(@formula(sales = tbats()))
fitted = fit(spec, data)

# With non-integer seasonal period (weekly data, yearly seasonality)
spec = TbatsSpec(@formula(sales = tbats(m=52.18)))
fitted = fit(spec, data)

# With multiple seasonal periods
spec = TbatsSpec(@formula(sales = tbats(m=[7, 365.25])))
fitted = fit(spec, data)

# With explicit Fourier orders
spec = TbatsSpec(@formula(sales = tbats(m=[7, 365.25], k=[3, 10])))
fitted = fit(spec, data)

# With custom Box-Cox bounds
spec = TbatsSpec(@formula(sales = tbats(m=52.18, use_box_cox=true)))
fitted = fit(spec, data, bc_lower=0.0, bc_upper=1.5)

# Grouped data fit (panel data)
spec = TbatsSpec(@formula(sales = tbats(m=52.18)))
fitted = fit(spec, data, groupby = [:product, :location])

fitted = fit(spec, data, groupby=:product, parallel=true, fail_fast=false)
```

# See Also
- [`TbatsSpec`](@ref)
- [`forecast`](@ref)
- [`BatsSpec`](@ref) - BATS specification (integer seasonal periods only)
"""
function fit(spec::TbatsSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)

    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           groupby=groupby,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           kwargs...)
    end

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    if !haskey(tbl, target_col)
        available_cols = join(string.(keys(tbl)), ", ")
        throw(ArgumentError(
            "Target variable ':$(target_col)' not found in data. " *
            "Available columns: $(available_cols)"
        ))
    end

    target_data = tbl[target_col]
    if !(target_data isa AbstractVector)
        throw(ArgumentError(
            "Target variable ':$(target_col)' must be a vector, got $(typeof(target_data))"
        ))
    end

    parent_mod = parentmodule(@__MODULE__)
    Tbats_mod = getfield(parent_mod, :Tbats)
    tbats_fit = Tbats_mod.tbats(spec.formula, data; pairs(fit_options)...)

    return FittedTbats(
        spec,
        tbats_fit,
        target_col,
        tbl
    )
end

"""
    forecast(fitted::FittedTbats; h::Int, level::Vector{<:Real} = [80, 95], kwargs...)

Generate forecasts from a fitted TBATS model.

# Arguments
- `fitted::FittedTbats` - Fitted TBATS model from `fit(TbatsSpec, data)`

# Keyword Arguments
- `h::Int` - Forecast horizon (number of periods ahead)
- `level::Vector{<:Real}` - Confidence levels for prediction intervals (default [80, 95])
- `fan::Bool` - Generate fan chart with multiple levels (default false)
- `biasadj::Union{Bool, Nothing}` - Bias adjustment for Box-Cox back-transformation
- Additional kwargs passed to underlying `forecast` function

# Returns
`Forecast` object containing point forecasts and prediction intervals

# Examples
```julia
spec = TbatsSpec(@formula(sales = tbats(m=52.18)))
fitted = fit(spec, data)

# 12-period ahead forecast
fc = forecast(fitted, h = 12)

# Custom confidence levels
fc = forecast(fitted, h = 12, level = [90, 95, 99])

# Fan chart
fc = forecast(fitted, h = 12, fan = true)

# With bias adjustment
fc = forecast(fitted, h = 12, biasadj = true)

# Access forecast components
fc.mean        # Point forecasts
fc.lower       # Lower bounds for each level
fc.upper       # Upper bounds for each level
fc.level       # Confidence levels
```

# See Also
- [`TbatsSpec`](@ref)
- [`fit`](@ref)
"""
function forecast(fitted::FittedTbats; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for TBATS forecasts."
    end
    parent_mod = parentmodule(@__MODULE__)
    Generics_mod = getfield(parent_mod, :Generics)

    return Generics_mod.forecast(fitted.fit; h=h, level=level, kwargs...)
end

function fit(spec::TbatsSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end

    return fit(spec, panel.data; pairs(kwdict)...)
end

"""
    fit(spec::ThetaSpec, data; m=nothing, groupby=nothing, parallel=true, fail_fast=false, kwargs...)

Fit a Theta model specification to data (single series or grouped).

The Theta method decomposes a time series into "theta lines" capturing long-term
trend and short-term dynamics, then combines their forecasts. Supports four variants:
STM (Simple), OTM (Optimized), DSTM (Dynamic Simple), DOTM (Dynamic Optimized).

# Arguments
- `spec::ThetaSpec` - Theta model specification created with `@formula`
- `data` - Tables.jl-compatible data (NamedTuple, DataFrame, CSV.File, etc.)

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (required if not specified in spec)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by for panel data
- `datecol::Union{Symbol, Nothing}` - Date/time column (excluded from processing)
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)
- Additional kwargs passed to underlying `theta` or `auto_theta` function

# Returns
- If `groupby=nothing`: `FittedTheta` - Single fitted model
- If `groupby` specified: `GroupedFittedModels` - Fitted models for each group

# Examples
```julia
# Single series fit with auto model selection
spec = ThetaSpec(@formula(sales = theta()))
fitted = fit(spec, data, m=12)

spec = ThetaSpec(@formula(sales = theta(model=:OTM)))
fitted = fit(spec, data, m=12)

# With fixed parameters
spec = ThetaSpec(@formula(sales = theta(model=:OTM, alpha=0.3)))
fitted = fit(spec, data, m=12)

spec = ThetaSpec(@formula(sales = theta()))
fitted = fit(spec, data, m=12, groupby=[:product, :region])

fitted = fit(spec, data, m=12, groupby=:product, parallel=true, fail_fast=false)
```
"""
function fit(spec::ThetaSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)

    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m=m,
                           groupby=groupby,
                           datecol=datecol,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           kwargs...)
    end

    seasonal_period = if !isnothing(m)
        m
    elseif !isnothing(spec.m)
        spec.m
    else
        throw(ArgumentError(
            "Seasonal period 'm' must be specified either in ThetaSpec or as kwarg to fit(). " *
            "Example: fit(spec, data, m=12) or ThetaSpec(..., m=12). Use m=1 for non-seasonal data."
        ))
    end

    if seasonal_period < 1
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))
    end

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    if !haskey(tbl, target_col)
        available_cols = join(string.(keys(tbl)), ", ")
        throw(ArgumentError(
            "Target variable ':$(target_col)' not found in data. " *
            "Available columns: $(available_cols)"
        ))
    end

    target_data = tbl[target_col]
    if !(target_data isa AbstractVector)
        throw(ArgumentError(
            "Target variable ':$(target_col)' must be a vector, got $(typeof(target_data))"
        ))
    end

    el = Base.nonmissingtype(eltype(target_data))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_data))"))

    parent_mod = parentmodule(@__MODULE__)
    Theta_mod = getfield(parent_mod, :Theta)
    theta_fit = Theta_mod.theta(spec.formula, data; m=seasonal_period, pairs(fit_options)...)

    return FittedTheta(
        spec,
        theta_fit,
        target_col,
        tbl,
        seasonal_period
    )
end

"""
    forecast(fitted::FittedTheta; h::Int, level::Vector{<:Real} = [80, 95], kwargs...)

Generate forecasts from a fitted Theta model.

# Arguments
- `fitted::FittedTheta` - Fitted Theta model from `fit(ThetaSpec, data)`

# Keyword Arguments
- `h::Int` - Forecast horizon (number of periods ahead)
- `level::Vector{<:Real}` - Confidence levels for prediction intervals (default [80, 95])
- Additional kwargs passed to underlying `forecast` function

# Returns
`Forecast` object containing point forecasts and prediction intervals

# Examples
```julia
spec = ThetaSpec(@formula(sales = theta()))
fitted = fit(spec, data, m=12)

# 12-period ahead forecast
fc = forecast(fitted, h=12)

# Custom confidence levels
fc = forecast(fitted, h=12, level=[90, 95, 99])

# Access forecast components
fc.mean        # Point forecasts
fc.lower       # Lower bounds for each level
fc.upper       # Upper bounds for each level
fc.level       # Confidence levels
```

"""
function forecast(fitted::FittedTheta; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for Theta forecasts; Theta models do not support exogenous regressors."
    end

    parent_mod = parentmodule(@__MODULE__)
    Generics_mod = getfield(parent_mod, :Generics)

    return Generics_mod.forecast(fitted.fit; h=h, level=level, kwargs...)
end

function fit(spec::ThetaSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

# =============================================================================
# Naive Model Fitting
# =============================================================================

"""
    fit(spec::NaiveSpec, data; m=nothing, groupby=nothing, kwargs...)

Fit a naive forecasting model to data (single series or grouped).

The naive method uses the last observed value as the forecast for all future periods.

# Arguments
- `spec::NaiveSpec` - Naive model specification
- `data` - Tables.jl-compatible data

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (stored for reference)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)

# Returns
- If `groupby=nothing`: `FittedNaive`
- If `groupby` specified: `GroupedFittedModels`

# Examples
```julia
spec = NaiveSpec(@formula(sales = naive_term()))
fitted = fit(spec, data)
fc = forecast(fitted, h=12)

# Grouped
fitted = fit(spec, data, groupby=:store)
```
"""
function fit(spec::NaiveSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             lambda::Union{Nothing, Float64} = nothing,
             biasadj::Union{Nothing, Bool} = nothing,
             kwargs...)

    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m=m,
                           groupby=groupby,
                           datecol=datecol,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           lambda=lambda,
                           biasadj=biasadj,
                           kwargs...)
    end

    seasonal_period = isnothing(m) ? (isnothing(spec.m) ? 1 : spec.m) : m
    seasonal_period >= 1 ||
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    haskey(tbl, target_col) ||
        throw(ArgumentError("Target variable ':$(target_col)' not found in data."))

    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError("Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"))

    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"))

    # Allow kwargs to override spec values
    use_lambda = isnothing(lambda) ? spec.lambda : lambda
    use_biasadj = isnothing(biasadj) ? spec.biasadj : biasadj

    parent_mod = parentmodule(@__MODULE__)
    Naive_mod = getfield(parent_mod, :Naive)

    naive_fit = Naive_mod.naive(target_vector, seasonal_period;
                                 lambda=use_lambda,
                                 biasadj=use_biasadj)

    return FittedNaive(spec, naive_fit, target_col, tbl, seasonal_period)
end

"""
    fit(spec::SnaiveSpec, data; m, groupby=nothing, start=nothing, kwargs...)

Fit a seasonal naive forecasting model to data (single series or grouped).

The seasonal naive method uses the observation from m periods ago as the forecast.
The seasonal period `m` is required.

# Arguments
- `spec::SnaiveSpec` - Seasonal naive model specification
- `data` - Tables.jl-compatible data

# Keyword Arguments
- `m::Union{Int, Nothing}` - Seasonal period (required if not in spec)
- `start::Union{Int, Nothing}` - Seasonal position of first observation (1 to m)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)

# Returns
- If `groupby=nothing`: `FittedSnaive`
- If `groupby` specified: `GroupedFittedModels`

# Examples
```julia
spec = SnaiveSpec(@formula(sales = snaive_term()))
fitted = fit(spec, data, m=12)
fc = forecast(fitted, h=24)

# Monthly data starting in March
fitted = fit(spec, data, m=12, start=3)

# Grouped
fitted = fit(spec, data, m=12, groupby=:store)
```
"""
function fit(spec::SnaiveSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             lambda::Union{Nothing, Float64} = nothing,
             biasadj::Union{Nothing, Bool} = nothing,
             kwargs...)

    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m=m,
                           groupby=groupby,
                           datecol=datecol,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           lambda=lambda,
                           biasadj=biasadj,
                           kwargs...)
    end

    seasonal_period = if !isnothing(m)
        m
    elseif !isnothing(spec.m)
        spec.m
    else
        throw(ArgumentError(
            "Seasonal period 'm' must be specified for seasonal naive. " *
            "Provide it in the spec or as a kwarg to fit()."
        ))
    end

    seasonal_period >= 1 ||
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    haskey(tbl, target_col) ||
        throw(ArgumentError("Target variable ':$(target_col)' not found in data."))

    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError("Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"))

    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"))

    # Allow kwargs to override spec values
    use_lambda = isnothing(lambda) ? spec.lambda : lambda
    use_biasadj = isnothing(biasadj) ? spec.biasadj : biasadj

    parent_mod = parentmodule(@__MODULE__)
    Naive_mod = getfield(parent_mod, :Naive)

    snaive_fit = Naive_mod.snaive(target_vector, seasonal_period;
                                   lambda=use_lambda,
                                   biasadj=use_biasadj)

    return FittedSnaive(spec, snaive_fit, target_col, tbl, seasonal_period)
end

"""
    fit(spec::RwSpec, data; m=nothing, groupby=nothing, kwargs...)

Fit a random walk forecasting model to data (single series or grouped).

Without drift, equivalent to naive. With drift, includes a linear trend.

# Arguments
- `spec::RwSpec` - Random walk model specification
- `data` - Tables.jl-compatible data

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (stored for reference)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)

# Returns
- If `groupby=nothing`: `FittedRw`
- If `groupby` specified: `GroupedFittedModels`

# Examples
```julia
spec = RwSpec(@formula(sales = rw_term(drift=true)))
fitted = fit(spec, data)
fc = forecast(fitted, h=12)

# Grouped
fitted = fit(spec, data, groupby=:store)
```
"""
function fit(spec::RwSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             lambda::Union{Nothing, Float64} = nothing,
             biasadj::Union{Nothing, Bool} = nothing,
             kwargs...)

    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m=m,
                           groupby=groupby,
                           datecol=datecol,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           lambda=lambda,
                           biasadj=biasadj,
                           kwargs...)
    end

    seasonal_period = isnothing(m) ? (isnothing(spec.m) ? 1 : spec.m) : m
    seasonal_period >= 1 ||
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    haskey(tbl, target_col) ||
        throw(ArgumentError("Target variable ':$(target_col)' not found in data."))

    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError("Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"))

    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"))

    # Determine drift from spec or formula term
    use_drift = spec.drift

    # Allow kwargs to override spec values
    use_lambda = isnothing(lambda) ? spec.lambda : lambda
    use_biasadj = isnothing(biasadj) ? spec.biasadj : biasadj

    parent_mod = parentmodule(@__MODULE__)
    Naive_mod = getfield(parent_mod, :Naive)

    rw_fit = Naive_mod.rw(target_vector, seasonal_period;
                          drift=use_drift,
                          lambda=use_lambda,
                          biasadj=use_biasadj)

    return FittedRw(spec, rw_fit, target_col, tbl, seasonal_period)
end

# =============================================================================
# Naive Model Forecasting
# =============================================================================

"""
    forecast(fitted::FittedNaive; h, level=[80,95], kwargs...)

Generate forecasts from a fitted naive model.
"""
function forecast(fitted::FittedNaive; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for naive forecasts."
    end

    parent_mod = parentmodule(@__MODULE__)
    Naive_mod = getfield(parent_mod, :Naive)

    return Naive_mod.forecast(fitted.fit; h=h, level=level, kwargs...)
end

"""
    forecast(fitted::FittedSnaive; h, level=[80,95], kwargs...)

Generate forecasts from a fitted seasonal naive model.
"""
function forecast(fitted::FittedSnaive; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for seasonal naive forecasts."
    end

    parent_mod = parentmodule(@__MODULE__)
    Naive_mod = getfield(parent_mod, :Naive)

    return Naive_mod.forecast(fitted.fit; h=h, level=level, kwargs...)
end

"""
    forecast(fitted::FittedRw; h, level=[80,95], kwargs...)

Generate forecasts from a fitted random walk model.
"""
function forecast(fitted::FittedRw; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for random walk forecasts."
    end

    parent_mod = parentmodule(@__MODULE__)
    Naive_mod = getfield(parent_mod, :Naive)

    return Naive_mod.forecast(fitted.fit; h=h, level=level, kwargs...)
end

# =============================================================================
# PanelData support for Naive models
# =============================================================================

function fit(spec::NaiveSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

function fit(spec::SnaiveSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

function fit(spec::RwSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

# =============================================================================
# Mean Forecasting Model (MeanfSpec)
# =============================================================================

"""
    fit(spec::MeanfSpec, data; m=nothing, groupby=nothing, kwargs...)

Fit a mean forecasting model to data (single series or grouped).

The mean method uses the sample mean as the forecast for all future periods.

# Arguments
- `spec::MeanfSpec` - Mean model specification
- `data` - Tables.jl-compatible data

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (stored for reference)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by
- `lambda::Union{Nothing, Float64}` - Box-Cox transformation parameter (overrides spec)
- `biasadj::Union{Nothing, Bool}` - Bias adjustment (overrides spec)
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)

# Returns
- If `groupby=nothing`: `FittedMeanf`
- If `groupby` specified: `GroupedFittedModels`

# Examples
```julia
spec = MeanfSpec(@formula(sales = meanf_term()))
fitted = fit(spec, data, m=12)
fc = forecast(fitted, h=12)

# Grouped
fitted = fit(spec, data, m=12, groupby=:store)
```
"""
function fit(spec::MeanfSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             lambda::Union{Nothing, Float64} = nothing,
             biasadj::Union{Nothing, Bool} = nothing,
             kwargs...)

    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m=m,
                           groupby=groupby,
                           datecol=datecol,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           lambda=lambda,
                           biasadj=biasadj,
                           kwargs...)
    end

    seasonal_period = isnothing(m) ? (isnothing(spec.m) ? 1 : spec.m) : m
    seasonal_period >= 1 ||
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    haskey(tbl, target_col) ||
        throw(ArgumentError("Target variable ':$(target_col)' not found in data."))

    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError("Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"))

    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"))

    # Allow kwargs to override spec values
    use_lambda = isnothing(lambda) ? spec.lambda : lambda
    use_biasadj = isnothing(biasadj) ? spec.biasadj : biasadj

    parent_mod = parentmodule(@__MODULE__)
    Naive_mod = getfield(parent_mod, :Naive)

    meanf_fit = Naive_mod.meanf(target_vector, seasonal_period;
                                 lambda=use_lambda,
                                 biasadj=use_biasadj)

    return FittedMeanf(spec, meanf_fit, target_col, tbl, seasonal_period)
end

"""
    forecast(fitted::FittedMeanf; h, level=[80,95], kwargs...)

Generate forecasts from a fitted mean model.
"""
function forecast(fitted::FittedMeanf; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for mean forecasts."
    end

    parent_mod = parentmodule(@__MODULE__)
    Naive_mod = getfield(parent_mod, :Naive)

    # Convert level to Float64 for meanf forecast
    level_f64 = Float64.(level)
    return Naive_mod.forecast(fitted.fit; h=h, level=level_f64, kwargs...)
end

function fit(spec::MeanfSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    if !haskey(kwdict, :m) && !isnothing(panel.m)
        kwdict[:m] = resolve_m(panel.m, spec)
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end

# =============================================================================
# Diffusion Model Fitting and Forecasting
# =============================================================================

"""
    _diffusion_model_type(sym::Union{Symbol, Nothing})

Convert a Symbol from DiffusionTerm to a DiffusionModelType enum value.
"""
function _diffusion_model_type(sym::Union{Symbol, Nothing})
    isnothing(sym) && return nothing
    parent_mod = parentmodule(@__MODULE__)
    Diff_mod = getfield(parent_mod, :Diffusion)
    mapping = Dict{Symbol, Any}(
        :Bass => Diff_mod.Bass,
        :Gompertz => Diff_mod.Gompertz,
        :GSGompertz => Diff_mod.GSGompertz,
        :Weibull => Diff_mod.Weibull,
    )
    haskey(mapping, sym) || throw(ArgumentError(
        "Unknown diffusion model type ':$(sym)'. " *
        "Valid types: :Bass, :Gompertz, :GSGompertz, :Weibull"))
    return mapping[sym]
end

"""
    _diffusion_fixed_params(term::DiffusionTerm)

Build the `w` named tuple of fixed parameters from a DiffusionTerm.
Returns `nothing` if no parameters are fixed.
"""
function _diffusion_fixed_params(term)
    pairs_list = Pair{Symbol, Union{Float64, Nothing}}[]
    for field in (:m, :p, :q, :a, :b, :c)
        val = getfield(term, field)
        if !isnothing(val)
            push!(pairs_list, field => val)
        end
    end
    isempty(pairs_list) && return nothing
    return NamedTuple(pairs_list)
end

"""
    fit(spec::DiffusionSpec, data; m=nothing, groupby=nothing, parallel=true, fail_fast=false, kwargs...)

Fit a diffusion model specification to data (single series or grouped).

# Arguments
- `spec::DiffusionSpec` - Diffusion model specification created with `@formula`
- `data` - Tables.jl-compatible data

# Keyword Arguments
- `m::Union{Int, Nothing}=nothing` - Seasonal period (accepted for ModelCollection compatibility, not used by Diffusion)
- `groupby::Union{Symbol, Vector{Symbol}, Nothing}` - Column(s) to group by for panel data
- `parallel::Bool` - Use parallel processing for grouped data (default true)
- `fail_fast::Bool` - Stop on first error in grouped fitting (default false)
- Additional kwargs passed to underlying `diffusion` function

# Returns
- If `groupby=nothing`: `FittedDiffusion` - Single fitted model
- If `groupby` specified: `GroupedFittedModels` - Fitted models for each group

# Examples
```julia
spec = DiffusionSpec(@formula(adoption = diffusion()))
fitted = fit(spec, data)

spec = DiffusionSpec(@formula(adoption = diffusion(model=:Bass)))
fitted = fit(spec, data)

# Grouped
fitted = fit(spec, data, groupby=:product)
```
"""
function fit(spec::DiffusionSpec, data;
             m::Union{Int, Nothing} = nothing,
             groupby::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
             datecol::Union{Symbol, Nothing} = nothing,
             parallel::Bool = true,
             fail_fast::Bool = false,
             kwargs...)

    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           groupby=groupby,
                           datecol=datecol,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           kwargs...)
    end

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    haskey(tbl, target_col) ||
        throw(ArgumentError("Target variable ':$(target_col)' not found in data."))

    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError("Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"))

    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError("Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"))

    # Extract options from DiffusionTerm and spec
    diff_terms = filter(t -> isa(t, DiffusionTerm), spec.formula.terms)
    if isempty(diff_terms)
        throw(ArgumentError(
            "DiffusionSpec formula must contain a diffusion() term, " *
            "but none was found in: $(spec.formula)"
        ))
    end

    fit_options = copy(spec.options)
    merge!(fit_options, Dict{Symbol, Any}(kwargs))

    term = diff_terms[1]

    model_type = _diffusion_model_type(term.model_type)
    if !isnothing(model_type) && !haskey(fit_options, :model_type)
        fit_options[:model_type] = model_type
    end

    w = _diffusion_fixed_params(term)
    if !isnothing(w) && !haskey(fit_options, :w)
        fit_options[:w] = w
    end

    if !isnothing(term.loss) && !haskey(fit_options, :loss)
        fit_options[:loss] = term.loss
    end

    if !isnothing(term.cumulative) && !haskey(fit_options, :cumulative)
        fit_options[:cumulative] = term.cumulative
    end

    parent_mod = parentmodule(@__MODULE__)
    Diff_mod = getfield(parent_mod, :Diffusion)

    diff_fit = Diff_mod.diffusion(target_vector; pairs(fit_options)...)

    return FittedDiffusion(spec, diff_fit, target_col, tbl)
end

"""
    forecast(fitted::FittedDiffusion; h, level=[80, 95], kwargs...)

Generate forecasts from a fitted diffusion model.

# Keyword Arguments
- `h::Int` - Forecast horizon (number of periods ahead)
- `level::Vector{<:Real}` - Confidence levels for prediction intervals (default [80, 95])
- Additional kwargs passed to underlying `forecast` function

# Returns
`Forecast` object with mean, lower, and upper prediction intervals

# Examples
```julia
spec = DiffusionSpec(@formula(adoption = diffusion()))
fitted = fit(spec, data)
fc = forecast(fitted, h=12)
```
"""
function forecast(fitted::FittedDiffusion; h::Int, level::Vector{<:Real} = [80, 95], newdata = nothing, kwargs...)
    if !isnothing(newdata)
        @warn "newdata ignored for diffusion forecasts; diffusion models do not support exogenous regressors."
    end

    parent_mod = parentmodule(@__MODULE__)
    Diff_mod = getfield(parent_mod, :Diffusion)

    return Diff_mod.forecast(fitted.fit; h=h, level=level, kwargs...)
end

function fit(spec::DiffusionSpec, panel::PanelData; kwargs...)
    kwdict = Dict{Symbol, Any}(kwargs)
    if !haskey(kwdict, :groupby) && !isempty(panel.groups)
        kwdict[:groupby] = panel.groups
    end
    if !haskey(kwdict, :datecol) && !isnothing(panel.date)
        kwdict[:datecol] = panel.date
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end
