"""
    fit.jl

Implementation of fit() methods for model specifications.

This file provides the concrete implementations that connect model specs
to the underlying fitting functions (auto_arima, arima, etc.).
"""

# Import fit and forecast generics from Generics module
# Note: These will be available after Generics is loaded
import ..Generics: fit, forecast

# Import Tables for data handling
import Tables

# Import ARIMA fitting functions
# Note: These will be available after Arima module is loaded
# We use qualified names (..Arima.auto_arima) in function bodies instead of importing here

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
    # Route to grouped fitting if groupby specified
    if !isnothing(groupby)
        return fit_grouped(spec, data;
                           m=m,
                           groupby=groupby,
                           datecol=datecol,
                           parallel=parallel,
                           fail_fast=fail_fast,
                           kwargs...)
    end

    # Single series fitting below
    # Determine seasonal period
    # Priority: kwarg > spec.m > error
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

    # Validate seasonal period
    if seasonal_period < 1
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))
    end

    # Merge spec options with kwargs (kwargs take precedence)
    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))

# Convert data to columntable for schema extraction
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

    # Access Utils module (provides NamedMatrix/model_matrix helpers)
    parent_mod = parentmodule(@__MODULE__)
    Utils_mod = getfield(parent_mod, :Utils)

    # Build xreg design matrix when spec.xreg_formula is provided
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

    # Call existing auto_arima with formula interface
    # This handles all the heavy lifting:
    # - Target extraction
    # - Exogenous variable extraction from VarTerms
    # - Smart routing (search vs fixed)
    # - Validation
    # Note: Access Arima module via parent module since it's loaded after ModelSpecs
    Arima_mod = getfield(parent_mod, :Arima)
    arima_fit = Arima_mod.auto_arima(spec.formula, data, seasonal_period; pairs(fit_options)...)

    # Extract exogenous variable names from VarTerms
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

    # Create and return FittedArima
    return FittedArima(
        spec,
        arima_fit,
        target_col,
        xreg_cols,
        tbl,  # Pass columntable for schema extraction
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

                promoted_type = isempty(numeric_types) ? Float64 : promote_type(numeric_types...)
                matrix = Matrix{promoted_type}(undef, h, length(final_names))
                for (j, name) in enumerate(final_names)
                    if !haskey(combined_series, name)
                        available = join(sort(collect(keys(combined_series))), ", ")
                        throw(ArgumentError(
                            "Exogenous variable '$(name)' required by the fitted model not found in newdata. " *
                            "Available columns: $(available)"
                        ))
                    end
                    matrix[:, j] = convert.(promoted_type, combined_series[name])
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
    # Fit each spec
    fitted_models = [fit(spec, data; kwargs...) for spec in collection.specs]

    # Extract metrics from each fitted model
    metrics = Dict{String, Dict{Symbol, Float64}}()
    for (name, fitted) in zip(collection.names, fitted_models)
        metrics[name] = extract_metrics(fitted)
    end

    return FittedModelCollection(fitted_models, collection.names, metrics)
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
        kwdict[:m] = panel.m
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
        kwdict[:m] = panel.m
    end
    return fit(spec, panel.data; pairs(kwdict)...)
end
