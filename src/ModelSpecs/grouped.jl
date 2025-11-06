"""
    grouped.jl

Grouped/panel data support for fitting models to multiple time series.

This file provides functionality for fitting model specifications to grouped data,
enabling efficient parallel processing of thousands of time series.
"""

import Tables
import ..Generics: forecast, Forecast

"""
    GroupedFittedModels <: AbstractFittedModel

Container for fitted models from grouped/panel data.

Stores fitted models for each group, along with metadata about successful
and failed fits. Supports efficient lookup, iteration, and forecasting.

# Fields
- `spec::AbstractModelSpec` - Original model specification used for all groups
- `models::Dict{NamedTuple, Union{AbstractFittedModel, Exception}}` - Fitted models or errors per group
- `groups::Vector{NamedTuple}` - Sorted list of group keys for consistent ordering
- `groupby_cols::Vector{Symbol}` - Column names used for grouping
- `successful::Int` - Number of successfully fitted models
- `failed::Int` - Number of failed fits
- `metadata::Dict{Symbol, Any}` - Additional metadata (timing, settings, etc.)

# Examples
```julia
# Fit to grouped data
spec = ArimaSpec(@formula(sales = p(1) + d(1) + q(1)))
fitted = fit(spec, data, m = 7, groupby = [:product, :location])

# Access specific group
model = fitted[(product=1, location="US")]

# Iterate over all groups
for (key, model) in fitted
    println("Group: ", key)
    if model isa Exception
        println("  Failed: ", model)
    else
        println("  AIC: ", model.fit.aic)
    end
end

# Get summary
summary(fitted)
```

# See Also
- [`fit`](@ref)
- [`forecast`](@ref)
"""
struct GroupedFittedModels <: AbstractFittedModel
    spec::AbstractModelSpec
    models::Dict{NamedTuple, Union{AbstractFittedModel, Exception}}
    groups::Vector{NamedTuple}
    groupby_cols::Vector{Symbol}
    successful::Int
    failed::Int
    metadata::Dict{Symbol, Any}

    function GroupedFittedModels(
        spec::AbstractModelSpec,
        models::Dict{NamedTuple, Union{AbstractFittedModel, Exception}},
        groupby_cols::Vector{Symbol},
        metadata::Dict{Symbol, Any} = Dict{Symbol, Any}()
    )
    
        groups = sort(collect(keys(models)))

        
        successful = count(v -> !(v isa Exception), values(models))
        failed = count(v -> v isa Exception, values(models))

        new(spec, models, groups, groupby_cols, successful, failed, metadata)
    end
end

"""
    GroupedForecasts

Container for forecasts generated from grouped fitted models.

Stores forecast results for each group while tracking success and failure
counts. Provides convenient lookup and iteration over group forecasts.
"""
struct GroupedForecasts
    spec::AbstractModelSpec
    forecasts::Dict{NamedTuple, Any}
    groups::Vector{NamedTuple}
    successful::Int
    failed::Int
    metadata::Dict{Symbol, Any}

    function GroupedForecasts(
        spec::AbstractModelSpec,
        forecasts::Dict{NamedTuple, Any},
        groups::Vector{NamedTuple},
        metadata::Dict{Symbol, Any} = Dict{Symbol, Any}()
    )
        successful = count(v -> !(v isa Exception), values(forecasts))
        failed = count(v -> v isa Exception, values(forecasts))
        new(spec, forecasts, groups, successful, failed, metadata)
    end
end

"""
    Base.getindex(fitted::GroupedFittedModels, key::NamedTuple)

Access fitted model for a specific group by NamedTuple key.

# Examples
```julia
model = fitted[(product=1, location="US")]
```
"""
Base.getindex(fitted::GroupedFittedModels, key::NamedTuple) = fitted.models[key]

"""
    Base.iterate(fitted::GroupedFittedModels, state=1)

Iterate over all groups and their fitted models.

Yields `(key, model)` tuples where key is NamedTuple and model is
either a fitted model or an Exception.
"""
function Base.iterate(fitted::GroupedFittedModels, state=1)
    if state > length(fitted.groups)
        return nothing
    end
    key = fitted.groups[state]
    return (key, fitted.models[key]), state + 1
end

Base.length(fitted::GroupedFittedModels) = length(fitted.groups)

"""
    Base.getindex(fc::GroupedForecasts, key::NamedTuple)

Access forecast for a specific group by NamedTuple key.
"""
Base.getindex(fc::GroupedForecasts, key::NamedTuple) = fc.forecasts[key]

function Base.iterate(fc::GroupedForecasts, state=1)
    if state > length(fc.groups)
        return nothing
    end
    key = fc.groups[state]
    return (key, fc.forecasts[key]), state + 1
end

Base.length(fc::GroupedForecasts) = length(fc.groups)

"""
    successful_models(fitted::GroupedFittedModels)

Return dictionary of successfully fitted models only (excluding failures).
"""
function successful_models(fitted::GroupedFittedModels)
    return Dict(k => v for (k, v) in fitted.models if !(v isa Exception))
end

"""
    failed_groups(fitted::GroupedFittedModels)

Return vector of group keys that failed to fit.
"""
function failed_groups(fitted::GroupedFittedModels)
    return [k for (k, v) in fitted.models if v isa Exception]
end

"""
    errors(fitted::GroupedFittedModels)

Return dictionary of group keys to exceptions for failed fits.
"""
function errors(fitted::GroupedFittedModels)
    return Dict(k => v for (k, v) in fitted.models if v isa Exception)
end

"""
    successful_forecasts(fc::GroupedForecasts)

Return dictionary of successful forecasts only.
"""
function successful_forecasts(fc::GroupedForecasts)
    return Dict(k => v for (k, v) in fc.forecasts if !(v isa Exception))
end

"""
    failed_groups(fc::GroupedForecasts)

Return vector of group keys that failed to forecast.
"""
function failed_groups(fc::GroupedForecasts)
    return [k for (k, v) in fc.forecasts if v isa Exception]
end

"""
    errors(fc::GroupedForecasts)

Return dictionary of group keys to exceptions for failed forecasts.
"""
function errors(fc::GroupedForecasts)
    return Dict(k => v for (k, v) in fc.forecasts if v isa Exception)
end

function Base.show(io::IO, fitted::GroupedFittedModels)
    n_groups = length(fitted.groups)
    pct_success = round(100 * fitted.successful / n_groups, digits=1)

    print(io, "GroupedFittedModels: ")
    print(io, n_groups, " groups, ")
    print(io, fitted.successful, " successful (", pct_success, "%)")

    if fitted.failed > 0
        print(io, ", ", fitted.failed, " failed")
    end
end

function Base.show(io::IO, ::MIME"text/plain", fitted::GroupedFittedModels)
    n_groups = length(fitted.groups)
    pct_success = round(100 * fitted.successful / n_groups, digits=1)
    pct_failed = round(100 * fitted.failed / n_groups, digits=1)

    println(io, "GroupedFittedModels")
    println(io, "  Specification: ", fitted.spec)
    println(io, "  Grouped by: ", join(fitted.groupby_cols, ", "))
    println(io, "  Total groups: ", n_groups)
    println(io, "  Successful: ", fitted.successful, " (", pct_success, "%)")
    println(io, "  Failed: ", fitted.failed, " (", pct_failed, "%)")

    # Show timing if available
    if haskey(fitted.metadata, :fit_time)
        fit_time = fitted.metadata[:fit_time]
        println(io, "  Fit time: ", round(fit_time, digits=2), "s")
        avg_time = round(fit_time / n_groups * 1000, digits=1)
        println(io, "  Avg per group: ", avg_time, "ms")
    end

    # Show parallel setting if available
    if haskey(fitted.metadata, :parallel)
        parallel = fitted.metadata[:parallel]
        println(io, "  Parallel: ", parallel)
    end

    # Sample a few successful models for summary
    successful = successful_models(fitted)
    if !isempty(successful)
        println(io)
        println(io, "  Sample groups:")
        for (i, (key, model)) in enumerate(Iterators.take(successful, 3))
            println(io, "    ", key, ": ", model)
            if i >= 3
                break
            end
        end
    end

    # Show error summary if there are failures
    if fitted.failed > 0
        println(io)
        println(io, "  Error summary:")
        error_counts = Dict{String, Int}()
        for (_, err) in errors(fitted)
            err_type = string(typeof(err))
            error_counts[err_type] = get(error_counts, err_type, 0) + 1
        end
        for (err_type, count) in sort(collect(error_counts), by=x->x[2], rev=true)
            pct = round(100 * count / fitted.failed, digits=1)
            println(io, "    ", err_type, ": ", count, " (", pct, "% of failures)")
        end
    end
end


"""
    group_data(data, groupby_cols::Vector{Symbol}, target::Symbol, xreg_cols::Vector{Symbol})

Group data by specified columns and extract target and xreg for each group.

Returns Dict mapping group keys (NamedTuples) to data NamedTuples.

# Arguments
- `data` - Tables.jl-compatible data source
- `groupby_cols` - Columns to group by
- `target` - Target variable column
- `xreg_cols` - Exogenous variable columns (can be empty)

# Returns
`Dict{NamedTuple, NamedTuple}` - Maps group keys to data for that group
"""
function group_data(data,
                    groupby_cols::Vector{Symbol},
                    target::Symbol,
                    xreg_cols::Vector{Symbol},
                    xreg_formula_cols::Vector{Symbol},
                    auto_xreg_cols::Vector{Symbol})
    # Convert to columntable for efficient access
    tbl = data isa NamedTuple ? data : Tables.columntable(data)

    # Validate columns exist
    for col in groupby_cols
        if !haskey(tbl, col)
            available = join(string.(keys(tbl)), ", ")
            throw(ArgumentError(
                "Groupby column ':$(col)' not found in data. " *
                "Available columns: $(available)"
            ))
        end
    end

    if !haskey(tbl, target)
        available = join(string.(keys(tbl)), ", ")
        throw(ArgumentError(
            "Target column ':$(target)' not found in data. " *
            "Available columns: $(available)"
        ))
    end

    for col in xreg_cols
        if !haskey(tbl, col)
            available = join(string.(keys(tbl)), ", ")
            throw(ArgumentError(
                "Exogenous variable ':$(col)' not found in data. " *
                "Available columns: $(available)"
            ))
        end
    end

    for col in xreg_formula_cols
        if !haskey(tbl, col)
            available = join(string.(keys(tbl)), ", ")
            throw(ArgumentError(
                "Exogenous variable ':$(col)' not found in data. " *
                "Available columns: $(available)"
            ))
        end
    end

    # Build group indices
    n = length(tbl[first(keys(tbl))])  # Number of rows
    group_indices = Dict{NamedTuple, Vector{Int}}()

    for i in 1:n
        # Build key for this row
        key_values = [tbl[col][i] for col in groupby_cols]
        key = NamedTuple{Tuple(groupby_cols)}(key_values)

        # Add index to this group
        if !haskey(group_indices, key)
            group_indices[key] = Int[]
        end
        push!(group_indices[key], i)
    end

    all_xreg_cols = copy(xreg_cols)
    for col in xreg_formula_cols
        if !(col in all_xreg_cols)
            push!(all_xreg_cols, col)
        end
    end
    for col in auto_xreg_cols
        if !(col in all_xreg_cols)
            push!(all_xreg_cols, col)
        end
    end

    # Extract data for each group
    grouped_data = Dict{NamedTuple, NamedTuple}()

    for (key, indices) in group_indices
        # Extract target
        target_data = tbl[target][indices]

        # Extract xreg if present
        if isempty(all_xreg_cols)
            grouped_data[key] = (; target => target_data)
        else
            xreg_data = NamedTuple{Tuple(all_xreg_cols)}(
                tuple([tbl[col][indices] for col in all_xreg_cols]...)
            )
            grouped_data[key] = merge((; target => target_data), xreg_data)
        end
    end

    return grouped_data
end


function Base.show(io::IO, fc::GroupedForecasts)
    n_groups = length(fc.groups)
    pct_success = round(100 * fc.successful / max(n_groups, 1), digits=1)

    print(io, "GroupedForecasts: ")
    print(io, n_groups, " groups, ")
    print(io, fc.successful, " successful (", pct_success, "%)")

    if fc.failed > 0
        print(io, ", ", fc.failed, " failed")
    end
end

_level_label(level::Real) = begin
    lvl = level >= 1 ? level : level * 100
    string(round(Int, lvl))
end
_level_label(level) = string(level)

function _extract_interval_value(x, step::Int, idx::Int, horizon::Int)
    if x isa AbstractMatrix
        return x[step, idx]
    elseif x isa AbstractVector
        if length(x) == horizon && idx == 1 && x isa AbstractVector{<:Number}
            return x[step]
        elseif idx <= length(x)
            element = x[idx]
            if element isa AbstractVector
                return element[step]
            elseif element isa Number
                return element
            end
        end
        return missing
    elseif isnothing(x)
        return missing
    else
        return x
    end
end

function forecast_table(fc::Forecast)
    mean_vec = fc.mean
    horizon = length(mean_vec)
    steps = collect(1:horizon)
    method_col = fill(fc.method, horizon)

    level_vals = fc.level isa AbstractVector ? [Float64(l) for l in fc.level] : Float64[]
    n_levels = length(level_vals)
    lower_cols = [Vector{Union{Missing, Float64}}(undef, horizon) for _ in 1:n_levels]
    upper_cols = [Vector{Union{Missing, Float64}}(undef, horizon) for _ in 1:n_levels]

    for step in 1:horizon
        for j in 1:n_levels
            lv = _extract_interval_value(fc.lower, step, j, horizon)
            uv = _extract_interval_value(fc.upper, step, j, horizon)
            lower_cols[j][step] = lv === missing ? missing : Float64(lv)
            upper_cols[j][step] = uv === missing ? missing : Float64(uv)
        end
    end

    column_syms = Symbol[:step, :mean]
    column_vals = Any[steps, Float64.(mean_vec)]

    for (j, lvl) in enumerate(level_vals)
        label = _level_label(lvl)
        push!(column_syms, Symbol("lower_", label))
        push!(column_vals, lower_cols[j])
        push!(column_syms, Symbol("upper_", label))
        push!(column_vals, upper_cols[j])
    end

    push!(column_syms, :model)
    push!(column_vals, method_col)

    return NamedTuple{Tuple(column_syms)}(Tuple(column_vals))
end

function forecast_table(fc::GroupedForecasts; include_failures::Bool = false)
    group_keys = fc.groups
    isempty(group_keys) && throw(ArgumentError("GroupedForecasts has no groups."))

    success_found = false
    group_names = Symbol[]
    group_columns = Vector{Any}[]
    level_values = Float64[]
    level_map = Dict{Float64, Int}()
    key_example = nothing

    for key in group_keys
        result = fc.forecasts[key]
        if result isa Forecast
            if !success_found
                success_found = true
                key_example = key
                group_names = Symbol.(propertynames(key))
                group_columns = [Vector{typeof(getfield(key, name))}() for name in group_names]
            end
            for lvl in Float64.(result.level)
                if !haskey(level_map, lvl)
                    push!(level_values, lvl)
                    level_map[lvl] = length(level_values)
                end
            end
        end
    end

    success_found || throw(ArgumentError("No successful forecasts available."))

    step_col = Int[]
    mean_col = Float64[]
    method_col = String[]
    lower_cols = [Vector{Union{Missing, Float64}}() for _ in 1:length(level_values)]
    upper_cols = [Vector{Union{Missing, Float64}}() for _ in 1:length(level_values)]

    for key in group_keys
        result = fc.forecasts[key]
        if result isa Forecast
            mean_vec = result.mean
            horizon = length(mean_vec)
            levels = Float64.(result.level)

            for step in 1:horizon
                for (idx, name) in enumerate(group_names)
                    push!(group_columns[idx], getfield(key, name))
                end
                push!(step_col, step)
                push!(mean_col, Float64(mean_vec[step]))

                present = falses(length(level_values))
                for (j, lvl) in enumerate(levels)
                    idx_lvl = level_map[lvl]
                    present[idx_lvl] = true
                    lower_val = _extract_interval_value(result.lower, step, j, horizon)
                    upper_val = _extract_interval_value(result.upper, step, j, horizon)
                    push!(lower_cols[idx_lvl], lower_val === missing ? missing : Float64(lower_val))
                    push!(upper_cols[idx_lvl], upper_val === missing ? missing : Float64(upper_val))
                end
                for idx_lvl in 1:length(level_values)
                    if !present[idx_lvl]
                        push!(lower_cols[idx_lvl], missing)
                        push!(upper_cols[idx_lvl], missing)
                    end
                end
                push!(method_col, result.method)
            end
        elseif include_failures && result isa Exception
            continue
        end
    end

    if isempty(mean_col)
        throw(ArgumentError("No successful forecasts available."))
    end

    column_syms = Symbol[]
    column_vals = Any[]
    for (name, vec) in zip(group_names, group_columns)
        push!(column_syms, name)
        push!(column_vals, vec)
    end
    push!(column_syms, :step)
    push!(column_vals, step_col)
    push!(column_syms, :mean)
    push!(column_vals, mean_col)
    for (idx, lvl) in enumerate(level_values)
        label = _level_label(lvl)
        push!(column_syms, Symbol("lower_", label))
        push!(column_vals, lower_cols[idx])
        push!(column_syms, Symbol("upper_", label))
        push!(column_vals, upper_cols[idx])
    end
    push!(column_syms, :model)
    push!(column_vals, method_col)

    return NamedTuple{Tuple(column_syms)}(Tuple(column_vals))
end

function _ensure_model_name(table::NamedTuple, name::String)
    cols = propertynames(table)
    if :model_name in cols
        return table
    end
    len = isempty(cols) ? 0 : length(getfield(table, cols[1]))
    return merge(table, (model_name = fill(name, len),))
end

function _finalize_column(vec::Vector{Any})
    len = length(vec)
    len == 0 && return Vector{Any}()

    has_missing = false
    types = Type[]
    for v in vec
        if v === missing
            has_missing = true
        else
            push!(types, typeof(v))
        end
    end

    if isempty(types)
        return has_missing ? fill(missing, len) : Vector{Any}()
    end

    T = reduce(promote_type, types)
    if has_missing
        return [v === missing ? missing : convert(T, v) for v in vec]
    else
        return [convert(T, v) for v in vec]
    end
end

function forecast_table(collection::ForecastModelCollection; include_failures::Bool = false)
    tables = NamedTuple[]
    ordered_cols = Symbol[]

    for name in collection.names
        result = collection.forecasts[name]
        table = nothing
        if result isa GroupedForecasts
            try
                table = forecast_table(result; include_failures=include_failures)
            catch err
                if include_failures
                    table = (model_name = [name], status = [:error], message = [string(err)])
                else
                    continue
                end
            end
        elseif result isa Forecast
            table = forecast_table(result)
        elseif result isa Exception
            if include_failures
                table = (model_name = [name], status = [:error], message = [string(result)])
            else
                continue
            end
        elseif include_failures
            table = (model_name = [name], status = [:unsupported], message = [string(typeof(result))])
        else
            continue
        end

        table = _ensure_model_name(table, name)
        if !(:model in propertynames(table))
            len = length(getfield(table, propertynames(table)[1]))
            table = merge(table, (model = fill(name, len),))
        end

        push!(tables, table)
        for col in propertynames(table)
            if !(col in ordered_cols)
                push!(ordered_cols, col)
            end
        end
    end

    isempty(tables) && throw(ArgumentError("No successful forecasts available."))

    columns_data = Dict{Symbol, Vector{Any}}()
    for col in ordered_cols
        columns_data[col] = Any[]
    end

    for table in tables
        cols = propertynames(table)
        len = isempty(cols) ? 0 : length(getfield(table, cols[1]))
        for col in ordered_cols
            if col in cols
                append!(columns_data[col], getfield(table, col))
            else
                append!(columns_data[col], fill(missing, len))
            end
        end
    end

    finalized = Tuple(_finalize_column(columns_data[col]) for col in ordered_cols)
    return NamedTuple{Tuple(ordered_cols)}(finalized)
end


function Base.show(io::IO, ::MIME"text/plain", fc::GroupedForecasts)
    n_groups = length(fc.groups)
    pct_success = round(100 * fc.successful / max(n_groups, 1), digits=1)
    pct_failed = round(100 * fc.failed / max(n_groups, 1), digits=1)

    println(io, "GroupedForecasts")
    println(io, "  Specification: ", fc.spec)
    println(io, "  Total groups: ", n_groups)
    println(io, "  Successful: ", fc.successful, " (", pct_success, "%)")
    println(io, "  Failed: ", fc.failed, " (", pct_failed, "%)")

    if haskey(fc.metadata, :h)
        println(io, "  Horizon: ", fc.metadata[:h])
    end
    if haskey(fc.metadata, :level)
        println(io, "  Levels: ", fc.metadata[:level])
    end
    if haskey(fc.metadata, :forecast_time)
        println(io, "  Forecast time: ", round(fc.metadata[:forecast_time], digits=2), "s")
    end

    successful = successful_forecasts(fc)
    if !isempty(successful)
        println(io)
        println(io, "  Sample groups:")
        for (i, (key, forecast_obj)) in enumerate(Iterators.take(successful, 3))
            println(io, "    ", key, ": ", forecast_obj)
            if i >= 3
                break
            end
        end
    end

    if fc.failed > 0
        println(io)
        println(io, "  Error summary:")
        error_counts = Dict{String, Int}()
        for (_, err) in errors(fc)
            err_type = string(typeof(err))
            error_counts[err_type] = get(error_counts, err_type, 0) + 1
        end
        for (err_type, count) in sort(collect(error_counts), by=x->x[2], rev=true)
            pct = round(100 * count / fc.failed, digits=1)
            println(io, "    ", err_type, ": ", count, " (", pct, "% of failures)")
        end
    end
end

"""
    forecast(fitted::GroupedFittedModels; h, level=[80, 95], newdata=nothing, parallel=true, progress=false, kwargs...)

Generate forecasts for each group in a `GroupedFittedModels` container.
"""
function forecast(fitted::GroupedFittedModels;
                  h::Int,
                  level::Vector{<:Real} = [80, 95],
                  newdata = nothing,
                  parallel::Bool = true,
                  progress::Bool = false,
                  kwargs...)
    xreg_cols = _xreg_columns(fitted.spec)

    grouped_newdata = isnothing(newdata) ? nothing :
        _prepare_grouped_newdata(fitted, newdata, h, xreg_cols)

    if isempty(xreg_cols) && !isnothing(newdata)
        @warn "newdata provided but model has no exogenous variables. Ignoring newdata."
        grouped_newdata = nothing
    end

    if !isempty(xreg_cols) && !isnothing(grouped_newdata)
        missing = String[]
        for key in fitted.groups
            if fitted.models[key] isa Exception
                continue
            end
            if !haskey(grouped_newdata, key)
                push!(missing, string(key))
            end
        end
        if !isempty(missing)
            missing_str = join(missing, ", ")
            throw(ArgumentError(
                "newdata missing for groups required for forecasting: $(missing_str)."
            ))
        end
    end

    forecasts = Dict{NamedTuple, Any}()
    groups = fitted.groups
    n_groups = length(groups)
    start_time = time()

    if parallel && Threads.nthreads() > 1
        results = Vector{Any}(undef, n_groups)
        progress_stride = progress ? max(1, div(n_groups, 20)) : typemax(Int)
        completed = Threads.Atomic{Int}(0)

        Threads.@threads for idx in eachindex(groups)
            key = groups[idx]
            model = fitted.models[key]

            results[idx] = _forecast_single_group(
                model,
                key,
                h,
                level,
                grouped_newdata,
                kwargs...
            )

            done = Threads.atomic_add!(completed, 1) + 1
            if progress && done % progress_stride == 0
                pct = round(100 * done / n_groups, digits=1)
                println("  Forecast progress: ", done, "/", n_groups, " (", pct, "%)")
            end
        end

        for (idx, key) in enumerate(groups)
            forecasts[key] = results[idx]
        end
    else
        progress_stride = progress ? max(1, div(n_groups, 20)) : typemax(Int)

        for (idx, key) in enumerate(groups)
            model = fitted.models[key]
            forecasts[key] = _forecast_single_group(
                model,
                key,
                h,
                level,
                grouped_newdata,
                kwargs...
            )

            if progress && idx % progress_stride == 0
                pct = round(100 * idx / n_groups, digits=1)
                println("  Forecast progress: ", idx, "/", n_groups, " (", pct, "%)")
            end
        end
    end

    metadata = Dict{Symbol, Any}(
        :h => h,
        :level => level,
        :forecast_time => time() - start_time,
        :parallel => parallel
    )

    return GroupedForecasts(fitted.spec, forecasts, groups, metadata)
end

function _forecast_single_group(model,
                                key::NamedTuple,
                                h::Int,
                                level,
                                grouped_newdata,
                                kwargs...)
    if model isa Exception
        return model
    end

    group_newdata = isnothing(grouped_newdata) ? nothing : get(grouped_newdata, key, nothing)

    try
        forecast_kwargs = merge((; h = h, level = level), (; kwargs...))
        if !isnothing(group_newdata)
            forecast_kwargs = merge(forecast_kwargs, (; newdata = group_newdata))
        end
        return forecast(model; forecast_kwargs...)
    catch err
        return err
    end
end

function _prepare_grouped_newdata(fitted::GroupedFittedModels,
                                  newdata,
                                  h::Int,
                                  xreg_cols::Vector{Symbol})
    if newdata isa Dict{NamedTuple, Any}
        return Dict{NamedTuple, Any}(newdata)
    elseif Tables.istable(newdata)
        tbl = Tables.columntable(newdata)

        for col in fitted.groupby_cols
            if !haskey(tbl, col)
                available = join(string.(keys(tbl)), ", ")
                throw(ArgumentError(
                    "Groupby column ':$(col)' not found in newdata. Available columns: $(available)"
                ))
            end
        end

        for col in xreg_cols
            if !haskey(tbl, col)
                available = join(string.(keys(tbl)), ", ")
                throw(ArgumentError(
                    "Exogenous variable ':$(col)' not found in newdata. Available columns: $(available)"
                ))
            end
        end

        group_set = Set(fitted.groups)
        grouped = Dict{NamedTuple, Dict{Symbol, Vector{Any}}}()

        n = length(tbl[first(keys(tbl))])
        for i in 1:n
            key_values = [tbl[col][i] for col in fitted.groupby_cols]
            key = NamedTuple{Tuple(fitted.groupby_cols)}(key_values)

            if !(key in group_set)
                continue
            end

            if !haskey(grouped, key)
                grouped[key] = Dict{Symbol, Vector{Any}}()
            end

            for col in xreg_cols
                if !haskey(grouped[key], col)
                    grouped[key][col] = Any[]
                end
                push!(grouped[key][col], tbl[col][i])
            end
        end

        # Convert to NamedTuples
        processed = Dict{NamedTuple, Any}()
        for (key, col_dict) in grouped
            if isempty(xreg_cols)
                processed[key] = nothing
            else
                vectors = Vector{Any}(undef, length(xreg_cols))
                for (i, col) in enumerate(xreg_cols)
                    values = get(col_dict, col, Any[])
                    if length(values) != h
                        throw(ArgumentError(
                            "Group $(key) in newdata has $(length(values)) rows, " *
                            "but forecast horizon is $(h). Each group must provide exactly h rows."
                        ))
                    end
                    vectors[i] = Vector{eltype(tbl[col])}(values)
                end
                processed[key] = NamedTuple{Tuple(xreg_cols)}(Tuple(vectors))
            end
        end

        if !isempty(xreg_cols)
            missing = String[]
            for key in fitted.groups
                if !haskey(processed, key) && !(fitted.models[key] isa Exception)
                    push!(missing, string(key))
                end
            end
            if !isempty(missing)
                missing_str = join(missing, ", ")
                throw(ArgumentError(
                    "newdata missing for groups required for forecasting: $(missing_str)."
                ))
            end
        end

        return processed
    else
        throw(ArgumentError(
            "newdata for grouped forecasts must be a Dict or a Tables.jl-compatible object. " *
            "Got $(typeof(newdata))."
        ))
    end
end

function _xreg_columns(spec::AbstractModelSpec)
    if spec isa ArimaSpec
        var_terms = filter(t -> isa(t, VarTerm), spec.formula.terms)
        cols = Symbol[vt.name for vt in var_terms]
        for sym in _xreg_formula_symbols(spec.xreg_formula)
            if !(sym in cols)
                push!(cols, sym)
            end
        end
        return cols
    elseif spec isa ArarSpec
        # ARAR doesn't support exogenous variables
        return Symbol[]
    end
    return Symbol[]
end
