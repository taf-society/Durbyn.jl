"""
    fit_grouped.jl

Helper functions for fitting models to grouped/panel data.
"""

import Tables

"""
    fit_grouped(spec::ArimaSpec, data; m, groupby, parallel, fail_fast, kwargs...)

Internal function to fit ARIMA models to grouped data.

This function is called by `fit()` when `groupby` is specified.
It handles data grouping, parallel processing, and error collection.
"""
function fit_grouped(spec::ArimaSpec, data;
                     m::Union{Int, Nothing} = nothing,
                     groupby::Union{Symbol, Vector{Symbol}},
                     datecol::Union{Symbol, Nothing} = nothing,
                     parallel::Bool = true,
                     fail_fast::Bool = false,
                     kwargs...)
    groupby_cols = groupby isa Symbol ? [groupby] : groupby

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    if !haskey(tbl, target_col)
        available_cols = join(string.(keys(tbl)), ", ")
        throw(ArgumentError(
            "Target variable ':$(target_col)' not found in data. " *
            "Available columns: $(available_cols)"
        ))
    end

    target_vector = tbl[target_col]
    if !(target_vector isa AbstractVector)
        throw(ArgumentError(
            "Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"
        ))
    end
    n_total_rows = length(target_vector)

    seasonal_period = if !isnothing(m)
        m
    elseif !isnothing(spec.m)
        spec.m
    else
        throw(ArgumentError(
            "Seasonal period 'm' must be specified either in ArimaSpec or as kwarg to fit(). " *
            "Example: fit(spec, data, m = 12, groupby = [:product])"
        ))
    end

    if seasonal_period < 1
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))
    end

    var_terms = filter(t -> isa(t, VarTerm), spec.formula.terms)
    xreg_cols = Symbol[vt.name for vt in var_terms]
    xreg_formula_cols = _xreg_formula_symbols(spec.xreg_formula)
    auto_xreg_cols = Symbol[]
    if spec.auto_xreg
        exclude = _auto_exclusion_set(target_col, groupby_cols, datecol)
        auto_xreg_cols, skipped = _collect_auto_xreg_columns(tbl, exclude, n_total_rows)
        _warn_skipped_auto(skipped)
        if isempty(auto_xreg_cols)
            @info "Automatic exogenous selection requested ('.') but no eligible numeric columns were found after exclusions in grouped data."
        end
    end

    if !isempty(auto_xreg_cols)
        append!(xreg_cols, auto_xreg_cols)
    end
    if !isempty(xreg_cols)
        unique!(xreg_cols)
    end

    println("Grouping data by ", join(groupby_cols, ", "), "...")
    start_time = time()
    grouped_data = group_data(tbl, groupby_cols, target_col, xreg_cols, xreg_formula_cols, auto_xreg_cols)
    n_groups = length(grouped_data)
    println("Found ", n_groups, " groups")

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))
    if spec.auto_xreg && haskey(fit_options, :xreg)
        throw(ArgumentError("Cannot supply xreg kwargs when formula uses '.' for automatic exogenous selection."))
    end

    models = Dict{NamedTuple, Union{AbstractFittedModel, Exception}}()

    if parallel && Threads.nthreads() > 1
        
        println("Fitting models in parallel (", Threads.nthreads(), " threads)...")

        group_keys = collect(keys(grouped_data))
        results = Vector{Union{FittedArima, Exception}}(undef, n_groups)

        completed = Threads.Atomic{Int}(0)

        Threads.@threads for i in 1:n_groups
            key = group_keys[i]
            group_data_i = grouped_data[key]

            try
                
                fitted = fit_single_group(spec, group_data_i, seasonal_period, fit_options, auto_xreg_cols)
                results[i] = fitted
                
                n_done = Threads.atomic_add!(completed, 1) + 1
                if n_done % max(1, div(n_groups, 20)) == 0
                    pct = round(100 * n_done / n_groups, digits=1)
                    println("  Progress: ", n_done, "/", n_groups, " (", pct, "%)")
                end
            catch err
                results[i] = err

                if fail_fast
                    error("Failed on group $(key): ", err)
                end

                n_done = Threads.atomic_add!(completed, 1) + 1
                if n_done % max(1, div(n_groups, 20)) == 0
                    pct = round(100 * n_done / n_groups, digits=1)
                    println("  Progress: ", n_done, "/", n_groups, " (", pct, "%)")
                end
            end
        end

        for (i, key) in enumerate(group_keys)
            models[key] = results[i]
        end
    else
        
        println("Fitting models sequentially...")

        for (i, (key, group_data_i)) in enumerate(grouped_data)
            try
                
                fitted = fit_single_group(spec, group_data_i, seasonal_period, fit_options, auto_xreg_cols)
                models[key] = fitted
                
                if i % max(1, div(n_groups, 20)) == 0
                    pct = round(100 * i / n_groups, digits=1)
                    println("  Progress: ", i, "/", n_groups, " (", pct, "%)")
                end
            catch err
                models[key] = err
                
                if fail_fast
                    error("Failed on group $(key): ", err)
                end
                
                if i % max(1, div(n_groups, 20)) == 0
                    pct = round(100 * i / n_groups, digits=1)
                    println("  Progress: ", i, "/", n_groups, " (", pct, "%)")
                end
            end
        end
    end

    
    fit_time = time() - start_time
    println("Completed in ", round(fit_time, digits=2), "s")

    
    metadata = Dict{Symbol, Any}(
        :fit_time => fit_time,
        :parallel => parallel,
        :m => seasonal_period
    )

    return GroupedFittedModels(spec, models, groupby_cols, metadata)
end

"""
    fit_grouped(spec::EtsSpec, data; m, groupby, parallel, fail_fast, kwargs...)

Internal helper to fit ETS specifications on grouped data.
"""
function fit_grouped(spec::EtsSpec, data;
                     m::Union{Int, Nothing} = nothing,
                     groupby::Union{Symbol, Vector{Symbol}},
                     datecol::Union{Symbol, Nothing} = nothing,
                     parallel::Bool = true,
                     fail_fast::Bool = false,
                     kwargs...)
    groupby_cols = groupby isa Symbol ? [groupby] : groupby

    tbl = Tables.columntable(data)

    target_col = spec.formula.target
    if !haskey(tbl, target_col)
        available_cols = join(string.(keys(tbl)), ", ")
        throw(ArgumentError(
            "Target variable ':$(target_col)' not found in data. " *
            "Available columns: $(available_cols)"
        ))
    end

    target_vector = tbl[target_col]
    target_vector isa AbstractVector ||
        throw(ArgumentError(
            "Target variable ':$(target_col)' must be a vector, got $(typeof(target_vector))"
        ))

    seasonal_period = isnothing(m) ? (isnothing(spec.m) ? 1 : spec.m) : m
    seasonal_period >= 1 ||
        throw(ArgumentError("Seasonal period 'm' must be >= 1, got $(seasonal_period)"))

    println("Grouping data by ", join(groupby_cols, ", "), "...")
    grouped_data = group_data(tbl, groupby_cols, target_col, Symbol[], Symbol[], Symbol[])
    n_groups = length(grouped_data)
    println("Found ", n_groups, " groups")

    fit_options = merge(spec.options, Dict{Symbol, Any}(kwargs))

    models = Dict{NamedTuple, Union{AbstractFittedModel, Exception}}()

    start_time = time()

    if parallel && Threads.nthreads() > 1
        println("Fitting models in parallel (", Threads.nthreads(), " threads)...")
        group_keys = collect(keys(grouped_data))
        results = Vector{Union{FittedEts, Exception}}(undef, n_groups)
        completed = Threads.Atomic{Int}(0)
        progress_stride = max(1, div(n_groups, 20))

        Threads.@threads for idx in 1:n_groups
            key = group_keys[idx]
            group_data_i = grouped_data[key]
            try
                results[idx] = fit_single_group(spec, group_data_i, seasonal_period, fit_options)
            catch err
                results[idx] = err
                if fail_fast
                    error("Failed on group $(key): ", err)
                end
            end

            done = Threads.atomic_add!(completed, 1) + 1
            if done % progress_stride == 0
                pct = round(100 * done / n_groups, digits=1)
                println("  Progress: ", done, "/", n_groups, " (", pct, "%)")
            end
        end

        for (idx, key) in enumerate(group_keys)
            models[key] = results[idx]
        end
    else
        println("Fitting models sequentially...")
        progress_stride = max(1, div(n_groups, 20))
        idx = 0
        for (key, group_data_i) in grouped_data
            idx += 1
            try
                models[key] = fit_single_group(spec, group_data_i, seasonal_period, fit_options)
            catch err
                models[key] = err
                if fail_fast
                    error("Failed on group $(key): ", err)
                end
            end

            if idx % progress_stride == 0
                pct = round(100 * idx / n_groups, digits=1)
                println("  Progress: ", idx, "/", n_groups, " (", pct, "%)")
            end
        end
    end

    fit_time = time() - start_time
    println("Completed in ", round(fit_time, digits=2), "s")

    metadata = Dict{Symbol, Any}(
        :fit_time => fit_time,
        :parallel => parallel,
        :m => seasonal_period
    )

    return GroupedFittedModels(spec, models, groupby_cols, metadata)
end

"""
    fit_single_group(spec, group_data, seasonal_period, fit_options, auto_xreg_cols)

Fit a single ARIMA model to one group's data.

# Arguments
- `spec::ArimaSpec` - Model specification
- `group_data::NamedTuple` - Data for this group (target and xreg columns)
- `seasonal_period::Int` - Seasonal period
- `fit_options::Dict` - Additional fitting options

# Returns
`FittedArima` - Fitted model for this group
"""
function fit_single_group(spec::ArimaSpec,
                          group_data::NamedTuple,
                          seasonal_period::Int,
                          fit_options::Dict,
                          auto_xreg_cols::Vector{Symbol})
    parent_mod = parentmodule(@__MODULE__)
    Utils_mod = getfield(parent_mod, :Utils)
    Arima_mod = getfield(parent_mod, :Arima)

    tbl = Tables.columntable(group_data)
    target_col = spec.formula.target
    if !haskey(tbl, target_col)
        available_cols = join(string.(keys(tbl)), ", ")
        throw(ArgumentError(
            "Target variable ':$(target_col)' not found in grouped data. " *
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

    design_matrix, design_cols = _build_xreg_formula_matrix(spec, tbl, n_rows, Utils_mod)
    local_options = copy(fit_options)
    if !isnothing(design_matrix) && size(design_matrix.data, 2) > 0
        local_options[:xreg] = design_matrix
    end

    if !isempty(auto_xreg_cols)
        auto_matrix = _build_auto_xreg_matrix(tbl, auto_xreg_cols, n_rows, Utils_mod)
        local_options[:xreg] = auto_matrix
    end

    arima_fit = Arima_mod.auto_arima(spec.formula, group_data, seasonal_period; pairs(local_options)...)

    var_terms = filter(t -> isa(t, VarTerm), spec.formula.terms)
    xreg_cols = Symbol[vt.name for vt in var_terms]
    if !isempty(design_cols)
        append!(xreg_cols, design_cols)
    end
    if !isempty(auto_xreg_cols)
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
        group_data,
        seasonal_period
    )
end

function fit_single_group(spec::EtsSpec,
                          group_data::NamedTuple,
                          seasonal_period::Int,
                          base_options::Dict{Symbol, Any})
    parent_mod = parentmodule(@__MODULE__)
    Exp_mod = getfield(parent_mod, :ExponentialSmoothing)

    tbl = Tables.columntable(group_data)
    target_col = spec.formula.target
    target_vector = tbl[target_col]

    el = Base.nonmissingtype(eltype(target_vector))
    el <: Number ||
        throw(ArgumentError(
            "Target variable ':$(target_col)' must be numeric, got element type $(eltype(target_vector))"
        ))

    fit_options = copy(base_options)
    damped = haskey(fit_options, :damped) ? pop!(fit_options, :damped) : spec.damped
    if !(damped === nothing || damped isa Bool)
        throw(ArgumentError("damped option must be Bool or nothing, got $(typeof(damped))"))
    end

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
