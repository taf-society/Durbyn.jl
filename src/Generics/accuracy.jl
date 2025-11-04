"""
    accuracy.jl

Forecast accuracy evaluation functions for all forecast types.
"""

import Statistics: mean
import ..Utils: evaluation_metrics
import Tables

"""
    accuracy(forecast, actual; training_data=nothing, by=nothing)

Calculate forecast accuracy metrics by comparing forecast values against actual values.

Works with all forecast types:
- Single `Forecast` from `forecast(fitted_model, h=...)`
- `ForecastModelCollection` from `forecast(fitted_collection, h=...)`
- `GroupedForecasts` from panel/grouped data
- Forecast tables from `forecast_table()`
- Raw vectors

# Arguments
- `forecast`: Forecast object, collection, table, or vector
- `actual`: Actual values as vector or table (must align with forecast structure)
- `training_data`: (Optional) Training series for MASE calculation
- `by`: (Optional) Grouping columns for aggregated accuracy

# Returns
A NamedTuple containing accuracy metrics:
- `ME`: Mean Error
- `RMSE`: Root Mean Squared Error
- `MAE`: Mean Absolute Error
- `MPE`: Mean Percentage Error (%)
- `MAPE`: Mean Absolute Percentage Error (%)
- `MASE`: Mean Absolute Scaled Error (if training_data provided)
- `ACF1`: Autocorrelation of errors at lag 1

# Examples
```julia
# Single model, single series
spec = ArimaSpec(@formula(sales = p(1,3) + q(1,3)))
fitted = fit(spec, data, m = 7)
fc = forecast(fitted, h = 7)
accuracy(fc, test_values)  # test_values is a vector

# Multiple models
models = model(ArimaSpec(...), EtsSpec(...))
fitted = fit(models, data, m = 12)
fc = forecast(fitted, h = 12)
accuracy(fc, test_values)  # Returns accuracy for each model

# Panel data
panel = PanelData(train, groupby = :series, date = :date, m = 12)
fitted = fit(models, panel)
fc = forecast(fitted, h = 12)
accuracy(fc, test_table)
```
"""
function accuracy(forecast_input, actual; training_data=nothing, by=nothing)
    if forecast_input isa AbstractVector{<:Real} && actual isa AbstractVector{<:Real}
        return _compute_accuracy_metrics(forecast_input, actual, training_data)
    end

    if hasproperty(forecast_input, :mean) && hasproperty(forecast_input, :method)
        if isnothing(training_data) && hasproperty(forecast_input, :x)
            training_data = forecast_input.x
        end

        if actual isa AbstractVector{<:Real}
            return _compute_accuracy_metrics(forecast_input.mean, actual, training_data)
        elseif Tables.istable(actual)
            actual_vec = _extract_value_column(actual)
            return _compute_accuracy_metrics(forecast_input.mean, actual_vec, training_data)
        end
    end

    if hasproperty(forecast_input, :names) && hasproperty(forecast_input, :forecasts) &&
       forecast_input.names isa Vector{String}
        return _accuracy_forecast_collection(forecast_input, actual, training_data, by)
    end

    if hasproperty(forecast_input, :forecasts) && hasproperty(forecast_input, :groups) &&
       forecast_input.forecasts isa Dict
        return _accuracy_grouped_forecasts(forecast_input, actual, training_data, by)
    end

    if Tables.istable(forecast_input) && Tables.istable(actual)
        return _accuracy_tables(forecast_input, actual, training_data, by)
    end

    error("Unable to determine forecast type. Got: $(typeof(forecast_input))")
end


"""
    _compute_accuracy_metrics(forecast, actual, training_data)

Core function to compute accuracy metrics from aligned vectors.
"""
function _compute_accuracy_metrics(forecast_vec::AbstractVector,
                                   actual_vec::AbstractVector,
                                   training_data::Union{Nothing, AbstractVector}=nothing)
    n = length(actual_vec)
    if length(forecast_vec) != n
        error("Forecast and actual must have same length. " *
              "Got forecast: $(length(forecast_vec)), actual: $n")
    end

    errors = actual_vec .- forecast_vec

    me = mean(errors)

    rmse = sqrt(mean(errors.^2))

    mae = mean(abs.(errors))

    non_zero_idx = actual_vec .!= 0
    if any(non_zero_idx)
        mpe = mean((errors[non_zero_idx] ./ actual_vec[non_zero_idx]) .* 100)
        mape = mean(abs.((errors[non_zero_idx] ./ actual_vec[non_zero_idx])) .* 100)
    else
        mpe = NaN
        mape = NaN
    end

    mase = if !isnothing(training_data) && length(training_data) > 1
        naive_errors = diff(training_data)
        scale = mean(abs.(naive_errors))
        scale > 0 ? mae / scale : NaN
    else
        nothing
    end

    acf1 = if n > 1
        errors_centered = errors .- mean(errors)
        numerator = sum(errors_centered[1:end-1] .* errors_centered[2:end])
        denominator = sum(errors_centered.^2)
        denominator > 0 ? numerator / denominator : NaN
    else
        NaN
    end

    result = (
        ME = me,
        RMSE = rmse,
        MAE = mae,
        MPE = mpe,
        MAPE = mape,
        ACF1 = acf1
    )

    if !isnothing(mase)
        result = merge(result, (MASE = mase,))
    end

    return result
end


"""
    _accuracy_forecast_collection(fc_collection, actual, training_data, by)

Handle ForecastModelCollection (multiple models).
"""
function _accuracy_forecast_collection(fc_collection, actual, training_data, by)
    model_names = String[]
    metrics_list = []

    for (name, fc_obj) in zip(fc_collection.names,
                               [fc_collection.forecasts[n] for n in fc_collection.names])
        if fc_obj isa Exception
            @warn "Model '$name' failed to forecast. Skipping accuracy calculation."
            continue
        end

        try
            acc = accuracy(fc_obj, actual; training_data=training_data, by=by)
            push!(model_names, name)
            push!(metrics_list, acc)
        catch e
            @warn "Error calculating accuracy for model '$name': $e"
        end
    end

    if isempty(metrics_list)
        error("No successful accuracy calculations")
    end

    return _combine_model_accuracies(model_names, metrics_list)
end

"""
    _accuracy_grouped_forecasts(fc, actual, training_data, by)

Handle GroupedForecasts (single model, multiple series).
"""
function _accuracy_grouped_forecasts(fc, actual, training_data, by)
    if !Tables.istable(actual)
        error("For GroupedForecasts, actual must be a table with group identifiers")
    end

    act_ct = Tables.columntable(actual)

    value_col = _identify_value_column(act_ct)

    act_groups = _index_actual_by_groups(act_ct, value_col)
    
    return _accuracy_by_groups(fc, act_groups, training_data)
end

"""
    _accuracy_tables(fc_table, actual, training_data, by)

Handle table-based forecasts.
"""
function _accuracy_tables(fc_table, actual, training_data, by)
    fc_ct = Tables.columntable(fc_table)
    act_ct = Tables.columntable(actual)
    
    has_groups = _has_grouping_columns(fc_ct)

    if has_groups
        return _accuracy_table_grouped(fc_ct, act_ct, by)
    else
        return _accuracy_table_simple(fc_ct, act_ct)
    end
end

"""
    _extract_value_column(table)

Extract the value column from a table.
"""
function _extract_value_column(table)
    ct = Tables.columntable(table)
    col_names = propertynames(ct)

    for name in [:value, :y, :actual, :observed]
        if name in col_names
            return ct[name]
        end
    end
    
    for name in col_names
        if name ∉ [:date, :time, :step, :series, :model, :model_name, :group]
            col = ct[name]
            if eltype(col) <: Real
                return col
            end
        end
    end

    error("Could not identify value column. Available: $(col_names)")
end

"""
    _identify_value_column(ct::NamedTuple)

Identify which column contains the actual values.
"""
function _identify_value_column(ct::NamedTuple)
    col_names = propertynames(ct)

    for name in [:value, :y, :actual, :observed]
        if name in col_names
            return name
        end
    end

    for name in col_names
        if name ∉ [:date, :time, :step, :series, :model, :model_name, :group]
            col = ct[name]
            if eltype(col) <: Real
                return name
            end
        end
    end

    error("Could not identify value column. Available: $(col_names)")
end

"""
    _has_grouping_columns(ct::NamedTuple)

Check if table has grouping columns.
"""
function _has_grouping_columns(ct::NamedTuple)
    col_names = propertynames(ct)
    return any(col in col_names for col in [:series, :model, :model_name, :group])
end

"""
    _index_actual_by_groups(act_ct, value_col)

Build index of actual values organized by group keys.
Sorts by date if available to ensure proper alignment with forecasts.
"""
function _index_actual_by_groups(act_ct::NamedTuple, value_col::Symbol)
    col_names = propertynames(act_ct)
    n = length(act_ct[first(col_names)])
    
    group_cols = Symbol[]
    if :series in col_names
        push!(group_cols, :series)
    end
    
    has_date = :date in col_names
    
    groups_dict = Dict{NamedTuple, Vector{Float64}}()

    if isempty(group_cols)
        
        if has_date
            # Sort by date
            date_col = act_ct.date
            sorted_indices = sortperm(collect(date_col))
            groups_dict[NamedTuple()] = Float64[act_ct[value_col][i] for i in sorted_indices]
        else
            groups_dict[NamedTuple()] = Float64.(act_ct[value_col])
        end
    else
        
        temp_groups = Dict{NamedTuple, Vector{Tuple{Any, Float64}}}()

        for i in 1:n
            key_vals = [act_ct[col][i] for col in group_cols]
            key = NamedTuple{Tuple(group_cols)}(Tuple(key_vals))

            value = Float64(act_ct[value_col][i])
            date_val = has_date ? act_ct.date[i] : i

            if !haskey(temp_groups, key)
                temp_groups[key] = Tuple{Any, Float64}[]
            end
            push!(temp_groups[key], (date_val, value))
        end

        
        for (key, data) in temp_groups
            sorted_data = sort(data, by = x -> x[1])
            groups_dict[key] = Float64[x[2] for x in sorted_data]
        end
    end

    return groups_dict
end

"""
    _accuracy_by_groups(fc, act_groups, training_data)

Calculate accuracy for each group.
"""
function _accuracy_by_groups(fc, act_groups::Dict, training_data)
    group_keys = Symbol[]
    group_values = Vector{Any}[]
    metrics = Dict{Symbol, Vector{Float64}}()
    
    for m in [:ME, :RMSE, :MAE, :MPE, :MAPE, :ACF1]
        metrics[m] = Float64[]
    end

    n_successful = 0
    n_failed = 0
    n_missing = 0

    
    for (key, fc_obj) in fc.forecasts
        if fc_obj isa Exception
            n_failed += 1
            continue
        end

        if !haskey(act_groups, key)
            n_missing += 1
            
            if n_missing == 1
                @warn "No actual data for group $key (and possibly others)"
            end
            continue
        end

        actual_vec = act_groups[key]
        forecast_vec = fc_obj.mean

        
        h = length(forecast_vec)
        if length(actual_vec) > h
            actual_vec = actual_vec[1:h]
        elseif length(actual_vec) < h
            @warn "Group $key: actual has $(length(actual_vec)) values but forecast has $h. Using available values."
            forecast_vec = forecast_vec[1:length(actual_vec)]
        end

        if isempty(actual_vec) || isempty(forecast_vec)
            continue
        end

        
        acc = _compute_accuracy_metrics(forecast_vec, actual_vec, training_data)

        # Store group key
        if isempty(group_keys)
            group_keys = collect(Symbol, keys(key))
            for _ in 1:length(group_keys)
                push!(group_values, Any[])
            end
        end

        for (i, gk) in enumerate(group_keys)
            push!(group_values[i], key[gk])
        end

        
        for (metric_name, metric_val) in pairs(acc)
            if metric_name in keys(metrics)
                push!(metrics[metric_name], metric_val)
            end
        end

        n_successful += 1
    end
    
    if n_missing > 1
        @warn "Missing actual data for $n_missing groups"
    end
    if n_failed > 0
        @warn "Skipped $n_failed groups due to forecast failures"
    end
    if n_successful == 0
        error("No successful accuracy calculations. Check that test data has matching :series and :value columns")
    end

    result_cols = Symbol[]
    result_vals = Any[]

    for (i, gk) in enumerate(group_keys)
        push!(result_cols, gk)
        push!(result_vals, group_values[i])
    end

    for m in [:ME, :RMSE, :MAE, :MPE, :MAPE, :ACF1]
        push!(result_cols, m)
        push!(result_vals, metrics[m])
    end

    return NamedTuple{Tuple(result_cols)}(Tuple(result_vals))
end

"""
    _combine_model_accuracies(model_names, metrics_list)

Combine accuracy results from multiple models.
"""
function _combine_model_accuracies(model_names::Vector{String}, metrics_list::Vector)
    first_metric = metrics_list[1]

    if first_metric isa NamedTuple && all(v isa Real || isnothing(v) for v in values(first_metric))
        
        model_col = String[]
        metric_cols = Dict{Symbol, Vector{Any}}()

        for key in keys(first_metric)
            metric_cols[key] = Any[]
        end

        for (name, metrics) in zip(model_names, metrics_list)
            push!(model_col, name)
            for key in keys(metrics)
                push!(metric_cols[key], metrics[key])
            end
        end

        result_cols = [:model]
        result_vals = Any[model_col]

        for key in [:ME, :RMSE, :MAE, :MPE, :MAPE, :MASE, :ACF1]
            if haskey(metric_cols, key)
                push!(result_cols, key)
                push!(result_vals, metric_cols[key])
            end
        end

        return NamedTuple{Tuple(result_cols)}(Tuple(result_vals))
    else
        all_cols = Set{Symbol}()
        for metrics in metrics_list
            for col in propertynames(metrics)
                push!(all_cols, col)
            end
        end

        combined_cols = Dict{Symbol, Vector{Any}}()
        combined_cols[:model] = String[]

        for col in all_cols
            combined_cols[col] = Any[]
        end

        for (name, metrics) in zip(model_names, metrics_list)
            n_rows = length(metrics[first(propertynames(metrics))])

            append!(combined_cols[:model], fill(name, n_rows))

            for col in all_cols
                if col in propertynames(metrics)
                    append!(combined_cols[col], metrics[col])
                else
                    append!(combined_cols[col], fill(missing, n_rows))
                end
            end
        end

        result_cols = [:model]
        result_vals = Any[combined_cols[:model]]

        for col in all_cols
            if col != :model
                push!(result_cols, col)
                push!(result_vals, combined_cols[col])
            end
        end

        return NamedTuple{Tuple(result_cols)}(Tuple(result_vals))
    end
end

"""
    _accuracy_table_simple(fc_ct, act_ct)

Calculate accuracy for simple tables.
"""
function _accuracy_table_simple(fc_ct::NamedTuple, act_ct::NamedTuple)
    forecast_vec = fc_ct.mean
    value_col = _identify_value_column(act_ct)
    actual_vec = act_ct[value_col]

    return _compute_accuracy_metrics(forecast_vec, actual_vec, nothing)
end

"""
    _accuracy_table_grouped(fc_ct, act_ct, by)

Calculate accuracy for grouped tables.
"""
function _accuracy_table_grouped(fc_ct::NamedTuple, act_ct::NamedTuple, by)
    
    fc_cols = Set(propertynames(fc_ct))
    act_cols = Set(propertynames(act_ct))

    group_cols = Symbol[]
    for col in [:series, :model, :model_name]
        if col in fc_cols && col in act_cols
            push!(group_cols, col)
        end
    end

    if isempty(group_cols)
        return _accuracy_table_simple(fc_ct, act_ct)
    end

    return _calculate_grouped_table_accuracy(fc_ct, act_ct, group_cols)
end

"""
    _calculate_grouped_table_accuracy(fc_ct, act_ct, group_cols)

Calculate accuracy for grouped tables.
"""
function _calculate_grouped_table_accuracy(fc_ct::NamedTuple, act_ct::NamedTuple, group_cols::Vector{Symbol})
    
    fc_groups = _build_group_index(fc_ct, group_cols)
    act_groups = _build_group_index(act_ct, group_cols)

    value_col = _identify_value_column(act_ct)

    result_group_cols = Dict{Symbol, Vector{Any}}()
    for col in group_cols
        result_group_cols[col] = Any[]
    end

    metrics = Dict{Symbol, Vector{Float64}}()
    for m in [:ME, :RMSE, :MAE, :MPE, :MAPE, :ACF1]
        metrics[m] = Float64[]
    end

    for (key, fc_indices) in fc_groups
        if !haskey(act_groups, key)
            continue
        end

        act_indices = act_groups[key]

        forecast_vec = Float64[fc_ct.mean[i] for i in fc_indices]
        actual_vec = Float64[act_ct[value_col][i] for i in act_indices]

        if length(forecast_vec) != length(actual_vec)
            @warn "Length mismatch for group $key. Skipping."
            continue
        end

        acc = _compute_accuracy_metrics(forecast_vec, actual_vec, nothing)

        for col in group_cols
            push!(result_group_cols[col], key[col])
        end

        for (m, v) in pairs(acc)
            if m in keys(metrics)
                push!(metrics[m], v)
            end
        end
    end
    
    result_cols = Symbol[]
    result_vals = Any[]

    for col in group_cols
        push!(result_cols, col)
        push!(result_vals, result_group_cols[col])
    end

    for m in [:ME, :RMSE, :MAE, :MPE, :MAPE, :ACF1]
        push!(result_cols, m)
        push!(result_vals, metrics[m])
    end

    return NamedTuple{Tuple(result_cols)}(Tuple(result_vals))
end

"""
    _build_group_index(ct, group_cols)

Build index mapping group keys to row indices.
"""
function _build_group_index(ct::NamedTuple, group_cols::Vector{Symbol})
    n = length(ct[first(propertynames(ct))])
    groups = Dict{NamedTuple, Vector{Int}}()

    for i in 1:n
        key_vals = [ct[col][i] for col in group_cols]
        key = NamedTuple{Tuple(group_cols)}(Tuple(key_vals))

        if !haskey(groups, key)
            groups[key] = Int[]
        end
        push!(groups[key], i)
    end

    return groups
end
