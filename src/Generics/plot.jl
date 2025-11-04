import Tables

function plot end


function plot(forecast::Forecast; show_fitted=true, show_residuals=false)
    n_history = length(forecast.x)
    time_history = 1:n_history
    time_forecast = (n_history + 1):(n_history + length(forecast.mean))

    color_historical = "#2C3E50"
    color_forecast = "#3498DB"
    color_fitted = "#E74C3C"
    color_ci_80 = "#3498DB"
    color_ci_95 = "#85C1E9"

    p = Plots.plot(
        framestyle=:box,
        grid=true,
        gridstyle=:dash,
        gridalpha=0.3,
        gridlinewidth=0.5,
        background_color=:white,
        foreground_color=:black,
        legend=:best,
        legendfontsize=8,
        titlefontsize=12,
        guidefontsize=10,
        tickfontsize=9,
        size=(900, 550),
        left_margin=8Plots.mm,
        bottom_margin=6Plots.mm,
        top_margin=5Plots.mm,
        right_margin=8Plots.mm
    )

    Plots.plot!(
        p,
        time_history,
        forecast.x,
        label="Observed",
        linewidth=1.5,
        color=color_historical,
        alpha=0.8
    )

    if show_fitted && !isempty(forecast.fitted) && !all(isnan.(forecast.fitted))
        Plots.plot!(
            p,
            time_history,
            forecast.fitted,
            label="Fitted",
            linewidth=1,
            linestyle=:dash,
            color=color_fitted,
            alpha=0.7
        )
    end

    if !isnothing(forecast.upper) && !isnothing(forecast.lower)
        num_levels = size(forecast.upper, 2)

        function get_ci_color_alpha(i, num_levels)
            if num_levels == 1
                return (color_ci_80, 0.4)
            else
                ratio = (i - 1) / (num_levels - 1)

                r1, g1, b1 = 133, 193, 233  
                r2, g2, b2 = 52, 152, 219 

                r = round(Int, r1 + (r2 - r1) * ratio)
                g = round(Int, g1 + (g2 - g1) * ratio)
                b = round(Int, b1 + (b2 - b1) * ratio)

                color_hex = "#" * string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2)

                alpha_val = 0.25 + 0.3 * ratio

                return (color_hex, alpha_val)
            end
        end

        for i in num_levels:-1:1
            upper_bound = forecast.upper[:, i]
            lower_bound = forecast.lower[:, i]

            fill_color, fill_alpha = get_ci_color_alpha(i, num_levels)

            level_label = !isnothing(forecast.level) ? "$(Int(forecast.level[i]))%" : "CI"

            Plots.plot!(
                p,
                time_forecast,
                upper_bound,
                fillrange=lower_bound,
                fillalpha=fill_alpha,
                fillcolor=fill_color,
                linecolor=fill_color,
                linewidth=0.5,
                linealpha=0.6,
                label=level_label
            )
        end
    end

    Plots.plot!(
        p,
        time_forecast,
        forecast.mean,
        label="Forecast",
        linewidth=2,
        color=color_forecast,
        linestyle=:solid,
        alpha=0.9
    )

    title_text = "Forecasts from " * forecast.method
    Plots.plot!(
        p,
        title=title_text,
        xlabel="Time",
        ylabel="Value"
    )

    if show_residuals && !isempty(forecast.residuals) && !all(isnan.(forecast.residuals))
        p_resid = Plots.plot(
            framestyle=:box,
            grid=true,
            gridstyle=:dash,
            gridalpha=0.3,
            gridlinewidth=0.5,
            background_color=:white,
            foreground_color=:black,
            legend=false,
            titlefontsize=12,
            guidefontsize=10,
            tickfontsize=9,
            size=(900, 250),
            left_margin=8Plots.mm,
            bottom_margin=6Plots.mm,
            top_margin=3Plots.mm,
            right_margin=8Plots.mm
        )

        Plots.hline!(
            p_resid,
            [0],
            color=:gray,
            linestyle=:dash,
            linewidth=1.5,
            alpha=0.6,
            label=""
        )

        Plots.plot!(
            p_resid,
            time_history,
            forecast.residuals,
            linewidth=1.2,
            color=color_historical,
            alpha=0.7,
            seriestype=:line,
            title="Residuals",
            xlabel="Time",
            ylabel="Residual"
        )

        return Plots.plot(p, p_resid, layout=(2, 1), size=(900, 800))
    end

    return p
end

"""
    plot(fc::GroupedForecasts; series=nothing, facet=false, n_cols=2, actual=nothing, kwargs...)

Plot forecasts from grouped/panel data.

# Arguments
- `fc::GroupedForecasts`: Grouped forecasts object
- `series`: Which series to plot. Can be:
  - `nothing`: Plot first series (default)
  - Single value: Plot specific series (e.g., `"A"`, `1`)
  - Vector: Plot multiple series (e.g., `["A", "B"]`)
  - `:all`: Plot all series in facets
- `facet`: If `true`, create faceted plot (multiple panels). Default: `false`
- `n_cols`: Number of columns in facet layout. Default: `2`
- `actual`: Optional table with actual values (must have :series and :value columns)
- `show_fitted`: Show fitted values (default: `true`)

# Returns
Plots.jl plot object

# Examples
```julia
# Plot first series
plot(fc)

# Plot specific series
plot(fc, series = "A")

# Plot multiple series in facets
plot(fc, series = ["A", "B", "C"], facet = true)

# Plot all series
plot(fc, series = :all, facet = true, n_cols = 3)

# Include actual values from test set
plot(fc, series = "A", actual = test_data)
```
"""
function plot(fc; series=nothing, facet=false, n_cols=2, actual=nothing, show_fitted=true, kwargs...)
    if hasproperty(fc, :forecasts) && hasproperty(fc, :groups) && fc.forecasts isa Dict
        return _plot_grouped_forecasts(fc, series, facet, n_cols, actual, show_fitted; kwargs...)
    end

    if hasproperty(fc, :names) && hasproperty(fc, :forecasts) && fc.names isa Vector{String}
        return _plot_forecast_collection(fc, series, facet, n_cols, actual, show_fitted; kwargs...)
    end

    error("Unsupported forecast type for panel plotting: $(typeof(fc))")
end

"""
    _plot_grouped_forecasts(fc, series, facet, n_cols, actual, show_fitted; kwargs...)

Internal function to plot GroupedForecasts.
"""
function _plot_grouped_forecasts(fc, series_select, facet, n_cols, actual, show_fitted; kwargs...)
    available_keys = fc.groups

    if isnothing(series_select)
        
        selected_keys = [_first_successful_series(fc)]
    elseif series_select == :all
        
        selected_keys = [k for k in available_keys if !(fc.forecasts[k] isa Exception)]
    elseif series_select isa AbstractVector
        
        selected_keys = [_find_series_key(fc, s) for s in series_select]
    else
        
        selected_keys = [_find_series_key(fc, series_select)]
    end
    
    selected_keys = filter(k -> !isnothing(k) && !(fc.forecasts[k] isa Exception), selected_keys)

    if isempty(selected_keys)
        error("No valid series found to plot")
    end

    
    actual_dict = _prepare_actual_data(actual)

    if facet || length(selected_keys) > 1
        return _plot_faceted(fc, selected_keys, n_cols, actual_dict, show_fitted)
    else
        return _plot_single_grouped(fc, selected_keys[1], actual_dict, show_fitted)
    end
end

"""
    _plot_forecast_collection(fc, series, facet, n_cols, actual, show_fitted; kwargs...)

Internal function to plot ForecastModelCollection (multiple models).
"""
function _plot_forecast_collection(fc, series_select, facet, n_cols, actual, show_fitted; kwargs...)
    
    first_fc = fc.forecasts[fc.names[1]]

    if first_fc isa Exception
        error("First forecast failed. Cannot determine structure.")
    end
    
    if hasproperty(first_fc, :forecasts) && first_fc.forecasts isa Dict
        return _plot_model_collection_grouped(fc, series_select, facet, n_cols, actual, show_fitted)
    else
        
        return _plot_model_comparison(fc, actual, show_fitted)
    end
end

"""
    _first_successful_series(fc)

Find the first series with successful forecast.
"""
function _first_successful_series(fc)
    for key in fc.groups
        if !(fc.forecasts[key] isa Exception)
            return key
        end
    end
    error("No successful forecasts found")
end

"""
    _find_series_key(fc, series_value)

Find the group key matching a series value.
"""
function _find_series_key(fc, series_value)
    for key in fc.groups
        
        if haskey(key, :series)
            if key.series == series_value
                return key
            end
        elseif length(key) == 1
            
            if first(values(key)) == series_value
                return key
            end
        end
    end

    @warn "Series '$series_value' not found"
    return nothing
end

"""
    _prepare_actual_data(actual)

Prepare actual data dictionary indexed by series.
"""
function _prepare_actual_data(actual)
    if isnothing(actual)
        return nothing
    end

    if !Tables.istable(actual)
        return nothing
    end

    act_ct = Tables.columntable(actual)
    
    if !(:value in propertynames(act_ct))
        @warn "Actual data must have :value column"
        return nothing
    end

    actual_dict = Dict{Any, Vector{Float64}}()

    if :series in propertynames(act_ct)
        
        for i in 1:length(act_ct.value)
            s = act_ct.series[i]
            if !haskey(actual_dict, s)
                actual_dict[s] = Float64[]
            end
            push!(actual_dict[s], Float64(act_ct.value[i]))
        end
    else
        
        actual_dict[nothing] = Float64.(act_ct.value)
    end

    return actual_dict
end

"""
    _plot_single_grouped(fc, key, actual_dict, show_fitted)

Plot a single series from grouped forecasts.
"""
function _plot_single_grouped(fc, key, actual_dict, show_fitted)
    fc_obj = fc.forecasts[key]

    if fc_obj isa Exception
        error("Forecast for series $key failed: $fc_obj")
    end

    p = plot(fc_obj, show_fitted=show_fitted)

    if !isnothing(actual_dict)
        series_id = haskey(key, :series) ? key.series : nothing

        if haskey(actual_dict, series_id)
            actual_vals = actual_dict[series_id]
            n_history = length(fc_obj.x)
            time_actual = (n_history + 1):(n_history + length(actual_vals))

            Plots.plot!(
                p,
                time_actual,
                actual_vals,
                label="Actual",
                linewidth=2,
                color=:green,
                linestyle=:dash,
                alpha=0.8,
                markershape=:circle,
                markersize=3
            )
        end
    end

    series_label = _format_series_label(key)
    Plots.plot!(p, title="Forecasts from $(fc_obj.method) - $series_label")

    return p
end

"""
    _plot_faceted(fc, selected_keys, n_cols, actual_dict, show_fitted)

Create faceted plot for multiple series.
"""
function _plot_faceted(fc, selected_keys, n_cols, actual_dict, show_fitted)
    n_series = length(selected_keys)
    n_rows = ceil(Int, n_series / n_cols)

    plots_array = []

    for key in selected_keys
        fc_obj = fc.forecasts[key]

        if fc_obj isa Exception
            continue
        end

        p = plot(fc_obj, show_fitted=false) 

        if !isnothing(actual_dict)
            series_id = haskey(key, :series) ? key.series : nothing

            if haskey(actual_dict, series_id)
                actual_vals = actual_dict[series_id]
                n_history = length(fc_obj.x)
                h = min(length(actual_vals), length(fc_obj.mean))
                time_actual = (n_history + 1):(n_history + h)

                Plots.plot!(
                    p,
                    time_actual,
                    actual_vals[1:h],
                    label="Actual",
                    linewidth=1.5,
                    color=:green,
                    linestyle=:dash,
                    alpha=0.7
                )
            end
        end

        series_label = _format_series_label(key)
        Plots.plot!(p, title=series_label, titlefontsize=10, legendfontsize=7)

        push!(plots_array, p)
    end

    if isempty(plots_array)
        error("No valid plots created")
    end

    layout = (n_rows, n_cols)
    combined_height = 300 * n_rows
    combined_width = 450 * n_cols

    return Plots.plot(plots_array..., layout=layout, size=(combined_width, combined_height))
end

"""
    _plot_model_comparison(fc, actual, show_fitted)

Plot multiple models for comparison (single series).
"""
function _plot_model_comparison(fc, actual, show_fitted)
    
    first_fc = nothing
    for name in fc.names
        fc_obj = fc.forecasts[name]
        if !(fc_obj isa Exception)
            first_fc = fc_obj
            break
        end
    end

    if isnothing(first_fc)
        error("All models failed")
    end

    n_history = length(first_fc.x)
    time_history = 1:n_history

    p = Plots.plot(
        framestyle=:box,
        grid=true,
        legend=:best,
        size=(900, 550),
        title="Model Comparison",
        xlabel="Time",
        ylabel="Value"
    )

    Plots.plot!(p, time_history, first_fc.x, label="Observed", linewidth=2, color=:black, alpha=0.7)

    colors = [:blue, :red, :green, :purple, :orange, :brown, :pink, :gray]

    for (i, name) in enumerate(fc.names)
        fc_obj = fc.forecasts[name]

        if fc_obj isa Exception
            continue
        end

        time_forecast = (n_history + 1):(n_history + length(fc_obj.mean))
        color = colors[mod1(i, length(colors))]

        Plots.plot!(
            p,
            time_forecast,
            fc_obj.mean,
            label=name,
            linewidth=2,
            color=color,
            alpha=0.8
        )
    end

    if !isnothing(actual)
        if actual isa AbstractVector
            time_actual = (n_history + 1):(n_history + length(actual))
            Plots.plot!(p, time_actual, actual, label="Actual", linewidth=2.5,
                       color=:green, linestyle=:dash, markershape=:circle, markersize=3)
        end
    end

    return p
end

"""
    _plot_model_collection_grouped(fc, series_select, facet, n_cols, actual, show_fitted)

Plot ForecastModelCollection where each forecast is grouped (panel data with multiple models).
"""
function _plot_model_collection_grouped(fc, series_select, facet, n_cols, actual, show_fitted)
    
    first_grouped = nothing
    for name in fc.names
        fc_obj = fc.forecasts[name]
        if !(fc_obj isa Exception)
            first_grouped = fc_obj
            break
        end
    end

    if isnothing(first_grouped)
        error("All models failed")
    end

    available_keys = first_grouped.groups

    if isnothing(series_select)
        selected_keys = [_first_successful_series(first_grouped)]
    elseif series_select == :all
        selected_keys = [k for k in available_keys if !(first_grouped.forecasts[k] isa Exception)]
    elseif series_select isa AbstractVector
        selected_keys = [_find_series_key(first_grouped, s) for s in series_select]
    else
        selected_keys = [_find_series_key(first_grouped, series_select)]
    end

    selected_keys = filter(!isnothing, selected_keys)

    if isempty(selected_keys)
        error("No valid series found to plot")
    end

    actual_dict = _prepare_actual_data(actual)

    if facet || length(selected_keys) > 1
        
        plots_array = []

        for key in selected_keys
            p = _plot_models_for_series(fc, key, actual_dict)
            push!(plots_array, p)
        end

        n_series = length(plots_array)
        n_rows = ceil(Int, n_series / n_cols)
        combined_height = 300 * n_rows
        combined_width = 450 * n_cols

        return Plots.plot(plots_array..., layout=(n_rows, n_cols), size=(combined_width, combined_height))
    else
        return _plot_models_for_series(fc, selected_keys[1], actual_dict)
    end
end

"""
    _plot_models_for_series(fc, key, actual_dict)

Plot all models for a specific series.
"""
function _plot_models_for_series(fc, key, actual_dict)
    
    first_fc = nothing
    for name in fc.names
        grouped = fc.forecasts[name]
        if !(grouped isa Exception) && haskey(grouped.forecasts, key)
            series_fc = grouped.forecasts[key]
            if !(series_fc isa Exception)
                first_fc = series_fc
                break
            end
        end
    end

    if isnothing(first_fc)
        error("No successful forecasts for series $key")
    end

    
    n_history = length(first_fc.x)
    time_history = 1:n_history

    series_label = _format_series_label(key)

    p = Plots.plot(
        framestyle=:box,
        grid=true,
        legend=:best,
        size=(700, 450),
        title="Model Comparison - $series_label",
        xlabel="Time",
        ylabel="Value",
        titlefontsize=10
    )

    
    Plots.plot!(p, time_history, first_fc.x, label="Observed", linewidth=1.5, color=:black, alpha=0.7)

    
    colors = [:blue, :red, :green, :purple, :orange, :brown]
    
    for (i, name) in enumerate(fc.names)
        grouped = fc.forecasts[name]

        if grouped isa Exception || !haskey(grouped.forecasts, key)
            continue
        end

        series_fc = grouped.forecasts[key]

        if series_fc isa Exception
            continue
        end

        time_forecast = (n_history + 1):(n_history + length(series_fc.mean))
        color = colors[mod1(i, length(colors))]

        Plots.plot!(
            p,
            time_forecast,
            series_fc.mean,
            label=name,
            linewidth=2,
            color=color,
            alpha=0.8
        )
    end
    
    if !isnothing(actual_dict)
        series_id = haskey(key, :series) ? key.series : nothing

        if haskey(actual_dict, series_id)
            actual_vals = actual_dict[series_id]
            time_actual = (n_history + 1):(n_history + length(actual_vals))

            Plots.plot!(
                p,
                time_actual,
                actual_vals,
                label="Actual",
                linewidth=2,
                color=:green,
                linestyle=:dash,
                markershape=:circle,
                markersize=2,
                alpha=0.8
            )
        end
    end

    return p
end

"""
    _format_series_label(key::NamedTuple)

Format series key as readable label.
"""
function _format_series_label(key::NamedTuple)
    if isempty(key)
        return "Series"
    end

    parts = String[]
    for (k, v) in pairs(key)
        push!(parts, "$k=$v")
    end
    return join(parts, ", ")
end

"""
    list_series(fc)

List all available series in a grouped forecast.

# Examples
```julia
fc = forecast(fitted, h = 12)
series_list = list_series(fc)
println("Available series: ", series_list)
```
"""
function list_series(fc)
    if hasproperty(fc, :groups)
        # GroupedForecasts
        return [haskey(k, :series) ? k.series : k for k in fc.groups]
    elseif hasproperty(fc, :names) && hasproperty(fc, :forecasts)
        # ForecastModelCollection - get from first successful forecast
        for name in fc.names
            fc_obj = fc.forecasts[name]
            if !(fc_obj isa Exception) && hasproperty(fc_obj, :groups)
                return list_series(fc_obj)
            end
        end
    end
    return []
end