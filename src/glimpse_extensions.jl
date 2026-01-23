import .TableOps: glimpse
import .ModelSpecs: GroupedFittedModels, GroupedForecasts, PanelData, successful_models, failed_groups, errors
import Tables

function glimpse(io::IO, fitted::GroupedFittedModels; maxgroups::Integer = 5)
    println(io, "GroupedFittedModels glimpse")
    println(io, "  Target: ", fitted.spec.formula.target)
    println(io, "  Groups: ", length(fitted.groups))
    println(io, "  Successful: ", fitted.successful, "  Failed: ", fitted.failed)

    if haskey(fitted.metadata, :fit_time)
        fit_time = fitted.metadata[:fit_time]
        println(io, "  Fit time: ", round(fit_time, digits=2), "s")
    end

    sample = min(maxgroups, length(fitted.groups))
    if sample > 0
        println(io, "  Sample models:")
        count = 0
        for key in fitted.groups
            model = fitted.models[key]
            println(io, "    ", key, " => ", model)
            count += 1
            count ≥ sample && break
        end
    end

    failures = errors(fitted)
    if !isempty(failures)
        println(io, "  Sample failures:")
        count = 0
        for (key, err) in failures
            println(io, "    ", key, " => ", err)
            count += 1
            count ≥ sample && break
        end
    end
    return nothing
end

glimpse(fitted::GroupedFittedModels; kwargs...) = glimpse(stdout, fitted; kwargs...)

function glimpse(io::IO, panel::PanelData; maxrows::Integer = 5)
    println(io, "PanelData glimpse")
    println(io, "  Groups: ", isempty(panel.groups) ? "(none)" : join(string.(panel.groups), ", "))
    println(io, "  Date: ", isnothing(panel.date) ? "(none)" : string(panel.date))

    # Show frequency and m
    if panel.frequency !== nothing
        println(io, "  Frequency: ", panel.frequency)
    end
    if panel.m !== nothing
        if panel.m isa Vector
            println(io, "  Seasonal period(s) m: [", join(panel.m, ", "), "]")
        else
            println(io, "  Seasonal period m: ", panel.m)
        end
    else
        println(io, "  Seasonal period m: (not set)")
    end

    # Show target if set
    if panel.target !== nothing
        println(io, "  Target: ", panel.target)
    end

    # Show preprocessing metadata if any
    has_preprocessing = panel.time_fill_meta !== nothing ||
                        panel.target_meta !== nothing ||
                        !isempty(panel.xreg_meta)

    if has_preprocessing
        println(io, "  Preprocessing:")
        if panel.time_fill_meta !== nothing
            meta = panel.time_fill_meta
            println(io, "    Time grid: ", meta.n_added, " rows added (step: ", meta.step, ")")
        end
        if panel.target_meta !== nothing
            meta = panel.target_meta
            pct = round(100 * meta.n_imputed / meta.n_total, digits=1)
            println(io, "    Target (:", panel.target, "): ", meta.n_imputed, " imputed (", pct, "%) via :", meta.strategy)
        end
        for (col, meta) in panel.xreg_meta
            pct = round(100 * meta.n_imputed / meta.n_total, digits=1)
            println(io, "    Exog (:", col, "): ", meta.n_imputed, " imputed (", pct, "%) via :", meta.strategy)
        end
    end

    data = Tables.columntable(panel.data)
    TableOps.glimpse(io, data; maxrows=maxrows)
end

glimpse(panel::PanelData; kwargs...) = glimpse(stdout, panel; kwargs...)

function glimpse(io::IO, fc::GroupedForecasts; maxgroups::Integer = 5)
    println(io, "GroupedForecasts glimpse")
    println(io, "  Groups: ", length(fc.groups))
    println(io, "  Successful: ", fc.successful, "  Failed: ", fc.failed)
    if haskey(fc.metadata, :h)
        println(io, "  Horizon: ", fc.metadata[:h])
    end
    if haskey(fc.metadata, :forecast_time)
        println(io, "  Forecast time: ", round(fc.metadata[:forecast_time], digits=2), "s")
    end

    sample = min(maxgroups, length(fc.groups))
    if sample > 0
        println(io, "  Sample forecasts:")
        count = 0
        for key in fc.groups
            val = fc.forecasts[key]
            println(io, "    ", key, " => ", val)
            count += 1
            count ≥ sample && break
        end
    end

    failures = errors(fc)
    if !isempty(failures)
        println(io, "  Sample failures:")
        count = 0
        for (key, err) in failures
            println(io, "    ", key, " => ", err)
            count += 1
            count ≥ sample && break
        end
    end
    return nothing
end

glimpse(fc::GroupedForecasts; kwargs...) = glimpse(stdout, fc; kwargs...)
