import Tables

"""
    plot(forecast::Forecast; show_fitted=true, show_residuals=false)
    plot(fc; series=nothing, facet=false, n_cols=2, actual=nothing, show_fitted=true, kwargs...)

Plot forecasts. Requires loading a plotting backend.

# Usage
```julia
using Durbyn
using Plots  # Load Plots.jl to enable plotting

fc = forecast(model, h = 12)
plot(fc)
```

See also: [`forecast`](@ref)
"""
function plot end

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
        return [haskey(k, :series) ? k.series : k for k in fc.groups]
    elseif hasproperty(fc, :names) && hasproperty(fc, :forecasts)
        for name in fc.names
            fc_obj = fc.forecasts[name]
            if !(fc_obj isa Exception) && hasproperty(fc_obj, :groups)
                return list_series(fc_obj)
            end
        end
    end
    return []
end
