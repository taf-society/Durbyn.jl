"""
    croston_classic(x::Vector; init_strategy::String = "mean", number_of_params::Int = 2, 
                    cost_metric::String = "mar", optimize_init::Bool = true, rm_missing::Bool = false) 
        -> IntermittentDemandCrostonFit

Forecast intermittent demand using the classical Croston method.

# Arguments

- `x::Vector`: Input time series vector representing intermittent demand (including zeros and possibly `missing` values).

- `init_strategy::String`: Initialization strategy for smoothing. Accepts `"mean"` or `"naive"`. Default is `"mean"`.

- `number_of_params::Int`: Number of parameters to optimize (either 1 or 2). Default is `2`.

- `cost_metric::String`: Optimization cost function used to tune parameters. Options include:
    - `"mar"`: Mean absolute ratio
    - `"msr"`: Mean squared ratio
    - `"mae"`: Mean absolute error
    - `"mse"`: Mean squared error
  Default is `"mar"`.

- `optimize_init::Bool`: If `true`, initial values are optimized alongside smoothing parameters. Default is `true`.

- `rm_missing::Bool`: If `true`, `missing` values are removed from `x` before modeling. Default is `false`.

# Returns

- `IntermittentDemandCrostonFit`: A struct containing the fitted model parameters, forecast configuration, 
  and the method used ("Classical Croston Method").

# Description

The classical Croston method decomposes the intermittent demand time series into two separate processes:
demand size and demand interval.

# References

Kourentzes, N. (2014). *On intermittent demand model optimisation and selection*.  
International Journal of Production Economics, 156, 180–190.  
https://doi.org/10.1016/j.ijpe.2014.06.007
"""

function croston_classic(x::Vector, init_strategy::String = "mean", number_of_params::Int = 2, cost_metric::String = "mar", 
    optimize_init::Bool = true, rm_missing::Bool = false)

    return fit_croston(x, "croston", cost_metric, number_of_params, init_strategy, optimize_init, rm_missing)
end


"""
    croston_sba(x::Vector; init_strategy::String = "mean", number_of_params::Int = 2, 
                cost_metric::String = "mar", optimize_init::Bool = true, rm_missing::Bool = false) 
        -> IntermittentDemandCrostonFit

Forecast intermittent demand using the Croston method with Syntetos-Boylan Approximation (SBA).

# Arguments

- `x::Vector`: Input time series vector representing intermittent demand (including zeros and possibly `missing` values).

- `init_strategy::String`: Initialization method for smoothing. Accepts `"mean"` or `"naive"`. Default is `"mean"`.

- `number_of_params::Int`: Number of parameters to optimize (1 or 2). Default is `2`.

- `cost_metric::String`: Cost function used for optimization. Options include:
    - `"mar"`: Mean absolute ratio  
    - `"msr"`: Mean squared ratio  
    - `"mae"`: Mean absolute error  
    - `"mse"`: Mean squared error  
  Default is `"mar"`.

- `optimize_init::Bool`: Whether to optimize initial values alongside smoothing parameters. Default is `true`.

- `rm_missing::Bool`: If `true`, `missing` values will be removed from the input series before modeling. Default is `false`.

# Returns

- `IntermittentDemandCrostonFit`: A struct containing the fitted model configuration, parameters, and 
  method, labeled as `"Croston Method with Syntetos-Boylan Approximation"`.

# Description

This function implements the Syntetos-Boylan Approximation (SBA), a bias-corrected variant of 
the classical Croston method for forecasting intermittent demand. 

# References

Kourentzes, N. (2014). *On intermittent demand model optimisation and selection*.  
International Journal of Production Economics, 156, 180–190.  
https://doi.org/10.1016/j.ijpe.2014.06.007
"""
function croston_sba(x::Vector, init_strategy::String = "mean", number_of_params::Int = 2, cost_metric::String = "mar", 
    optimize_init::Bool = true, rm_missing::Bool = false)
    
    return fit_croston(x, "sba", cost_metric, number_of_params, init_strategy, optimize_init, rm_missing)
end

"""
    croston_sbj(x::Vector; init_strategy::String = "mean", number_of_params::Int = 2, 
                cost_metric::String = "mar", optimize_init::Bool = true, rm_missing::Bool = false) 
        -> IntermittentDemandCrostonFit

Forecast intermittent demand using the Croston method with Shale-Boylan-Johnston (SBJ) bias correction.

# Arguments

- `x::Vector`: Input time series vector representing intermittent demand (including zeros and possibly `missing` values).

- `init_strategy::String`: Initialization method for smoothing. Accepts `"mean"` or `"naive"`. Default is `"mean"`.

- `number_of_params::Int`: Number of parameters to optimize (1 or 2). Default is `2`.

- `cost_metric::String`: Cost function used for parameter optimization. Supported values:
    - `"mar"`: Mean absolute ratio  
    - `"msr"`: Mean squared ratio  
    - `"mae"`: Mean absolute error  
    - `"mse"`: Mean squared error  
  Default is `"mar"`.

- `optimize_init::Bool`: Whether to optimize initial values along with smoothing parameters. Default is `true`.

- `rm_missing::Bool`: If `true`, removes `missing` values from the input time series before modeling. Default is `false`.

# Returns

- `IntermittentDemandCrostonFit`: A struct containing the fitted model, its configuration, and method label, 
  described as `"Croston Method with Shale-Boylan-Johnston Bias Correction"`.

# Description

This method applies the Croston approach with the Shale-Boylan-Johnston (SBJ) bias correction, which introduces 
a more refined adjustment than the SBA method for very intermittent demand. 

# References

Kourentzes, N. (2014). *On intermittent demand model optimisation and selection*.  
International Journal of Production Economics, 156, 180–190.  
https://doi.org/10.1016/j.ijpe.2014.06.007
"""
function croston_sbj(x::Vector, init_strategy::String = "mean", number_of_params::Int = 2, cost_metric::String = "mar", 
    optimize_init::Bool = true, rm_missing::Bool = false)
    
    return fit_croston(x, "sbj", cost_metric, number_of_params, init_strategy, optimize_init, rm_missing)
end


"""
plot(object::IntermittentDemandForecast; show_fitted::Bool = false) -> Plot

Visualize an intermittent demand forecast result, including historical data, forecast mean, and optionally fitted values.

Arguments

object::IntermittentDemandForecast: The forecast result to plot.

show_fitted::Bool: Whether to include in-sample fitted values on the plot. Defaults to false.

Returns

Plot: A Plots.jl object showing the historical demand, forecast horizon, and optionally fitted values.

Description

This function generates a time series plot from an IntermittentDemandForecast object. It displays the historical input series and the forecast mean. If show_fitted is true, the fitted values from the in-sample model are also included as a dashed overlay.

Useful for validating forecast quality and understanding model behavior visually.
"""
function plot(object::IntermittentDemandForecast; show_fitted::Bool = false)

    history = object.model.x
    n_history = length(history)
    mean_fc = object.mean
    time_history = 1:n_history
    time_object = (n_history+1):(n_history+length(mean_fc))

    p = Plots.plot(
        time_history,
        history,
        label = "Historical Data",
        lw = 2,
        title = "Intermittent Demand Forecast using " * object.method * " Method",
        xlabel = "Time",
        ylabel = "Value",
        linestyle = :dash,
    )

    Plots.plot!(time_object, mean_fc, label = "Forecast Mean", lw = 3, color = :blue)

    if show_fitted
        fitted_val = fitted(object.model)
        Plots.plot!(
            time_history,
            fitted_val,
            label = "Fitted Values",
            lw = 3,
            linestyle = :dash,
            color = :blue,
        )
    end
    return p
end