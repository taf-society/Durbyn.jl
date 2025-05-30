function plot(forecast::Forecast; show_fitted=true, show_residuals=false, title="Forecast Plot")
    n_history = length(forecast.x)
    time_history = 1:n_history
    time_forecast = (n_history + 1):(n_history + length(forecast.mean))
    p = Plots.plot(time_history, forecast.x, label="Historical Data", lw=2, title=title, xlabel="Time", ylabel="Value", linestyle=:dash)
    Plots.plot!(time_forecast, forecast.mean, label="Forecast Mean", lw=3, color=:blue)
    num_levels = size(forecast.upper, 2)
    for i in 1:num_levels
        upper_bound = forecast.upper[:, i]
        lower_bound = forecast.lower[:, i]
        fill_color = i == num_levels ? "#D5DBFF" : "#596DD5"
        Plots.plot!(time_forecast, upper_bound, ribbon=upper_bound - lower_bound, label="", fillcolor=fill_color, linecolor=fill_color)
    end
    if show_fitted && !isempty(forecast.fitted)
        Plots.plot!(time_history, forecast.fitted, label="Fitted Values", linestyle=:dot)
    end
    if show_residuals && !isempty(forecast.residuals)
        p2 = Plots.plot(forecast.residuals, label="Residuals", lw=1, color=:red)
        Plots.plot!(p, p2)
    end
    return p
end