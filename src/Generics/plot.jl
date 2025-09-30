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

    # Create main plot with ggplot2-style theme
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

    # Plot historical data
    Plots.plot!(
        p,
        time_history,
        forecast.x,
        label="Observed",
        linewidth=1.5,
        color=color_historical,
        alpha=0.8
    )

    # Plot fitted values if requested
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

    # Plot confidence intervals (from widest to narrowest for proper layering)
    if forecast.upper !== nothing && forecast.lower !== nothing
        num_levels = size(forecast.upper, 2)

        # Generate colors with gradient from lighter to darker blue
        # For multiple CIs, use a gradient
        function get_ci_color_alpha(i, num_levels)
            if num_levels == 1
                return (color_ci_80, 0.4)
            else
                # Create gradient from light to dark blue
                # Widest (largest i) gets lightest color
                ratio = (i - 1) / (num_levels - 1)

                # Interpolate between light blue and darker blue
                r1, g1, b1 = 133, 193, 233  # Light blue (#85C1E9)
                r2, g2, b2 = 52, 152, 219   # Darker blue (#3498DB)

                r = round(Int, r1 + (r2 - r1) * ratio)
                g = round(Int, g1 + (g2 - g1) * ratio)
                b = round(Int, b1 + (b2 - b1) * ratio)

                color_hex = "#" * string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2)

                # Alpha: wider intervals are more transparent
                alpha_val = 0.25 + 0.3 * ratio

                return (color_hex, alpha_val)
            end
        end

        # Reverse order so narrowest CI is on top
        for i in num_levels:-1:1
            upper_bound = forecast.upper[:, i]
            lower_bound = forecast.lower[:, i]

            fill_color, fill_alpha = get_ci_color_alpha(i, num_levels)

            # Label: show percentage for all levels
            level_label = forecast.level !== nothing ? "$(Int(forecast.level[i]))%" : "CI"

            # Plot upper and lower bounds with visible lines
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

    # Plot forecast mean
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

    # Add title and labels
    title_text = "Forecasts from " * forecast.method
    Plots.plot!(
        p,
        title=title_text,
        xlabel="Time",
        ylabel="Value"
    )

    # If residuals plot is requested, create a layout
    if show_residuals && !isempty(forecast.residuals) && !all(isnan.(forecast.residuals))
        # Create residuals plot
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

        # Add zero reference line first
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

        # Combine plots vertically
        return Plots.plot(p, p_resid, layout=(2, 1), size=(900, 800))
    end

    return p
end