# ─── TBATS public API ─────────────────────────────────────────────────────────
#
# User-facing entry points: tbats() model selection (two overloads),
# forecast(), fitted(), residuals(), and Base.show().

"""
    tbats(y, m; use_box_cox=nothing, use_trend=nothing, use_damped_trend=nothing,
          use_arma_errors=true, bc_lower=0.0, bc_upper=1.0, biasadj=false,
          model=nothing)

Fit a TBATS model (Exponential smoothing state space with Box-Cox
transformation, ARMA errors, trend, and trigonometric seasonality) to a
univariate time series following De Livera, Hyndman & Snyder (2011).
It searches over Box-Cox, trend, and
damped-trend options, optimizes Fourier orders per seasonal period, and can
optionally refit a supplied TBATS/BATS model.

# Arguments
- `y`: Univariate series to model.
- `m`: Seasonal periods; pass `nothing` to infer nonseasonal.
- `use_box_cox`: Bool or vector of Bools; if `nothing`, both FALSE/TRUE are tried and chosen by AIC.
- `use_trend`: Bool or vector; if `nothing`, both are tried and chosen by AIC.
- `use_damped_trend`: Bool or vector; if `nothing`, both are tried and chosen by AIC (ignored when trend is FALSE).
- `use_arma_errors`: Whether to fit ARMA errors (orders selected by `auto_arima` on residuals).
- `bc_lower`/`bc_upper`: Bounds for Box-Cox lambda search.
- `biasadj`: Use bias-adjusted inverse Box-Cox for fitted/forecast means.
- `model`: Previous TBATS/BATS fit to refit without re-estimating parameters.
- `...`: Extra keywords forwarded to `auto_arima` when selecting ARMA(p,q).

# Returns
A `TBATSModel` (or `BATSModel` when no seasonality) storing parameters,
states, fitted values, residuals, variance, likelihood, AIC, and the
descriptor `TBATS(omega, {p,q}, phi, <m1,k1>,...,<mJ,kJ>)`.

# References
- De Livera, A.M., Hyndman, R.J., & Snyder, R.D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. Journal of the American Statistical Association, 106(496), 1513-1527.
"""
function tbats(
    y::AbstractVector{<:Union{Missing,Real}},
    m::Union{Vector{<:Real},Nothing} = nothing;
    use_box_cox::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_damped_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_arma_errors::Bool = true,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
    biasadj::Bool = false,
    model = nothing,
    k::Union{Vector{Int},Int,Nothing} = nothing,
    kwargs...,
)

    if ndims(y) != 1
        throw(ArgumentError("y should be a univariate time series (1D vector)"))
    end

    orig_y = Float64[ismissing(v) ? NaN : Float64(v) for v in y]
    orig_len = length(y)

    seasonal_periods = if m === nothing
        [1]
    else
        copy(m)
    end

    y_contig = longest_contiguous(y)
    if length(y_contig) != orig_len
        @warn "Missing values encountered. Using longest contiguous portion of time series"
    end
    y = Float64[v for v in y_contig]

    seasonal_periods = seasonal_periods[seasonal_periods .< length(y)]
    if isempty(seasonal_periods)
        seasonal_periods = [1]
    end
    seasonal_periods = unique(max.(seasonal_periods, 1))
    if all(seasonal_periods .== 1)
        seasonal_periods = nothing
    end

    if model !== nothing
        if model isa TBATSModel
            result = fit_previous_tbats_model(collect(Float64, y); model = model)
            result.y = collect(Float64, orig_y)
            result.method = tbats_descriptor(result)
            return result
        elseif model isa BATSModel
            return bats(orig_y; model = model)
        else
            throw(ArgumentError("Unsupported model type for refit in tbats"))
        end
    end

    if is_constant(y)
        m_const = create_constant_tbats_model(collect(Float64, y))
        m_const.y = collect(Float64, orig_y)
        m_const.method = tbats_descriptor(m_const)
        return m_const
    end

    if any(yi -> yi <= 0, y)
        if use_box_cox === true || (use_box_cox isa AbstractVector && any(use_box_cox))
            @warn "Series contains non-positive values. Box-Cox transformation disabled."
        end
        use_box_cox = false
    end

    normalize_bool_vector(x) =
        x === nothing ? Bool[false, true] :
        x isa Bool ? Bool[x] :
        x isa AbstractVector{Bool} ? collect(x) :
        throw(ArgumentError("use_* arguments must be Bool, Vector{Bool}, or nothing"))

    box_cox_values = normalize_bool_vector(use_box_cox)
    if any(box_cox_values)
        bc_period = (seasonal_periods === nothing || isempty(seasonal_periods)) ? 1 : round(Int, first(seasonal_periods))
        init_box_cox = box_cox_lambda(y, bc_period; lower = bc_lower, upper = bc_upper)
    else
        init_box_cox = nothing
    end

    trend_values = begin
        if use_trend === nothing
            use_trend = [false, true]
        elseif use_trend isa Bool && use_trend == false
            use_damped_trend = false
        end
        normalize_bool_vector(use_trend)
    end

    damping_values = begin
        if use_damped_trend === nothing
            use_damped_trend = [false, true]
        end
        normalize_bool_vector(use_damped_trend)
    end

    model_params = Bool[any(box_cox_values), any(trend_values), any(damping_values)]

    y_num = Float64.(y)
    n = length(y_num)

    nonseasonal_model = bats(
        y_num;
        use_box_cox = use_box_cox,
        use_trend = use_trend,
        use_damped_trend = use_damped_trend,
        use_arma_errors = use_arma_errors,
        bc_lower = bc_lower,
        bc_upper = bc_upper,
        biasadj = biasadj,
    )

    if seasonal_periods === nothing
        nonseasonal_model.y = orig_y
        return nonseasonal_model
    else
        mask = seasonal_periods .== 1
        seasonal_periods = seasonal_periods[.!mask]
    end

    user_k = k
    if user_k !== nothing
        if user_k isa Int
            user_k = fill(user_k, length(seasonal_periods))
        end
        if length(user_k) != length(seasonal_periods)
            throw(ArgumentError("k must have the same length as seasonal_periods (got $(length(user_k)) vs $(length(seasonal_periods)))"))
        end
        if any(ki -> ki < 1, user_k)
            throw(ArgumentError("k values must be positive integers"))
        end
        for (i, period) in enumerate(seasonal_periods)
            max_k = floor(Int, (period - 1) / 2)
            if user_k[i] > max_k
                throw(ArgumentError("k[$(i)]=$(user_k[i]) exceeds max Fourier order $(max_k) for seasonal period $(period)"))
            end
        end
        k_vector = collect(Int, user_k)
    else
        k_vector = ones(Int, length(seasonal_periods))
    end

    function safe_fit_specific_tbats(
        y_num;
        use_box_cox::Bool,
        use_beta::Bool,
        use_damping::Bool,
        seasonal_periods::Vector{<:Real},
        k_vector::Vector{Int},
        init_box_cox,
        bc_lower,
        bc_upper,
        biasadj,
    )
        try
            return fit_specific_tbats(
                y_num;
                use_box_cox = use_box_cox,
                use_beta = use_beta,
                use_damping = use_damping,
                seasonal_periods = seasonal_periods,
                k_vector = k_vector,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
            )
        catch e
            @warn "fit_specific_tbats failed: $e"
            return nothing
        end
    end

    best_model = safe_fit_specific_tbats(
        y_num;
        use_box_cox = model_params[1],
        use_beta = model_params[2],
        use_damping = model_params[3],
        seasonal_periods = seasonal_periods,
        k_vector = k_vector,
        init_box_cox = init_box_cox,
        bc_lower = bc_lower,
        bc_upper = bc_upper,
        biasadj = biasadj,
    )

    best_aic = _aic_val(best_model)

    if user_k !== nothing
        # User provided k — skip k-search, go straight to model selection
        @goto k_search_done
    end

    for (i, period) in enumerate(seasonal_periods)
        if period == 2
            continue
        end

        max_k = floor(Int, (period - 1) / 2)

        if i != 1
            current_k = 2
            while current_k <= max_k
                if period % current_k != 0
                    current_k += 1
                    continue
                end
                latter = period / current_k
                if any(seasonal_periods[1:i-1] .% latter .== 0)
                    max_k = current_k - 1
                    break
                else
                    current_k += 1
                end
            end
        end

        if max_k == 1
            continue
        end


        if max_k <= 6
            k_vector[i] = max_k
            local_best_model = best_model
            local_best_aic = Inf

            while true
                new_model = safe_fit_specific_tbats(
                    y_num;
                    use_box_cox = model_params[1],
                    use_beta = model_params[2],
                    use_damping = model_params[3],
                    seasonal_periods = seasonal_periods,
                    k_vector = k_vector,
                    init_box_cox = init_box_cox,
                    bc_lower = bc_lower,
                    bc_upper = bc_upper,
                    biasadj = biasadj,
                )

                new_aic = _aic_val(new_model)

                if new_aic > local_best_aic
                    k_vector[i] += 1
                    break
                else
                    if k_vector[i] == 1
                        local_best_model =
                            new_model === nothing ? local_best_model : new_model
                        local_best_aic = min(local_best_aic, new_aic)
                        break
                    end
                    k_vector[i] -= 1
                    local_best_model = new_model === nothing ? local_best_model : new_model
                    local_best_aic = min(local_best_aic, new_aic)
                end
            end

            if local_best_aic < best_aic
                best_aic = local_best_aic
                best_model = local_best_model
            end

        else
            step_up_k = copy(k_vector)
            step_down_k = copy(k_vector)
            step_up_k[i] = 7
            step_down_k[i] = 5
            k_vector[i] = 6

            up_model = safe_fit_specific_tbats(
                y_num;
                use_box_cox = model_params[1],
                use_beta = model_params[2],
                use_damping = model_params[3],
                seasonal_periods = seasonal_periods,
                k_vector = step_up_k,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
            )
            level_model = safe_fit_specific_tbats(
                y_num;
                use_box_cox = model_params[1],
                use_beta = model_params[2],
                use_damping = model_params[3],
                seasonal_periods = seasonal_periods,
                k_vector = k_vector,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
            )
            down_model = safe_fit_specific_tbats(
                y_num;
                use_box_cox = model_params[1],
                use_beta = model_params[2],
                use_damping = model_params[3],
                seasonal_periods = seasonal_periods,
                k_vector = step_down_k,
                init_box_cox = init_box_cox,
                bc_lower = bc_lower,
                bc_upper = bc_upper,
                biasadj = biasadj,
            )

            a_up = _aic_val(up_model)
            a_level = _aic_val(level_model)
            a_down = _aic_val(down_model)

            if a_down <= a_up && a_down <= a_level
                best_local = down_model
                best_local_aic = a_down
                k_vector[i] = 5

                while true
                    k_vector[i] -= 1
                    new_model = safe_fit_specific_tbats(
                        y_num;
                        use_box_cox = model_params[1],
                        use_beta = model_params[2],
                        use_damping = model_params[3],
                        seasonal_periods = seasonal_periods,
                        k_vector = k_vector,
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                    )
                    new_aic = _aic_val(new_model)

                    if new_aic > best_local_aic
                        k_vector[i] += 1
                        break
                    else
                        best_local = new_model === nothing ? best_local : new_model
                        best_local_aic = min(best_local_aic, new_aic)
                    end
                    if k_vector[i] == 1
                        break
                    end
                end

                if best_local_aic < best_aic
                    best_aic = best_local_aic
                    best_model = best_local
                end

            elseif a_level <= a_up && a_level <= a_down
                if a_level < best_aic
                    best_aic = a_level
                    best_model = level_model
                end

            else
                best_local = up_model
                best_local_aic = a_up
                k_vector[i] = 7

                while true
                    k_vector[i] += 1
                    new_model = safe_fit_specific_tbats(
                        y_num;
                        use_box_cox = model_params[1],
                        use_beta = model_params[2],
                        use_damping = model_params[3],
                        seasonal_periods = seasonal_periods,
                        k_vector = k_vector,
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                    )
                    new_aic = _aic_val(new_model)

                    if new_aic > best_local_aic
                        k_vector[i] -= 1
                        break
                    else
                        best_local = new_model === nothing ? best_local : new_model
                        best_local_aic = min(best_local_aic, new_aic)
                    end
                    if k_vector[i] == max_k
                        break
                    end
                end

                if best_local_aic < best_aic
                    best_aic = best_local_aic
                    best_model = best_local
                end
            end
        end
    end

    @label k_search_done

    aux_model = best_model

    if _aic_val(nonseasonal_model) < _aic_val(best_model)
        best_model = nonseasonal_model
    end

    for box_cox in box_cox_values
        for trend in trend_values
            for damping in damping_values
                if !trend && damping
                    continue
                end

                if model_params == Bool[box_cox, trend, damping]
                    new_model = filter_tbats_specifics(
                        y_num,
                        box_cox,
                        trend,
                        damping,
                        seasonal_periods,
                        k_vector,
                        use_arma_errors;
                        aux_model = aux_model,
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                                kwargs...,
                    )
                elseif trend || !damping
                    new_model = filter_tbats_specifics(
                        y_num,
                        box_cox,
                        trend,
                        damping,
                        seasonal_periods,
                        k_vector,
                        use_arma_errors;
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                                kwargs...,
                    )
                else
                    continue
                end

                if new_model === nothing
                    continue
                end

                if best_model === nothing || _aic_val(new_model) < _aic_val(best_model)
                    best_model = new_model
                end
            end
        end
    end


    if hasproperty(best_model, :optim_return_code) &&
       getfield(best_model, :optim_return_code) != 0
        @warn "optimize() did not converge."
    end

    if best_model === nothing
        error("Failed to fit any TBATS/BATS model")
    end

    if best_model isa BATSModel
        return best_model
    end


    method_label = tbats_descriptor(
        best_model.lambda,
        best_model.ar_coefficients,
        best_model.ma_coefficients,
        best_model.damping_parameter,
        best_model.seasonal_periods,
        best_model.k_vector,
    )

    result = TBATSModel(
        best_model.lambda,
        best_model.alpha,
        best_model.beta,
        best_model.damping_parameter,
        best_model.gamma_one_values,
        best_model.gamma_two_values,
        best_model.ar_coefficients,
        best_model.ma_coefficients,
        best_model.seasonal_periods,
        best_model.k_vector,
        best_model.fitted_values,
        best_model.errors,
        best_model.x,
        best_model.seed_states,
        best_model.variance,
        best_model.aic,
        best_model.likelihood,
        best_model.optim_return_code,
        orig_y,
        Dict{Symbol,Any}(
            :vect => best_model.parameters.vect,
            :control => best_model.parameters.control,
        ),
        method_label,
        biasadj,
    )

    return result
end



"""
    tbats(y::AbstractVector, m::Real; kwargs...)

Convenience wrapper for `tbats` when a single seasonal period is supplied as
a scalar. It forwards all keyword arguments to the primary `tbats` method.
"""
function tbats(
    y::AbstractVector{<:Real},
    m::Real;
    use_box_cox::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_damped_trend::Union{Bool,AbstractVector{Bool},Nothing} = nothing,
    use_arma_errors::Bool = true,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
    biasadj::Bool = false,
    model = nothing,
    kwargs...,
)
    return tbats(
        y,
        [m];
        use_box_cox = use_box_cox,
        use_trend = use_trend,
        use_damped_trend = use_damped_trend,
        use_arma_errors = use_arma_errors,
        bc_lower = bc_lower,
        bc_upper = bc_upper,
        biasadj = biasadj,
        model = model,
        kwargs...,
    )
end

function forecast(
    model::TBATSModel;
    h::Union{Int,Nothing} = nothing,
    level::AbstractVector{<:Real} = [80, 95],
    fan::Bool = false,
    biasadj::Union{Bool,Nothing} = nothing,
)
    seasonal_periods = model.seasonal_periods
    ts_frequency =
        isnothing(seasonal_periods) || isempty(seasonal_periods) ? 1 :
        maximum(seasonal_periods)

    if h === nothing
        if isnothing(seasonal_periods) || isempty(seasonal_periods)
            h = (ts_frequency == 1) ? 10 : round(Int, 2 * ts_frequency)
        else
            h = round(Int, 2 * maximum(seasonal_periods))
        end
    elseif h <= 0
        throw(ArgumentError("Forecast horizon out of bounds"))
    end

    if fan
        level = collect(51.0:3.0:99.0)
    else
        if minimum(level) > 0 && maximum(level) < 1
            level = 100.0 .* level
        elseif minimum(level) < 0 || maximum(level) > 99.99
            throw(ArgumentError("Confidence limit out of range"))
        end
    end
    n_levels = length(level)

    p = isnothing(model.ar_coefficients) ? 0 : length(model.ar_coefficients)
    q = isnothing(model.ma_coefficients) ? 0 : length(model.ma_coefficients)
    tau = isnothing(model.k_vector) ? 0 : 2 * sum(model.k_vector)

    w = make_tbats_wmatrix(
        model.damping_parameter,
        model.k_vector,
        model.ar_coefficients,
        model.ma_coefficients,
        tau,
    )

    g_result = make_tbats_gmatrix(
        model.alpha,
        model.beta,
        model.gamma_one_values,
        model.gamma_two_values,
        model.k_vector,
        p,
        q,
    )

    F = make_tbats_fmatrix(
        model.alpha,
        model.beta,
        model.damping_parameter,
        model.seasonal_periods,
        model.k_vector,
        g_result.gamma_bold_matrix,
        model.ar_coefficients,
        model.ma_coefficients,
    )

    n_state = size(model.x, 1)
    x_states = zeros(Float64, n_state, h)
    y_forecast = zeros(Float64, h)

    x_last = model.x[:, end]

    y_forecast[1] = (w.w_transpose*x_last)[1]
    x_states[:, 1] = F * x_last

    if h > 1
        for t = 2:h
            x_states[:, t] = F * x_states[:, t-1]
            y_forecast[t] = (w.w_transpose*x_states[:, t-1])[1]
        end
    end

    variance_multiplier = ones(Float64, h)
    if h > 1
        f_running = Matrix{Float64}(I, n_state, n_state)
        for j = 1:(h-1)
            if j > 1
                f_running = f_running * F
            end

            c_j_vec = w.w_transpose * f_running * g_result.g
            c_j = c_j_vec[1]
            variance_multiplier[j+1] = variance_multiplier[j] + c_j^2
        end
    end

    variance = model.variance .* variance_multiplier
    stdev = sqrt.(variance)

    lower_bounds = Array{Float64}(undef, h, n_levels)
    upper_bounds = Array{Float64}(undef, h, n_levels)

    for (idx, lev) in enumerate(level)
        z = abs(quantile(Normal(), (100.0 - lev) / 200.0))
        marg_error = stdev .* z
        lower_bounds[:, idx] = y_forecast .- marg_error
        upper_bounds[:, idx] = y_forecast .+ marg_error
    end

    y_fc_out = copy(y_forecast)
    lb_out = copy(lower_bounds)
    ub_out = copy(upper_bounds)

    if !isnothing(model.lambda)
        λ = model.lambda
        ba = biasadj === nothing ? model.biasadj : biasadj
        y_fc_out = inv_box_cox(y_forecast; lambda=λ, biasadj=ba, fvar=variance)

        lb_out = inv_box_cox(lower_bounds; lambda=λ)
        ub_out = inv_box_cox(upper_bounds; lambda=λ)

        if λ < 1
            lb_out = max.(lb_out, 0.0)
        end
    end

    stored_method = hasproperty(model, :method) ? String(model.method) : ""
    method = isempty(stored_method) ? tbats_descriptor(model) : stored_method
    x_series = getproperty(model, :y)
    fitted = getproperty(model, :fitted_values)
    residuals = getproperty(model, :errors)

    forecast_obj = Forecast(
        model,
        method,
        y_fc_out,
        level,
        x_series,
        ub_out,
        lb_out,
        fitted,
        residuals,
    )

    return forecast_obj
end

function fitted(model::TBATSModel)
    return model.fitted_values
end

function residuals(model::TBATSModel)
    return model.errors
end

function Base.show(io::IO, model::TBATSModel)
    println(io, model.method)
    println(io, "")
    println(io, "Parameters:")

    !isnothing(model.lambda) && println(io, "  Lambda:  ", round(model.lambda, digits = 4))
    println(io, "  Alpha:   ", round(model.alpha, digits = 4))

    if !isnothing(model.beta)
        println(io, "  Beta:    ", round(model.beta, digits = 4))
        damping = isnothing(model.damping_parameter) ? 1.0 : model.damping_parameter
        println(io, "  Damping: ", round(damping, digits = 4))
    end

    if !isnothing(model.gamma_one_values)
        println(
            io,
            "  Gamma-1: ",
            join([round(g, digits = 4) for g in model.gamma_one_values], ", "),
        )
    end

    if !isnothing(model.gamma_two_values)
        println(
            io,
            "  Gamma-2: ",
            join([round(g, digits = 4) for g in model.gamma_two_values], ", "),
        )
    end

    if !isnothing(model.ar_coefficients)
        println(
            io,
            "  AR:      ",
            join([round(φ, digits = 4) for φ in model.ar_coefficients], ", "),
        )
    end

    if !isnothing(model.ma_coefficients)
        println(
            io,
            "  MA:      ",
            join([round(θ, digits = 4) for θ in model.ma_coefficients], ", "),
        )
    end

    if !isnothing(model.seasonal_periods) && !isnothing(model.k_vector)
        println(io, "  Seasonal periods: ", model.seasonal_periods)
        println(io, "  Fourier terms (k): ", model.k_vector)
    end

    println(io, "")
    println(io, "Sigma:   ", round(sqrt(model.variance), digits = 4))
    if !isnothing(model.aic)
        println(io, "AIC:     ", round(model.aic, digits = 2))
    end
end
