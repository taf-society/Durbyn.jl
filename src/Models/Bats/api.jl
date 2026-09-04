# ─── BATS public API ──────────────────────────────────────────────────────────
#
# User-facing entry points: bats() model selection (two overloads),
# forecast(), fitted(), and Base.show().

"""
    bats(y::AbstractVector{<:Real}, m::Union{Vector{Int},Nothing}=nothing;
         use_box_cox=nothing,
         use_trend=nothing,
         use_damped_trend=nothing,
         use_arma_errors=true,
         bc_lower=0.0,
         bc_upper=1.0,
         biasadj=false,
         model=nothing)

Fit a BATS model (Box-Cox transformation, ARMA errors, Trend and Seasonal
components) to the univariate series `y`, following De Livera, Hyndman & Snyder (2011).
When `model === nothing` the function automatically searches over Box-Cox,
trend and damping combinations (and optionally ARMA errors) selecting the
best model by AIC; if `model` is supplied the same structure is refit to `y`.

# Arguments
- `y`: real-valued vector representing the time series.
- `m`: optional vector of seasonal periods (ignored when all are 1); the
  `bats(y, m::Int)` method simply wraps this call with `[m]`.
- `use_box_cox`, `use_trend`, `use_damped_trend`: `Bool`, vector of `Bool`,
  or `nothing` to let the algorithm test both `true`/`false`.
- `use_arma_errors`: include ARMA(p, q) error structure via `auto_arima`.
- `bc_lower`, `bc_upper`: bounds for Box-Cox lambda search.
- `biasadj`: request bias-adjusted inverse Box-Cox transformation.
- `model`: result from a previous `bats` call; reuses its specification.
- `kwargs...`: forwarded to `auto_arima` when estimating ARMA errors.

# Returns
A [`BATSModel`](@ref) storing the fitted parameters, state vectors, original
series and a descriptive method label such as `BATS(λ, {p,q}, φ, {m…})`.

# References
- De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011). *Forecasting time
  series with complex seasonal patterns using exponential smoothing*.
  Journal of the American Statistical Association, 106(496), 1513‑1527.

# Examples
```julia
fit = bats(rand(120), 12)
fc = forecast(fit; h = 12)
```
"""
function bats(
    y::AbstractVector{<:Union{Missing,Real}},
    m::Union{Vector{Int},Nothing} = nothing;
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

    if ndims(y) != 1
        throw(ArgumentError("y should be a univariate time series (1D vector)"))
    end

    orig_y = Float64[ismissing(v) ? NaN : Float64(v) for v in y]
    orig_len = length(y)

    if m === nothing
        m = [1]
    end

    y_contig = longest_contiguous(y)
    if length(y_contig) != orig_len
        @warn "Missing values encountered. Using longest contiguous portion of time series"
    end
    y = Float64[v for v in y_contig]

    m = m[m.<length(y)]

    if isempty(m)
        m = [1]
    end

    m = unique(max.(m, 1))

    if all(m .== 1)
        m = nothing
    end

    if model !== nothing
        result = fit_previous_bats_model(collect(Float64, y), model)
        result.y = collect(Float64, orig_y)
        result.method = bats_descriptor(result)
        return result
    end

    if is_constant(y)
        @info "Series is constant. Returning trivial BATS model."
        m_const = create_constant_model(collect(Float64, y))


        m_const.method = bats_descriptor(m_const)
        m_const.y = collect(Float64, orig_y)

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

    if use_box_cox === nothing
        use_box_cox = [false, true]
    end

    box_cox_values = normalize_bool_vector(use_box_cox)

    init_box_cox = nothing
    if any(box_cox_values)
        bc_period = (m === nothing || isempty(m)) ? 1 : first(m)
        init_box_cox = box_cox_lambda(y, bc_period; lower = bc_lower, upper = bc_upper)
    end

    if use_trend === nothing
        use_trend = [false, true]
    elseif use_trend isa Bool && use_trend == false
        use_damped_trend = false
    end
    trend_values = normalize_bool_vector(use_trend)

    if use_damped_trend === nothing
        use_damped_trend = [false, true]
    end
    damping_values = normalize_bool_vector(use_damped_trend)

    y_num = Float64.(y)

    best_aic = Inf
    best_model = nothing

    model_count = 0
    for box_cox in box_cox_values
        for trend in trend_values
            for damping in damping_values
                model_count += 1
                current_model = try
                    filter_specifics(
                        y_num,
                        box_cox = box_cox,
                        trend = trend,
                        damping = damping,
                        seasonal_periods = m,
                        use_arma_errors = use_arma_errors,
                        init_box_cox = init_box_cox,
                        bc_lower = bc_lower,
                        bc_upper = bc_upper,
                        biasadj = biasadj,
                                kwargs...,
                    )
                catch e
                    @warn "    Model failed: $e"
                    nothing
                end

                if current_model === nothing
                    continue
                end

                current_aic = _aic_val(current_model)
                if current_aic < best_aic
                    best_aic = current_aic
                    best_model = current_model
                end
            end
        end
    end

    if best_model === nothing
        error("Unable to fit a model")
    end

    if _REFINE[]
        best_model = _bats_refine_final(y_num, best_model, bc_lower, bc_upper, biasadj)
    end


    if hasproperty(best_model, :optim_return_code) &&
       getfield(best_model, :optim_return_code) != 0
        @warn "optimize() did not converge."
    end

    method_label = bats_descriptor(
        best_model.lambda,
        best_model.ar_coefficients,
        best_model.ma_coefficients,
        best_model.damping_parameter,
        best_model.seasonal_periods,
    )


    result = BATSModel(
        best_model.lambda,
        best_model.alpha,
        best_model.beta,
        best_model.damping_parameter,
        best_model.gamma_values,
        best_model.ar_coefficients,
        best_model.ma_coefficients,
        best_model.seasonal_periods,
        best_model.fitted_values,
        best_model.errors,
        best_model.x,
        best_model.seed_states,
        best_model.variance,
        best_model.aic,
        best_model.likelihood,
        best_model.optim_return_code,
        orig_y,
        Dict{Symbol,Any}(:vect => best_model.parameters.vect, :control => best_model.parameters.control),
        method_label,
        biasadj,
    )

    return result
end

"""
    bats(y::AbstractVector{<:Real}, m::Int; kwargs...)

Convenience wrapper for the primary [`bats`](@ref) method when a single
seasonal period is supplied. Promotes `m` to a one-element vector and
forwards all keyword arguments so Box-Cox, trend/damping selection and
ARMA-error logic match the full interface.
"""
function bats(
    y::AbstractVector{<:Real},
    m::Int;
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
    return bats(
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
    model::BATSModel;
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
            h = (ts_frequency == 1) ? 10 : 2 * ts_frequency
        else
            h = 2 * maximum(seasonal_periods)
        end
    elseif h <= 0
        throw(ArgumentError("Forecast horizon out of bounds"))
    end

    level = _normalize_levels(level; fan=fan)
    n_levels = length(level)

    p = isnothing(model.ar_coefficients) ? 0 : length(model.ar_coefficients)
    q = isnothing(model.ma_coefficients) ? 0 : length(model.ma_coefficients)

    w = make_wmatrix(
        model.damping_parameter,
        model.seasonal_periods,
        model.ar_coefficients,
        model.ma_coefficients,
    )

    g_result = make_gmatrix(
        model.alpha,
        model.beta,
        model.gamma_values,
        model.seasonal_periods,
        p,
        q,
    )

    F = make_fmatrix(
        model.alpha,
        model.beta,
        model.damping_parameter,
        model.seasonal_periods,
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
    method = isempty(stored_method) ? bats_descriptor(model) : stored_method
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

function fitted(model::BATSModel)
    return model.fitted_values
end

function residuals(model::BATSModel)
    return model.errors
end

function Base.show(io::IO, model::BATSModel)
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

    if !isnothing(model.gamma_values)
        println(
            io,
            "  Gamma:   ",
            join([round(g, digits = 4) for g in model.gamma_values], ", "),
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

    !isnothing(model.seasonal_periods) &&
        println(io, "  Seasonal periods: ", model.seasonal_periods)

    println(io, "")
    println(io, "Sigma:   ", round(sqrt(model.variance), digits = 4))
    if !isnothing(model.aic)
        println(io, "AIC:     ", round(model.aic, digits = 2))
    end
end
