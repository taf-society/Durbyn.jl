# ─── ETS orchestrator ─────────────────────────────────────────────────────────

"""
    ets(
        y::AbstractArray,
        m::Int,
        model::Union{String,ETS};
        damped::Union{Bool,Nothing} = nothing,
        alpha::Union{Float64,Bool,Nothing} = nothing,
        beta::Union{Float64,Bool,Nothing} = nothing,
        gamma::Union{Float64,Bool,Nothing} = nothing,
        phi::Union{Float64,Bool,Nothing} = nothing,
        additive_only::Bool = false,
        lambda::Union{Float64,Bool,Nothing,Symbol} = nothing,
        biasadj::Bool = false,
        lower::AbstractArray = [0.0001, 0.0001, 0.0001, 0.8],
        upper::AbstractArray = [0.9999, 0.9999, 0.9999, 0.98],
        opt_crit::Symbol = :lik,
        nmse::Int = 3,
        bounds::Symbol = :both,
        ic::Symbol = :aicc,
        restrict::Bool = true,
        allow_multiplicative_trend::Bool = false,
        use_initial_values::Bool = false,
        missing_method::MissingMethod = Contiguous(),
        options::NelderMeadOptions = NelderMeadOptions()
    ) -> EtsModel

Exponential smoothing state space model (ETS).

Fits an ETS model to the series `y` using the state space framework of
Hyndman et al. (2002, 2008). If `model` is a three-letter specification,
the corresponding model is fitted; otherwise the model structure and
damping are selected automatically (subject to constraints) using the
information criterion given by `ic`.

# Positional arguments
- `y::AbstractArray`: Univariate numeric series (vector or `AbstractArray`) to model.
- `m::Int`: Seasonal period (e.g. `12` for monthly data with yearly seasonality).
- `model::Union{String,ETS}`: Either a three-character code `"E T S"` where
  - `E ∈ {"A","M","Z"}` is the error type (Additive, Multiplicative, or auto),
  - `T ∈ {"N","A","M","Z"}` is the trend type (None, Additive, Multiplicative, or auto),
  - `S ∈ {"N","A","M","Z"}` is the seasonal type (None, Additive, Multiplicative, or auto);
  for example `"ANN"` = simple exponential smoothing with additive errors,
  `"MAM"` = multiplicative Holt-Winters with multiplicative errors.
  Alternatively, pass a previously fitted `ETS` object to refit the same structure
  to new data (see `use_initial_values`).

# Keyword arguments
- `damped::Union{Bool,Nothing}=nothing`: If `true`, use a damped trend; if `false`, do not;
  if `nothing`, both variants are considered during model selection.
- `alpha::Union{Float64,Bool,Nothing}=nothing`: Smoothing level. If `nothing`, estimated.
  If `false`, treated as "not set" during model selection (estimate if needed).
- `beta::Union{Float64,Bool,Nothing}=nothing`: Trend smoothing parameter. Same rules as `alpha`.
- `gamma::Union{Float64,Bool,Nothing}=nothing`: Seasonal smoothing parameter. Same rules as `alpha`.
- `phi::Union{Float64,Bool,Nothing}=nothing`: Damping parameter (for damped trend). If `nothing`, estimated.
- `additive_only::Bool=false`: If `true`, restrict search to additive-error/season/trend models.
- `lambda::Union{Float64,Bool,Nothing,Symbol}=nothing`: Box-Cox transform parameter. Use a numeric
  value to apply a fixed transform; pass `:auto` to select via `BoxCox.lambda`-style search;
  `nothing` = no transform. When set (including `:auto`), only additive models are considered.
- `biasadj::Bool=false`: Use bias-adjusted back-transformation to return *mean* (not median) forecasts
  and fits when Box-Cox is used.
- `lower::AbstractArray=[1e-4, 1e-4, 1e-4, 0.8]`: Lower bounds for `(alpha, beta, gamma, phi)`.
  Ignored if `bounds === :admissible`.
- `upper::AbstractArray=[0.9999, 0.9999, 0.9999, 0.98]`: Upper bounds for `(alpha, beta, gamma, phi)`.
  Ignored if `bounds === :admissible`.
- `opt_crit::Symbol=:lik`: Optimization criterion. One of
  `:lik` (log-likelihood), `:amse` (average MSE over first `nmse` horizons),
  `:mse`, `:sigma` (stdev of residuals), or `:mae`.
- `nmse::Int=3`: Horizons for `:amse` (1 ≤ `nmse` ≤ 30).
- `bounds::Symbol=:both`: Parameter space restriction:
  `:usual` enforces `[lower, upper]`, `:admissible` enforces ETS admissibility,
  `:both` uses their intersection.
- `ic::Symbol=:aicc`: Information criterion used for model selection; one of `:aicc`, `:aic`, `:bic`.
- `restrict::Bool=true`: Disallow models with infinite variance.
- `allow_multiplicative_trend::Bool=false`: If `true`, multiplicative trends may be considered when
  searching. Ignored if a multiplicative trend is explicitly requested in `model`.
- `use_initial_values::Bool=false`: If `model isa ETS` and `true`, reuse both its structure and
  initial states (no re-estimation of initials). If `false`, initials are re-estimated.
- `missing_method::MissingMethod=Contiguous()`: Strategy for handling missing values in `y`.
  Use `Contiguous()` (largest contiguous block), `Interpolate()` (interpolate), or `FailMissing()` (error).
- `options::NelderMeadOptions=NelderMeadOptions()`: Optimizer configuration for parameter estimation.

# Details
The ETS family encompasses exponential smoothing methods within a state space
formulation. The only required input is the series `y` (and `m` if seasonal).
If `model` contains `"Z"` components or `damped == nothing`, the procedure searches
over the admissible model space (respecting `additive_only`, `restrict`,
and `allow_multiplicative_trend`) and selects the specification minimizing `ic`.
Parameter estimates comply with `bounds` and are obtained by numerical optimization
using the criterion `opt_crit`. Box-Cox transformation (via `lambda`) occurs prior
to estimation; with transformation, bias adjustment via `biasadj` returns mean-scale
fits/forecasts.

# Returns
- `EtsModel`: A fitted EtsModel model object containing parameter estimates, initial states,
  model specification, fitted values, residuals, and information-criterion values.

Convenience accessors such as `fitted(::ETS)` and `residuals(::ETS)` return the
fitted values and residuals, respectively.

# References
- Hyndman, R.J., Koehler, A.B., Snyder, R.D., & Grose, S. (2002). *A state space framework for automatic forecasting using exponential smoothing methods*. International Journal of Forecasting, 18(3), 439-454.
- Hyndman, R.J., Akram, Md., & Archibald, B. (2008). *The admissible parameter space for exponential smoothing models*. Annals of the Institute of Statistical Mathematics, 60(2), 407-426.
- Hyndman, R.J., Koehler, A.B., Ord, J.K., & Snyder, R.D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.


# Examples
```julia
# Fit automatically selected ETS model to a monthly series (m = 12)
using Durbyn
using Durbyn.ExponentialSmoothing

ap = air_passengers()
fit = ets(ap(), 12, "ZZZ")

# Specify a particular structure (multiplicative seasonality, additive trend, additive errors)
fit2 = ets(ap, 12, "AAM")
fc2 = forecast(fit2, h=12)
plot(fc2)

# Use a damped trend search and automatic Box-Cox selection
fit3 = ets(ap, 12, "ZZZ"; damped=nothing, lambda=:auto, biasadj=true)
fc3 = forecast(fit3, h=12)
plot(fc3)
```
"""
function ets(
    y::AbstractArray,
    m::Int,
    model::Union{String,ETS};
    damped::Union{Bool,Nothing} = nothing,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    gamma::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    additive_only::Bool = false,
    lambda::Union{Float64,Bool,Nothing,Symbol} = nothing,
    biasadj::Bool = false,
    lower::AbstractArray = [0.0001, 0.0001, 0.0001, 0.8],
    upper::AbstractArray = [0.9999, 0.9999, 0.9999, 0.98],
    opt_crit::Symbol = :lik,
    nmse::Int = 3,
    bounds::Symbol = :both,
    ic::Symbol = :aicc,
    restrict::Bool = true,
    allow_multiplicative_trend::Bool = false,
    use_initial_values::Bool = false,
    missing_method::MissingMethod = Contiguous(),
    options::NelderMeadOptions = NelderMeadOptions(maxit=2000)
)

    if model == "ZZZ" && is_constant(y)
        return ses(y, alpha = 0.99999, initial = :simple)
    end

    out = ets_base_model(
        y,
        m,
        model,
        damped = damped,
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        phi = phi,
        additive_only = additive_only,
        lambda = lambda,
        biasadj = biasadj,
        lower = lower,
        upper = upper,
        opt_crit = opt_crit,
        nmse = nmse,
        bounds = bounds,
        ic = ic,
        restrict = restrict,
        allow_multiplicative_trend = allow_multiplicative_trend,
        use_initial_values = use_initial_values,
        missing_method = missing_method,
        options=options
    )

    return out
end

# ─── Preprocessing ────────────────────────────────────────────────────────────

function process_parameters(
    y,
    m,
    model,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    additive_only,
    lambda,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    ic,
    missing_method::MissingMethod,
)

    opt_crit = _check_arg(opt_crit, (:lik, :amse, :mse, :sigma, :mae), "opt_crit")
    bounds = _check_arg(bounds, (:both, :usual, :admissible), "bounds")
    ic = _check_arg(ic, (:aicc, :aic, :bic), "ic")

    ny = length(y)
    y = handle_missing(y, missing_method; m=m)

    if ny != length(y) && missing_method isa Contiguous
        @warn "Missing values encountered. Using longest contiguous portion of time series"
        ny = length(y)
    end

    orig_y = y

    if typeof(model) == ETS && isnothing(lambda)
        lambda = model.lambda
    end

    if !isnothing(lambda)
        y, lambda = box_cox(y, m, lambda = lambda)
        additive_only = true
    end

    if nmse < 1 || nmse > 30
        throw(ArgumentError("nmse out of range"))
    end

    if any(x -> x < 0, upper .- lower)
        throw(ArgumentError("Lower limits must be less than upper limits"))
    end

    return orig_y,
    y,
    lambda,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    additive_only,
    opt_crit,
    bounds,
    ic,
    missing_method,
    ny
end

# ─── Refit ────────────────────────────────────────────────────────────────────

function ets_refit(
    y::AbstractArray,
    m::Int,
    model::ETS;
    biasadj::Bool = false,
    use_initial_values::Bool = false,
    nmse::Int = 3,
    kwargs...,
)
    alpha = max(model.par["alpha"], 1e-10)
    beta = get(model.par, "beta", nothing)
    gamma = get(model.par, "gamma", nothing)
    phi = get(model.par, "phi", nothing)

    modelcomponents = string(model.components[1], model.components[2], model.components[3])
    damped = parse(Bool, model.components[4])

    if use_initial_values
        errortype = string(modelcomponents[1])
        trendtype = string(modelcomponents[2])
        seasontype = string(modelcomponents[3])

        # Handle both 1D and 2D initstate arrays
        if ndims(model.initstate) == 1
            initstates = Vector(model.initstate)
        else
            initstates = Vector(model.initstate[1, :])
        end

        lik, amse, e, states = calculate_residuals(
            y,
            m,
            initstates,
            errortype,
            trendtype,
            seasontype,
            damped,
            alpha,
            beta,
            gamma,
            phi,
            nmse,
        )

        np = length(model.par) + 1
        loglik = -0.5 * lik
        aic = lik + 2 * np
        bic = lik + log(length(y)) * np
        aicc = model.aic + 2 * np * (np + 1) / (length(y) - np - 1)
        mse = amse[1]
        amse = mean(amse)

        if errortype == "A"
            fitted = y .- e
        else
            fitted = y ./ (1 .+ e)
        end

        sigma2 = sum(e .^ 2) / (length(y) - np)
        x = y

        if biasadj
            model.fitted = inv_box_cox(model.fitted, lambda=model.lambda, biasadj=biasadj, fvar=var(model.residuals))
        end
        model = EtsModel(
            fitted,
            e,
            model.components,
            x,
            model.par,
            loglik,
            model.initstate,
            states,
            String[],
            model.sse,
            sigma2,
            m,
            model.lambda,
            biasadj,
            aic,
            bic,
            aicc,
            mse,
            amse,
            model.fit,
            model.method,
        )

        return model
    else
        model = modelcomponents
        @info "Model is being refit with current smoothing parameters but initial states are being re-estimated.\nSet 'use_initial_values=true' if you want to re-use existing initial values."
        return EtsRefit(modelcomponents, alpha, beta, gamma, phi)
    end
end

# ─── Main orchestrator ────────────────────────────────────────────────────────

function ets_base_model(
    y::AbstractArray,
    m::Int,
    model;
    damped::Union{Bool,Nothing} = nothing,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    gamma::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    additive_only::Bool = false,
    lambda::Union{Float64,Bool,Nothing,Symbol} = nothing,
    biasadj::Bool = false,
    lower::AbstractArray = [0.0001, 0.0001, 0.0001, 0.8],
    upper::AbstractArray = [0.9999, 0.9999, 0.9999, 0.98],
    opt_crit::Symbol = :lik,
    nmse::Int = 3,
    bounds::Symbol = :both,
    ic::Symbol = :aicc,
    restrict::Bool = true,
    allow_multiplicative_trend::Bool = false,
    use_initial_values::Bool = false,
    missing_method::MissingMethod = Contiguous(),
    options::NelderMeadOptions,
)

    orig_y,
    y,
    lambda,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    additive_only,
    opt_crit,
    bounds,
    ic,
    missing_method,
    ny = process_parameters(
        y,
        m,
        model,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        additive_only,
        lambda,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        ic,
        missing_method,
    )

    if model isa ETS
        refit_result = ets_refit(
            y,
            m,
            model,
            biasadj = biasadj,
            use_initial_values = use_initial_values,
            nmse = nmse,
        )
        if refit_result isa ETS
            return refit_result
        end
        # EtsRefit was returned, extract model string for further processing
        model = refit_result.model
    end

    errortype, trendtype, seasontype, npars, data_positive =
        validate_and_set_model_params(model, y, m, damped, restrict, additive_only)

    if ny <= npars + 4

        if !isnothing(damped) && damped
            @warn "Not enough data to use damping"
        end

        return fit_small_dataset(
            orig_y,
            m,
            alpha,
            beta,
            gamma,
            phi,
            trendtype,
            seasontype,
            lambda,
            biasadj,
            options
        )
    end

    model = fit_best_ets_model(
        Float64.(y),
        m,
        errortype,
        trendtype,
        seasontype,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        ic,
        data_positive,
        restrict = restrict,
        additive_only = additive_only,
        allow_multiplicative_trend = allow_multiplicative_trend,
        options = options)

    method = model["method"]
    components = model["components"]
    model = model["model"]
    np = length(model["par"])
    sigma2 = sum(skipmissing(model["residuals"] .^ 2)) / (ny - np)
    sse = NaN

    if !isnothing(lambda)
        model["fitted"] =
            inv_box_cox(model["fitted"], lambda = lambda, biasadj = biasadj, fvar = sigma2)
    end

    initstates = model["states"][1, :]

    model = EtsModel(
        model["fitted"],
        model["residuals"],
        components,
        orig_y,
        model["par"],
        model["loglik"],
        initstates,
        model["states"],
        ["cff"],
        sse,
        sigma2,
        m,
        lambda,
        biasadj,
        model["aic"],
        model["bic"],
        model["aicc"],
        model["mse"],
        model["amse"],
        model["fit"],
        method,
    )
    return model
end

# ─── Simulation ───────────────────────────────────────────────────────────────

function simulate_ets(
    object::ETS,
    nsim::Union{Int,Nothing} = nothing;
    future::Bool = true,
    bootstrap::Bool = false,
    innov::Union{Vector{Float64},Nothing} = nothing,
)

    x = object.x
    m = object.m
    states = object.states
    residuals = object.residuals
    sigma2 = object.sigma2
    components = object.components
    par = object.par
    lambda = object.lambda
    biasadj = object.biasadj

    nsim = isnothing(nsim) ? length(x) : nsim

    if !isnothing(innov)
        nsim = length(innov)
    end

    if !all(ismissing.(x))
        if isnothing(m)
            m = 1
        end
    else
        if nsim == 0
            nsim = 100
        end
        x = [10]
        future = false
    end

    initstate = future ? states[end, :] : states[rand(1:size(states, 1)), :]

    if bootstrap
        res = filter(!ismissing, residuals) .- mean(filter(!ismissing, residuals))
        e = rand(res, nsim)
    elseif isnothing(innov)
        e = rand(Normal(0, sqrt(sigma2)), nsim)
    elseif length(innov) == nsim
        e = innov
    else
        throw(ArgumentError("Length of innov must be equal to nsim"))
    end

    if components[1] == "M"
        e = max.(-1, e)
    end

    y = zeros(nsim)
    errors = ets_model_type_code(components[1])
    trend = ets_model_type_code(components[2])
    season = ets_model_type_code(components[3])
    alpha = check_component(par, "alpha")
    beta = (trend == 0) ? 0.0 : check_component(par, "beta")
    gamma = (season == 0) ? 0.0 : check_component(par, "gamma")
    phi = parse(Bool, components[4]) ? check_component(par, "phi") : 1.0
    simulate_ets_base(
        initstate,
        m,
        errors,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        nsim,
        y,
        e,
    )

    if abs(y[1] - (-99999.0)) < 1e-7
        error("Problem with multiplicative damped trend")
    end

    if !isnothing(lambda)
        y = inv_box_cox(y, lambda=lambda)
    end

    return y
end

fitted(model::EtsModel) = model.fitted
residuals(model::EtsModel) = model.residuals
