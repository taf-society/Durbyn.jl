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
        lambda::Union{Float64,Bool,Nothing,String} = nothing,
        biasadj::Bool = false,
        lower::AbstractArray = [0.0001, 0.0001, 0.0001, 0.8],
        upper::AbstractArray = [0.9999, 0.9999, 0.9999, 0.98],
        opt_crit::String = "lik",
        nmse::Int = 3,
        bounds::String = "both",
        ic::String = "aicc",
        restrict::Bool = true,
        allow_multiplicative_trend::Bool = false,
        use_initial_values::Bool = false,
        na_action_type::String = "na_contiguous",
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
  If `false`, treated as “not set” during model selection (estimate if needed).
- `beta::Union{Float64,Bool,Nothing}=nothing`: Trend smoothing parameter. Same rules as `alpha`.
- `gamma::Union{Float64,Bool,Nothing}=nothing`: Seasonal smoothing parameter. Same rules as `alpha`.
- `phi::Union{Float64,Bool,Nothing}=nothing`: Damping parameter (for damped trend). If `nothing`, estimated.
- `additive_only::Bool=false`: If `true`, restrict search to additive-error/season/trend models.
- `lambda::Union{Float64,Bool,Nothing,String}=nothing`: Box-Cox transform parameter. Use a numeric
  value to apply a fixed transform; pass `"auto"` to select via `BoxCox.lambda`-style search;
  `nothing` = no transform. When set (including `"auto"`), only additive models are considered.
- `biasadj::Bool=false`: Use bias-adjusted back-transformation to return *mean* (not median) forecasts
  and fits when Box-Cox is used.
- `lower::AbstractArray=[1e-4, 1e-4, 1e-4, 0.8]`: Lower bounds for `(alpha, beta, gamma, phi)`.
  Ignored if `bounds=="admissible"`.
- `upper::AbstractArray=[0.9999, 0.9999, 0.9999, 0.98]`: Upper bounds for `(alpha, beta, gamma, phi)`.
  Ignored if `bounds=="admissible"`.
- `opt_crit::String="lik"`: Optimization criterion. One of
  `"lik"` (log-likelihood), `"amse"` (average MSE over first `nmse` horizons),
  `"mse"`, `"sigma"` (stdev of residuals), or `"mae"`.
- `nmse::Int=3`: Horizons for `"amse"` (1 ≤ `nmse` ≤ 30).
- `bounds::String="both"`: Parameter space restriction:
  `"usual"` enforces `[lower, upper]`, `"admissible"` enforces ETS admissibility,
  `"both"` uses their intersection.
- `ic::String="aicc"`: Information criterion used for model selection; one of `"aicc"`, `"aic"`, `"bic"`.
- `restrict::Bool=true`: Disallow models with infinite variance.
- `allow_multiplicative_trend::Bool=false`: If `true`, multiplicative trends may be considered when
  searching. Ignored if a multiplicative trend is explicitly requested in `model`.
- `use_initial_values::Bool=false`: If `model isa ETS` and `true`, reuse both its structure and
  initial states (no re-estimation of initials). If `false`, initials are re-estimated.
- `na_action_type::String="na_contiguous"`: Handling of missing values in `y`. One of
  `"na_contiguous"` (use largest contiguous block), `"na_interp"` (interpolate), `"na_fail"` (error).
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
fit3 = ets(ap, 12, "ZZZ"; damped=nothing, lambda="auto", biasadj=true)
fc3 = forecast(fit3, h=12)
plot(fc3)
````
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
    lambda::Union{Float64,Bool,Nothing,String} = nothing,
    biasadj::Bool = false,
    lower::AbstractArray = [0.0001, 0.0001, 0.0001, 0.8],
    upper::AbstractArray = [0.9999, 0.9999, 0.9999, 0.98],
    opt_crit::String = "lik",
    nmse::Int = 3,
    bounds::String = "both",
    ic::String = "aicc",
    restrict::Bool = true,
    allow_multiplicative_trend::Bool = false,
    use_initial_values::Bool = false,
    na_action_type::String = "na_contiguous",
    options::NelderMeadOptions = NelderMeadOptions()
)

    if model == "ZZZ" && is_constant(y)
        return ses(y, alpha = 0.99999, initial = "simple")
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
        na_action_type = na_action_type,
        options=options
    )

    return out
end
