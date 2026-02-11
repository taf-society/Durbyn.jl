time_index(n::Int, m::Int; start::Float64 = 1.0) = start .+ (0:n-1) ./ max(m, 1)

has_coef(fit::ArimaFit, name::AbstractString) = any(==(name), fit.coef.colnames)

npar(fit::ArimaFit) = count(identity, fit.mask) + 1

function n_and_nstar(fit::ArimaFit)
    # R's Arima() computes n = count(non-NA residuals) = length(x) for all
    # methods (CSS sets first ncond residuals to 0, not NA; ML/CSS-ML are
    # all valid). So n = length(y) always.
    n = length(fit.y)
    d = Int(fit.arma[6])
    D = Int(fit.arma[7])
    m = Int(fit.arma[5])
    nstar = n - d - D * m
    return n, nstar
end

function prepend_drift(xreg::Union{Nothing,NamedMatrix}, drift::AbstractVector{<:Real})
    driftcol = reshape(Float64.(drift), :, 1)
    return isnothing(xreg) ? NamedMatrix(driftcol, ["drift"]) :
           add_drift_term(xreg, driftcol, "drift")
end

function prepare_drift(model::ArimaFit, x, xreg::Union{Nothing,NamedMatrix})
    n_train = length(model.y)
    m_train = Int(model.arma[5])
    t_train = time_index(n_train, m_train)
    @assert model.xreg isa NamedMatrix "Original model has no xreg for drift reconstruction"
    @assert any(==("drift"), model.xreg.colnames) "Original model has no 'drift' column"

    drift_vec = get_vector(model.xreg; col = "drift")

    # OLS: drift_vec ~ a + b * t_train
    X = hcat(ones(n_train), t_train)
    fitt = ols(drift_vec, X)
    coef = coefficients(fitt)
    a, b = coef[1], coef[2]

    n_new = length(x)
    m_new = Int(model.arma[5])
    t_new = time_index(n_new, m_new)
    newdr = a .+ b .* t_new
    xreg_with_drift = prepend_drift(xreg, newdr)
    return align_columns(xreg_with_drift, model.xreg.colnames)
end

function refit_arima_model(
    x,
    _m::Int,
    model::ArimaFit,
    xreg::Union{Nothing,NamedMatrix},
    method::String;
    kwargs...,
)

    p, q, P, Q, m_model, d, D = model.arma
    if _m != m_model
        @warn "Ignoring supplied m=$_m; using model's seasonal period m=$m_model"
    end

    order = PDQ(p, d, q)
    seasonal = PDQ(P, D, Q)

    fit = arima(
        x,
        m_model;
        order = order,
        seasonal = seasonal,
        xreg = xreg,
        include_mean = has_coef(model, "intercept"),
        method = method,
        fixed = model.coef.data,
        kwargs...,
    )

    k = size(fit.coef.data, 2)
    fit.var_coef .= 0.0
    fit.sigma2 = model.sigma2
    if !isnothing(xreg)
        fit.xreg = xreg
    end
    return fit
end

"""
    arima_rjh(
        y,
        m::Int;
        order::PDQ = PDQ(0, 0, 0),
        seasonal::PDQ = PDQ(0, 0, 0),
        xreg::Union{Nothing,NamedMatrix} = nothing,
        include_mean::Bool = true,
        include_drift::Bool = false,
        include_constant = nothing,
        lambda = nothing,
        biasadj::Bool = false,
        method::String = "CSS-ML",
        model::Union{Nothing,ArimaFit} = nothing,
        kwargs...
    ) -> ArimaFit

Fit an ARIMA model to a **univariate** time series.

This is a Julia adaptation of Rob J. Hyndman's ARIMA routine (a wrapper around
`stats::arima`) with two key extensions: support for a **drift** term and
optional **Box-Cox transformation** with mean (bias) adjustment on the
back-transformed scale. You can also pass a previously fitted model and
re-apply it to new data `y` without re-estimating parameters.

# Arguments
- `y`: Univariate time series (vector or `AbstractArray`) to be modeled.
- `m::Int`: Seasonal period (e.g., 12 for monthly data, 4 for quarterly).
- `order::PDQ`: Non-seasonal orders `(p, d, q)`.
- `seasonal::PDQ`: Seasonal orders `(P, D, Q)`; use `m` to control the period.
- `xreg::Union{Nothing,NamedMatrix}`: Optional exogenous regressors with the same number of rows as `y`. Must be numeric; not a DataFrame.
- `include_mean::Bool = true`: Include an intercept/mean term for **undifferenced** series; ignored when differencing is present in a way that the mean is not identifiable.
- `include_drift::Bool = false`: Include a **linear drift** term (i.e., regression with ARIMA errors).
- `include_constant = nothing`: If `true`, sets `include_mean = true` for undifferenced series and `include_drift = true` for differenced series. If more than one difference is taken in total (`d + D > 1`), **no constant** is included regardless of this setting (to avoid inducing higher-order polynomial trends).
- `lambda = nothing`: Box-Cox transformation parameter.
  - `nothing` → no transformation.
  - Real value → apply Box-Cox with that λ (`λ = 0` corresponds to log transform, `λ = 1` to no transform).
  - `:auto` (if used) → select λ automatically (BoxCox.lambda equivalent).
- `biasadj::Bool = false`: When `lambda` is set, use bias-adjusted back-transformation so that fitted values and forecasts approximate **means** on the original scale (otherwise they approximate medians).
- `method::String = "CSS-ML"`: Estimation method. One of `"CSS-ML"`, `"ML"`, `"CSS"`.
  - `"CSS-ML"` uses Conditional Sum of Squares for starts, then Maximum Likelihood.
  - `"ML"` uses full Maximum Likelihood.
  - `"CSS"` minimizes Conditional Sum of Squares.
- `model::Union{Nothing,ArimaFit} = nothing`: Output from a previous call; when provided, the same model structure is refit to `y` **without** re-estimating parameters.
- `kwargs...`: Passed through to the underlying optimizer/likelihood routine (advanced use).

# Details
The fitted model is a regression with ARIMA errors:

 ```math
y_t = c + β' x_t + z_t
```

where `x_t` are exogenous regressors (if any) and `z_t` follows an
ARIMA(p, d, q)*(P, D, Q)[m] process.  
If there are no regressors and `d = D = 0`, the intercept `c` estimates the
series mean. When a Box-Cox transform is used, the model is estimated on the
transformed scale; fitted values and forecasts are then back-transformed, with
optional bias adjustment (`biasadj = true`) to target the mean on the original
scale.

# Returns
An [`ArimaFit`] object containing:
- `model::ArimaFit`: coefficients, fitted values, residuals, information criteria,
  likelihood, variance estimates, convergence info, etc.

Notable components inside `model` include:
- `sigma2`: Bias-adjusted MLE of the innovation variance.
- `x`: The (possibly transformed) time series used in estimation.
- `xreg`: The regressors used (if supplied).

# Notes
- `include_constant=true` is a convenience that sets sensible defaults for
  `include_mean`/`include_drift` based on differencing, mirroring Hyndman's design.
- If `d + D > 1`, no constant is included irrespective of settings, to avoid
  induced quadratic or higher-order trends.
- Passing `model` refits that model structure to `y` without re-optimization.

# References
- Hyndman, R.J. & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.), OTexts.
- Base algorithmic details follow `arima`.

# Examples
```julia
# Seasonal monthly series with drift and automatic λ selection
fit = arima_rjh(
    y, 12;
    order = PDQ(1,1,1),
    seasonal = PDQ(0,1,1),
    include_drift = true,
    lambda = :auto,
    method = "CSS-ML"
)

# Refit the same model structure to new data y2
fit2 = arima_rjh(y2, 12; model = fit.model)

# With exogenous regressors
fitx = arima_rjh(
    y, 4;
    order = PDQ(0,1,1),
    seasonal = PDQ(0,1,1),
    xreg = X,
    include_mean = false,
    method = "ML"
)
```
"""
function arima_rjh(
    y,
    m::Int;
    order::PDQ = PDQ(0, 0, 0),
    seasonal::PDQ = PDQ(0, 0, 0),
    xreg::Union{Nothing,NamedMatrix} = nothing,
    include_mean::Bool = true,
    include_drift::Bool = false,
    include_constant = nothing,
    lambda = nothing,
    biasadj::Bool = false,
    method::String = "CSS-ML",
    model::Union{Nothing,ArimaFit} = nothing,
    kwargs...,
)

    x2 = copy(y)
    method = match_arg(method, ["CSS-ML", "ML", "CSS"])

    if !isnothing(lambda)
        x2, lambda = box_cox(x2, m; lambda = lambda)
    end

    seasonal2 = (m <= 1) ? PDQ(0, 0, 0) : seasonal
    if (m <= 1) && (length(x2) <= order.d)
        error("Not enough data to fit the model")
    elseif (m > 1) && (length(x2) <= order.d + seasonal2.d * m)
        error("Not enough data to fit the model")
    end

    if !isnothing(include_constant)
        if include_constant === true
            include_mean = true
            if (order.d + seasonal2.d) == 1
                include_drift = true
            end
        else
            include_mean = false
            include_drift = false
        end
    end
    if (order.d + seasonal2.d) > 1 && include_drift
        @warn "No drift term fitted as the order of difference is 2 or more."
        include_drift = false
    end

    fit = nothing
    if !isnothing(model)
        had_xreg = (model.xreg isa NamedMatrix)
        use_drift = had_xreg && any(==("drift"), model.xreg.colnames)

        if had_xreg && isnothing(xreg)
            error("No regressors provided")
        end

        xreg2 =
            use_drift ? prepare_drift(model, x2, xreg) :
            (had_xreg ? align_columns(xreg, model.xreg.colnames) : xreg)

        fit = refit_arima_model(x2, m, model, xreg2, method; kwargs...)
    else
        xreg2 = include_drift ? prepend_drift(xreg, 1:length(x2)) : xreg
        fit = arima(
            x2,
            m;
            order = order,
            seasonal = seasonal2,
            xreg = xreg2,
            include_mean = include_mean,
            method = method,
            kwargs...,
        )
    end

    n, nstar = n_and_nstar(fit)
    np = npar(fit)

    if !isnothing(fit.aic)
        fit.aicc = fit.aic + 2 * np * (nstar / (nstar - np - 1) - 1)
        fit.bic = fit.aic + np * (log(nstar) - 2)
    end

    if isnothing(model)
        ss = sum(v -> v * v, fit.residuals[(fit.n_cond+1):end])
        fit.sigma2 = ss / (nstar - np + 1)
    end
    if isnothing(model) && !isnothing(xreg2)
        fit.xreg = xreg2
    end
    
    fit.lambda = lambda

    return fit
end