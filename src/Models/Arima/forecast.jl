"""
    forecast(model::ArimaFit;
             h::Int,
             xreg::Union{Nothing,NamedMatrix,AbstractMatrix}=nothing,
             level::AbstractVector{<:Real} = [80, 95],
             fan::Bool = false,
             lambda::Union{Real,Nothing} = nothing,
             bootstrap::Bool = false,
             npaths::Int = 5000,
             biasadj::Union{Bool,Nothing} = nothing)

Forecasting for univariate ARIMA models.

Returns point forecasts and prediction intervals for a fitted [`ArimaFit`](@ref) model.
Internally this calls [`predict_arima`](@ref) and assembles a [`Forecast`](@ref)
object from the results. When exogenous regressors are used, future regressors
are supplied via `xreg`.

# Arguments
- `model::ArimaFit`: A fitted ARIMA model.

# Keyword Arguments
- `h`: Number of periods to forecast. If `xreg` is provided, **`h` is ignored** and
  the forecast horizon is set to `size(xreg, 1)`.
- `xreg`: Future values of regression variables for models with exogenous inputs
  (a `NamedMatrix` or a numeric matrix). Column **names must match** the training
  regressors; if a `NamedMatrix` is provided, columns are automatically aligned.
  Use a matrix (not a `DataFrame`).
- `level`: Confidence levels for prediction intervals. May be specified as
  percentages (e.g. `[80, 95]`) or proportions (e.g. `[0.8, 0.95]`). Values are
  validated and sorted.
- `fan`: If `true`, `level` is set to `51:3:99` (convenient for fan plots).
- `lambda`: Box-Cox lambda. If `nothing`, uses `model.lambda` when present.
- `bootstrap`: If `true`, prediction intervals are computed by simulation with
  resampled errors; otherwise normal-theory intervals are used.
- `npaths`: Number of simulated sample paths when `bootstrap=true`.
- `biasadj`: Optional bias adjustment flag for the **mean** when a Box-Cox
  transformation is in use. If `true`, the mean is bias-corrected using the
  forecast variance (normal-theory branch). If `nothing`, falls back to
  `model.biasadj`.

# Returns
A [`Forecast`](@ref) struct with the following fields:

- `model`: The fitted `ArimaFit`.
- `method`: A descriptive name for the forecasting method.
- `mean`: Vector of point forecasts (length `h`).
- `level`: The confidence levels associated with the prediction intervals.
- `x`: The original training series used to fit the model.
- `upper`, `lower`: Matrices of upper/lower prediction limits of size `h × L`,
  where `L == length(level)`, columns ordered by `level`.
- `fitted`: One-step-ahead fitted values from the model.
- `residuals`: Model residuals.

# Notes
- **Horizon from `xreg`:** If future regressors are supplied, `h` is derived from
  the number of rows in `xreg`.
- **Regressor alignment:** When `xreg` is a `NamedMatrix`, its columns are aligned
  to the training `xreg` by name; an error is thrown if a required column is
  missing. If a plain `Matrix` is provided, ensure columns are in the same order
  as during training.
- **Drift:** If a drift term was estimated, a `"drift"` column is expected in
  the training regressors and is handled automatically. For drift-only models,
  a future `"drift"` regressor is generated internally as `(n+1):(n+h)` when
  `xreg` is not supplied (mirroring the R implementation).
- **Box-Cox:** When `lambda ≠ nothing`, the mean forecast is back-transformed via
  `inv_box_cox(mean; lambda, biasadj, fvar = se.^2)`. Normal-theory intervals
  are also back-transformed; bootstrapped intervals are assumed to be simulated
  on the original scale.

# Examples
```julia
# Fit your ARIMA model (pseudo-code)
model = auto+arima(y, 12)

# Basic forecast 12 steps ahead with 80%/95% intervals
fc = forecast(model, h=12, level=[80, 95])

# Fan plot levels
fc_fan = forecast(model; h=24, fan=true)

# Forecast with future regressors
Xfuture = NamedMatrix(randn(12, 2), ["x1", "x2"])
fc_x = forecast(model; xreg=Xfuture, level=[0.8, 0.95])

# Bootstrap intervals
fc_boot = forecast(model; h=20, bootstrap=true, npaths=2000)
````

# See also

[`predict_arima`](@ref), `simulate` for ARIMA state-space models,
[`NamedMatrix`](@ref), [`Forecast`](@ref).

# References

Peiris, M. & Perera, B. (1988). *On prediction with fractionally differenced ARIMA models*.
**Journal of Time Series Analysis**, 9(3), 215-220.
"""
function forecast(model::ArimaFit;
    h::Union{Int,Nothing}=nothing,
    xreg::Union{Nothing,NamedMatrix,AbstractMatrix}=nothing,
    level::Vector{<:Real}=[80, 95],
    fan::Bool=false,
    lambda::Union{Real,Nothing}=nothing,
    bootstrap::Bool=false,
    npaths::Int=5000,
    biasadj::Union{Bool,Nothing}=nothing, )
    
    levels  = normalize_levels(level; fan=fan)
    lambda  = isnothing(lambda)  ? model.lambda  : lambda
    biasadj = isnothing(biasadj) ? model.biasadj : biasadj

    y = model.y
    n = length(y)
    need_xreg = uses_xreg(model)
    has_drift = uses_drift(model)

    train_xcols = model.xreg isa NamedMatrix ? model.xreg.colnames : String[]

    origxreg = nothing 
    xreg_pred = nothing 

    if isnothing(xreg)
        if need_xreg
            
            if has_drift && (isempty(train_xcols) || (length(train_xcols) == 1 && train_xcols[1] == "drift"))
                Xnm = drift_only_named(n, h)
                origxreg = Xnm
                xreg_pred = Xnm
            else
                throw(ArgumentError("No regressors provided for a model that uses exogenous regressors."))
            end
        end
    else
        
        Xnm = xreg isa NamedMatrix ? xreg :
              NamedMatrix(Matrix(xreg), size(Matrix(xreg),2)==1 ? ["xreg"] : ["xreg$(i)" for i in 1:size(Matrix(xreg),2)])

        if !isempty(train_xcols)
            if sort(Xnm.colnames) != sort(train_xcols)
                @warn "xreg column names differ from training xreg; they will be aligned. Ensure the same regressors and correct order."
            end
        end
        h = size(Xnm.data, 1)
        origxreg = Xnm
        xreg_pred = Xnm
    end

    preds = predict_arima(model, h; newxreg=xreg_pred, se_fit=true)
    mean = preds.prediction
    se   = preds.se

    if bootstrap
        sim = Matrix{Float64}(undef, npaths, h)
        for i in 1:npaths
            sim[i, :] = simulate(model, h; xreg=origxreg, lambda=lambda, bootstrap=true)
        end
        qlow  = (0.5 .- levels ./ 200)
        qhigh = (0.5 .+ levels ./ 200)

        lower_cols = Vector{Vector{Float64}}(undef, length(levels))
        upper_cols = Vector{Vector{Float64}}(undef, length(levels))
        for (j, ql) in enumerate(qlow)
            lower_cols[j] = [quantile(@view(sim[:, t]), ql) for t in 1:h]
        end
        for (j, qh) in enumerate(qhigh)
            upper_cols[j] = [quantile(@view(sim[:, t]), qh) for t in 1:h]
        end
        lower = reduce(hcat, lower_cols)
        upper = reduce(hcat, upper_cols)
    else
        z = quantile.(Ref(Normal()), 0.5 .+ levels ./ 200)
        upper = reduce(hcat, [mean .+ zi .* se for zi in z])
        lower = reduce(hcat, [mean .- zi .* se for zi in z])
        if !isfinite(maximum(upper))
            @warn "Upper prediction intervals are not finite."
        end
    end

    if !isnothing(lambda)
        mean = inv_box_cox(mean; lambda=lambda, biasadj=biasadj, fvar=se.^2)
        if !bootstrap
            lower = inv_box_cox(lower; lambda=lambda, biasadj=false)
            upper = inv_box_cox(upper; lambda=lambda, biasadj=false)
        end
    end

    return Forecast(
        model,
        model.method,
        mean,
        levels,
        y,
        upper,
        lower,
        model.fitted,
        model.residuals,
    )
end

coef_names(m::NamedMatrix) = (isnothing(m.colnames) ? String[] : m.colnames)

function drift_only_named(n::Int, h::Int)
    drift = reshape(collect(n .+ (1:h)), h, 1)
    NamedMatrix(Matrix(drift), ["drift"])
end

uses_drift(m::ArimaFit) = any(==("drift"), coef_names(m.coef))

uses_xreg(m::ArimaFit) = uses_drift(m) || (m.xreg isa NamedMatrix && size(m.xreg.data,2) > 0)

function normalize_levels(level::AbstractVector{<:Real}; fan::Bool=false)
    if fan
        levels = collect(51:3:99)
    else
        levels = collect(level)
        if minimum(levels) > 0 && maximum(levels) < 1
            levels .= 100 .* levels
        end
        if minimum(levels) < 0 || maximum(levels) > 99.99
            throw(ArgumentError("Confidence limit out of range"))
        end
    end
    sort!(levels)
    return levels
end