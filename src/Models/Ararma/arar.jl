"""
    struct ARAR

Holds the fitted parameters of an ARAR (AutoRegressive with Adaptive Reduction) time series model.

# Fields

- `Ψ::AbstractArray`  
  The memory-shortening filter vector derived during the preprocessing stage of the ARAR algorithm.

- `Sbar::Float64`  
  The mean of the memory-shortened series `S`. Used for reconstructing forecasts and fitted values.

- `gamma::AbstractArray`  
  The autocovariance sequence of the shortened, mean-adjusted series. Used in selecting AR lags 
  and estimating AR coefficients.

- `best_ϕ::AbstractArray`  
  The vector of autoregressive coefficients (φ) estimated after fitting an AR(4) model to the 
  shortened series with selected lags.

- `σ2::Float64`  
  The estimated innovation variance (white noise variance) of the fitted AR model.

- `best_lag::Tuple{Int, Int, Int, Int}`  
  The selected lags used in the final AR model (typically (1, i, j, k)) that yielded the minimum 
  estimated variance.

- `y::AbstractArray`  
  The original time series data the model was fit on (prior to shortening or transformation).

# Notes

This struct is typically returned by a `arar(y)` function and can be passed to forecast and diagnostic 
    methods such as `forecast`, `fitted`, or `residuals`.

"""
struct ARAR
    Ψ::AbstractArray
    Sbar::Float64
    gamma::AbstractArray
    best_ϕ::AbstractArray
    σ2::Float64
    best_lag::Tuple{Int, Int, Int, Int}
    y::AbstractArray
end

function Base.show(io::IO, model::ARAR)
    println(io, "ARAR Model Summary")
    println(io, "------------------")
    println(io, "Number of observations: ", length(model.y))
    println(io, "Selected AR lags: ", model.best_lag)
    println(io, "AR coefficients (ϕ): ", round.(model.best_ϕ, digits=4))
    println(io, "Residual variance (σ²): ", round(model.σ2, digits=4))
    println(io, "Mean of shortened series (S̄): ", round(model.Sbar, digits=4))
    println(io, "Length of memory-shortening filter (Ψ): ", length(model.Ψ))
end

"""
    arar(y::AbstractArray; max_ar_depth::Int=26, max_lag::Int=40) -> ARAR

Fits an Autoregressive model with memory-shortening transformations to a univariate time series.

# Arguments

- `y::AbstractArray`:  
  A one-dimensional array containing the observed time series data.

# Keyword Arguments

- `max_ar_depth::Int=26`:  
  The maximum lag to consider when selecting the best 4-lag AR model. Must be at least 4.

- `max_lag::Int=40`:  
  The maximum lag to use when computing the autocovariance sequence of the memory-shortened series.

# Returns

- `ARAR`:  
  A struct containing the fitted ARAR model components, including memory-shortening coefficients, selected AR lags, AR coefficients, and residual variance.

# Description

The ARAR model combines memory-shortening transformations with autoregressive modeling. It first reduces the memory in the input series via iterative 
filtering (`Ψ`), then fits an AR model with adaptively selected lags to the shortened series.

This method is useful for short or nonstationary time series, especially when conventional ARIMA methods are unstable or overfit.

# Example

```julia
y = randn(100)
model = arar(y; max_ar_depth=20, max_lag=30)
````

Use the resulting `ARAR` model with `forecast(model, h, ...)`, `fitted(model)`, and `residuals(model)` for analysis and prediction.

"""
function arar(y::AbstractArray; max_ar_depth::Int=26, max_lag::Int=40)
    Y = copy(y)
    Ψ = [1.0]

    for _ in 1:3
        n = length(y)
        ϕ = map(τ -> sum(y[(τ + 1):n] .* y[1:(n - τ)]) / sum(y[1:(n - τ)].^2), 1:15)
        err = map(τ -> sum((y[(τ + 1):n] - ϕ[τ] * y[1:(n - τ)]).^2) / sum(y[(τ + 1):n].^2), 1:15)
        τ = argmin(err)
        
        if err[τ] <= 8/n || (ϕ[τ] >= 0.93 && τ > 2)
            y = y[(τ + 1):n] .- ϕ[τ] * y[1:(n - τ)]
            Ψ = [Ψ; zeros(τ)] .- ϕ[τ] .* [zeros(τ); Ψ]
        elseif ϕ[τ] >= 0.93
            A = zeros(2, 2)
            A[1, 1] = sum(y[2:(n - 1)].^2)
            A[1, 2] = A[2, 1] = sum(y[1:(n - 2)] .* y[2:(n - 1)])
            A[2, 2] = sum(y[1:(n - 2)].^2)
            b = [sum(y[3:n] .* y[2:(n - 1)]), sum(y[3:n] .* y[1:(n - 2)])]
            ϕ_2 = (A' * A) \ (A' * b)
            y = y[3:n] .- ϕ_2[1] * y[2:(n - 1)] .- ϕ_2[2] * y[1:(n - 2)]
            Ψ = vcat(Ψ, 0.0, 0.0) .- ϕ_2[1] .* vcat(0.0, Ψ, 0.0) .- ϕ_2[2] .* vcat(0.0, 0.0, Ψ)
        else
            break
        end
    end

    S = y
    Sbar = mean(S)
    X = S .- Sbar
    n = length(X)
    xbar = mean(X)
    gamma = map(i -> sum((X[1:(n - i)] .- xbar) .* (X[(i + 1):n] .- xbar)) / n, 0:max_lag)

    A = fill(gamma[1], 4, 4)
    b = zeros(4)
    best_σ2 = Inf
    best_lag = (1, 0, 0, 0)
    best_phi = zeros(4)

    for i in 2:(max_ar_depth - 2), j in (i + 1):(max_ar_depth - 1), k in (j + 1):max_ar_depth
        A[1,2] = A[2,1] = gamma[i]
        A[1,3] = A[3,1] = gamma[j]
        A[2,3] = A[3,2] = gamma[j - i + 1]
        A[1,4] = A[4,1] = gamma[k]
        A[2,4] = A[4,2] = gamma[k - i + 1]
        A[3,4] = A[4,3] = gamma[k - j + 1]
        b .= [gamma[2], gamma[i + 1], gamma[j + 1], gamma[k + 1]]
        ϕ = (A' * A) \ (A' * b)
        σ2 = gamma[1] - ϕ[1] * gamma[2] - ϕ[2] * gamma[i + 1] - ϕ[3] * gamma[j + 1] - ϕ[4] * gamma[k + 1]

        if σ2 < best_σ2
            best_σ2 = σ2
            best_phi = copy(ϕ)
            best_lag = (1, i, j, k)
        end
    end

    return ARAR(Ψ, Sbar, gamma, best_phi, best_σ2, best_lag, Y)
end

function compute_xi(Ψ::AbstractArray, ϕ::AbstractArray, lags::Tuple{Int, Int, Int, Int})
    _, i, j, k = lags
    xi = [Ψ; zeros(k)]
    xi .-= ϕ[1] .* vcat(0.0, Ψ, zeros(k - 1))
    xi .-= ϕ[2] .* vcat(zeros(i), Ψ, zeros(k - i))
    xi .-= ϕ[3] .* vcat(zeros(j), Ψ, zeros(k - j))
    xi .-= ϕ[4] .* vcat(zeros(k), Ψ)
    return xi
end
"""
    fitted(model::ARAR) -> Vector{Union{Float64, Missing}}

Computes the fitted values from an `ARAR` model.

# Arguments

- `model::ARAR`:  
  A fitted ARAR model, as returned by the `arar(y)` function.

# Returns

- `Vector{Union{Float64, Missing}}`:  
  A vector of fitted values corresponding to the original input series.  
  The first `k` values (where `k` is the length of the filter `xi`) are returned as `missing`, 
  since the model cannot generate forecasts for the initial time points due to lag requirements.

# Description

The function reconstructs the one-step-ahead in-sample forecasts using the `Ψ` filter, 
    the selected AR coefficients `ϕ`, the estimated mean `S̄` of the memory-shortened series, 
    and the full lag filter `xi`.

This is useful for evaluating model fit and computing residuals.

# Example

```julia
y = randn(100)
model = arar(y)
fitted_vals = fitted(model)
```
"""
function fitted(model::ARAR)
    y = model.y
    ϕ = model.best_ϕ
    Ψ = model.Ψ
    Sbar = model.Sbar
    lags = model.best_lag
    xi = compute_xi(Ψ, ϕ, lags)
    k = length(xi)
    n = length(y)
    c = (1 - sum(ϕ)) * Sbar

    fitted = fill(NaN, n)
    for t in k:n
        fitted[t] = -sum(xi[2:k] .* y[t .- (1:(k-1))]) + c
    end
    return fitted
end

"""
    residuals(model::ARAR) -> Vector{Union{Float64, Missing}}

Computes the residuals (observed - fitted) from a fitted `ARAR` model.

# Arguments

- `model::ARAR`:  
  A fitted ARAR model, as returned by the `arar(y)` function.

# Returns

- `Vector{Union{Float64, Missing}}`:  
  The residuals from the model.  
  The first `k` values (where `k = length(xi)`) are returned as `missing` because the model 
  cannot generate valid fitted values for those early points due to lag requirements.

# Example

```julia
y = randn(120)
model = arar(y)
resid = residuals(model)
"""
function residuals(model::ARAR)
    y = model.y
    fits = fitted(model)
    return y .- fits
end
"""
    forecast(model::ARAR, h::Int; level::Vector{Int} = [80, 95]) -> Forecast

Generates point forecasts and prediction intervals from a fitted `ARAR` model.

# Arguments

- `model::ARAR`:  
  A fitted ARAR model, returned by the `arar(y)` function.

# Keyword Arguments
- `h::Int`:  
  The number of steps ahead to forecast.
  
- `level::Vector{Int} = [80, 95]`:  
  Confidence levels (in percentages) for prediction intervals.  
  Each value should be between 1 and 99.

# Returns

- `Forecast`:  
  A struct containing:
  - `model`: the original ARAR model
  - `method`: the name of the forecasting method ("ARAR")
  - `mean`: point forecasts of length `h`
  - `level`: the confidence levels used
  - `x`: the original time series
  - `upper`: upper bounds for each confidence level
  - `lower`: lower bounds for each confidence level
  - `fitted`: in-sample fitted values
  - `residuals`: residuals (observed - fitted)

# Description

The function uses the memory-shortening filter `Ψ`, selected AR coefficients `ϕ`, and the impulse response 
    filter `xi` to generate iterative forecasts.  
It computes standard errors for the forecast horizon and uses the normal distribution to construct symmetric 
    prediction intervals at the specified confidence levels.

This is useful for both short-term forecasting and uncertainty quantification in time series with long memory.

# Example

```julia
y = randn(100)
model = arar(y)
fc = forecast(model, 12, level = [80, 95])

println("Forecast mean:", fc.mean)
println("95% upper bound:", fc.upper[2])
"""
function forecast(model::ARAR; h::Int, level::Vector{Int}=[80, 95])
    i, j, k = model.best_lag[2:end]
    ϕ = model.best_ϕ
    σ2 = model.σ2
    Ψ = model.Ψ
    Sbar = model.Sbar
    y = copy(model.y)
    n = length(y)

    xi = [Ψ; zeros(k)] .-
         ϕ[1] .* vcat(0.0, Ψ, zeros(k - 1)) .-
         ϕ[2] .* vcat(zeros(i), Ψ, zeros(k - i)) .-
         ϕ[3] .* vcat(zeros(j), Ψ, zeros(k - j)) .-
         ϕ[4] .* vcat(zeros(k), Ψ)

    y = [y; zeros(h)]
    k = length(xi)
    c = (1 - sum(ϕ)) * Sbar

    meanfc = map(i -> y[n + i] = -sum(xi[2:k] .* y[n + i .+ 1 .- (2:k)]) + c, 1:h)

    if h > k
        xi = append!(xi, zeros(h - k))
    end

    τ = zeros(h)
    τ[1] = 1.0
    for j in 2:h
        τ[j] = -sum(τ[1:j - 1] .* xi[j:-1:2])
    end

    se = sqrt.(σ2 .* map(j -> sum(τ[1:j] .^ 2), 1:h))
    z = level .|> l -> quantile(Normal(), 0.5 + l / 200)
    upper = reduce(hcat,[meanfc .+ zi .* se for zi in z])
    lower = reduce(hcat,[meanfc .- zi .* se for zi in z])

    fits = fitted(model)
    y = copy(model.y)
    res = y .- fits

    return Forecast(model, "Arar Model", meanfc, level, y, upper, lower, fits, res)
end