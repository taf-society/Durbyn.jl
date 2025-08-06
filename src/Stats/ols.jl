"""
    OlsFit

A struct that holds the results of ordinary least squares (OLS) linear regression.

# Fields
- `coef::Vector{Float64}`: Estimated coefficients.
- `fitted::Vector{Float64}`: Fitted values.
- `residuals::Vector{Float64}`: Residuals (observed minus fitted).
- `sigma2::Float64`: Estimated residual variance.
- `cov::Matrix{Float64}`: Covariance matrix of coefficient estimates.
- `se::Vector{Float64}`: Standard errors of the coefficients.
- `df_residual::Int`: Residual degrees of freedom (n - p).

# Examples
```julia
ols_model = ols(y, X)
println(ols_model.coef)
````

"""
struct OlsFit
coef::Vector{Float64}
fitted::Vector{Float64}
residuals::Vector{Float64}
sigma2::Float64
cov::Matrix{Float64}
se::Vector{Float64}
df_residual::Int
end

"""
ols(y, X) -> OlsFit

Fits an ordinary least squares (OLS) linear regression model.

# Arguments

* `y::Vector{<:Real}`: The response variable (length n).
* `X::Matrix{<:Real}`: The design matrix (size n × p).
*Must include a column of ones if an intercept is desired.*

# Returns

* An `OlsFit` object containing coefficients, residuals, 
fitted values, variance, standard errors, and more.

# Example

```julia
using Random
Random.seed!(1)
n = 100
x1 = randn(n)
x2 = randn(n)
X = hcat(ones(n), x1, x2)
y = X * [2, 3, -1] .+ randn(n) * 2
model = ols(y, X)
println(model.coef)
```

"""
function ols(y, X)
β = X \ y
fitted = X * β
residuals = y - fitted
n, p = size(X)
df_residual = n - p
σ2 = sum(residuals .^ 2) / df_residual
XtX = X' * X
cov_β = σ2 * inv(XtX)
se = sqrt.(diag(cov_β))
return OlsFit(β, fitted, residuals, σ2, cov_β, se, df_residual)
end

"""
predict(model::OlsFit, Xnew) -> Vector{Float64}

Predicts the response variable for new data using a fitted OLS model.

# Arguments

* `model::OlsFit`: The fitted OLS model (from `ols`).
* `Xnew::Matrix{<:Real}`: New data matrix (n_new × p), same structure as X used in training.

# Returns

* Vector of predicted values.

# Example

```julia
Xnew = hcat(ones(5), randn(5), randn(5))
yhat = predict(model, Xnew)
```

"""
function predict(model::OlsFit, Xnew)
return Xnew * model.coef
end

"""
    residuals(model::OlsFit)

Return the residuals from an OlsFit object.
"""
residuals(model::OlsFit) = model.residuals

"""
    coef(model::OlsFit)
    coefficients(model::OlsFit)
    coefs(model::OlsFit)

Return the estimated regression coefficients from the fitted model.
"""
coef(model::OlsFit) = model.coef
coefficients(model::OlsFit) = model.coef
coefs(model::OlsFit) = model.coef

"""
    modelrank(model::OlsFit)

Returns the rank of the OlsFit.
"""
modelrank(model::OlsFit) = rank(model.X)


function show(io::IO, model::OlsFit)
    println(io, "Ordinary Least Squares Fit:")
    println(io, "Coefficients:")
    for (i, b) in enumerate(model.coef)
        println(io, "  β[$i] = $(round(b, digits=4)) (SE = $(round(model.se[i], digits=4)))")
    end
    println(io, "Residual standard error: $(round(sqrt(model.sigma2), digits=4)) on $(model.df_residual) degrees of freedom")
end