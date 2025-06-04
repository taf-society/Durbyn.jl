



using Statistics
using Optim

struct ArarmaModel
    Ψ::Vector{Float64}
    Sbar::Float64
    gamma::Vector{Float64}
    best_ϕ::Vector{Float64}  # AR part of residuals
    σ2::Float64
    best_lag::Tuple{Int, Int, Int, Int}
    y_original::Vector{Float64}
    best_θ::Vector{Float64}  # MA part of residuals
    ma_order::Int
    ar_order::Int
end

function compute_xi(Ψ::Vector{Float64}, ϕ::Vector{Float64}, lags::NTuple{4, Int})
    _, i, j, k_lag = lags
    Ψlen = length(Ψ)
    xi = [Ψ; zeros(k_lag)]

    ϕ_pad = vcat(ϕ, zeros(4 - length(ϕ)))  # ensures ϕ has 4 elements

    xi .-= ϕ_pad[1] .* vcat(0.0, Ψ, zeros(k_lag - 1))
    xi .-= ϕ_pad[2] .* vcat(zeros(i), Ψ, zeros(k_lag - i))
    xi .-= ϕ_pad[3] .* vcat(zeros(j), Ψ, zeros(k_lag - j))
    xi .-= ϕ_pad[4] .* vcat(zeros(k_lag), Ψ)

    return xi
end



function fit_arma(p::Int, q::Int, y::Vector{Float64})
    n = length(y)
    μ = mean(y)
    y_demeaned = y .- μ
    max_lag = max(p, q)

    function arma_loss(params)
        ϕ = params[1:p]
        θ = params[p+1:p+q]
        σ2 = abs(params[end])
        ε = zeros(n)

        for t in (max_lag + 1):n
            ar_part = dot(ϕ, reverse(y_demeaned[t-p:t-1]))
            ma_part = dot(θ, reverse(ε[t-q:t-1]))
            ε[t] = y_demeaned[t] - (ar_part + ma_part)
        end
        return sum(ε.^2) / σ2 + n * log(σ2 + 1e-8)
    end

    init = [zeros(p + q); var(y)]
    result = optimize(arma_loss, init, BFGS())
    est_params = Optim.minimizer(result)
    return (est_params[1:p], est_params[p+1:p+q], est_params[end])
end

function ararma(y::Vector{Float64}; max_ar_depth::Int=26, max_lag::Int=40, p::Int=4, q::Int=1)
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

    Sbar = mean(y)
    X = y .- Sbar
    n = length(X)
    gamma = map(i -> sum((X[1:(n - i)] .- mean(X)) .* (X[(i + 1):n] .- mean(X))) / n, 0:max_lag)

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

    xi = compute_xi(Ψ, best_phi, best_lag)
    k = length(xi)
    Y_ext = copy(Y)
    n = length(Y_ext)
    c = (1 - sum(best_phi)) * Sbar

    fitted_vals = fill(NaN, n)
    for t in k:n
        fitted_vals[t] = -sum(xi[2:k] .* Y_ext[t .- (1:(k - 1))]) + c
    end

    residuals = Y_ext .- fitted_vals
    ϕ_ar, θ_ma, σ2_hat = fit_arma(p, q, residuals[k:end])

    return ArarmaModel(Ψ, Sbar, gamma, ϕ_ar, σ2_hat, best_lag, Y, θ_ma, q, p)
end

function fitted(model::ArarmaModel)
    y = model.y_original
    xi = compute_xi(model.Ψ, model.best_ϕ, model.best_lag)
    n = length(y)
    k = length(xi)
    q = model.ma_order
    c = (1 - sum(model.best_ϕ)) * model.Sbar
    ϕ = model.best_ϕ
    θ = model.best_θ

    fitted_vals = fill(NaN, n)
    ε = zeros(n)

    for t in k:n
        ar_part = -sum(xi[2:k] .* y[t .- (1:(k - 1))]) + c
        ma_part = (t > q) ? dot(θ, reverse(ε[t-q:t-1])) : 0.0
        fitted_vals[t] = ar_part + ma_part
        ε[t] = y[t] - fitted_vals[t]
    end

    return fitted_vals
end

function residuals(model::ArarmaModel)
    return model.y_original .- fitted(model)
end

function forecast(model::ArarmaModel, h::Int, level::Vector{Int} = [80, 95])
    y = copy(model.y_original)
    n = length(y)
    xi = compute_xi(model.Ψ, model.best_ϕ, model.best_lag)
    k = length(xi)
    p = model.ar_order
    q = model.ma_order
    θ = model.best_θ
    ϕ = model.best_ϕ
    σ2 = model.σ2
    Sbar = model.Sbar
    c = (1 - sum(ϕ)) * Sbar

    # Initialize vectors
    y_ext = [y; zeros(h)]
    ε_ext = zeros(n + h)
    forecasts = zeros(h)

    for t in 1:h
        idx = n + t
        ar_part = -sum(xi[2:k] .* y_ext[idx .- (1:(k - 1))]) + c
        ma_part = t > q ? dot(θ[1:q], reverse(ε_ext[idx - q:idx - 1])) : 0.0
        forecasts[t] = ar_part + ma_part
        ε_ext[idx] = 0.0  # Set to 0 (conditional expectation)
        y_ext[idx] = forecasts[t]
    end

    # Forecast error variance (recursive)
    τ = zeros(h)
    τ[1] = 1.0
    for j in 2:h
        τ[j] = -sum(τ[1:j-1] .* xi[j:-1:2])
    end

    se = sqrt.(σ2 .* map(j -> sum(τ[1:j] .^ 2), 1:h))
    z = level .|> l -> quantile(Normal(), 0.5 + l / 200)

    upper = [forecasts .+ zi .* se for zi in z]
    lower = [forecasts .- zi .* se for zi in z]

    return Dict(
        "forecast" => forecasts,
        "lower" => lower,
        "upper" => upper,
        "level" => level
    )
end


using Durbyn

ap = air_passengers();
using LinearAlgebra
using Distributions

model = ararma(ap, p=3, q=1)

forecast2(model, 12)

yhat = fitted(model)
resid = residuals(model)
