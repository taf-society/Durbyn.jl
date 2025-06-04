using Statistics
using Optim
using Plots

# Existing struct
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
    pvh::Vector{Float64}
    aic::Vector{Float64}
    cat::Vector{Float64}
end

function compute_xi(Ψ::Vector{Float64}, ϕ::Vector{Float64}, lags::NTuple{4, Int})
    _, i, j, k_lag = lags
    Ψlen = length(Ψ)
    xi = [Ψ; zeros(k_lag)]

    ϕ_pad = vcat(ϕ, zeros(4 - length(ϕ)))

    xi .-= ϕ_pad[1] .* vcat(0.0, Ψ, zeros(k_lag - 1))
    xi .-= ϕ_pad[2] .* vcat(zeros(i), Ψ, zeros(k_lag - i))
    xi .-= ϕ_pad[3] .* vcat(zeros(j), Ψ, zeros(k_lag - j))
    xi .-= ϕ_pad[4] .* vcat(zeros(k_lag), Ψ)

    return xi
end

function compute_aic_cat(gamma::Vector{Float64}, max_order::Int, n::Int)
    aic = Float64[]
    cat = Float64[]
    for m in 1:max_order
        R = [gamma[abs(i - j) + 1] for i in 1:m, j in 1:m]
        r = gamma[2:m+1]
        ϕ = R \ r
        σ2 = gamma[1] - dot(ϕ, r)
        push!(aic, log(σ2) + 2 * m / n)
        push!(cat, log(σ2) + 2 * m * (m + 1) / n^2)
    end
    return aic, cat
end

function compute_pvh(ϕ::Vector{Float64}, h_max::Int, σ2::Float64)
    ma_coefs = [1.0]
    for h in 1:h_max
        coef = sum(-ϕ[j] * ma_coefs[h - j + 1] for j in 1:min(h, length(ϕ)))
        push!(ma_coefs, coef)
    end
    forecast_var = [sum(ma_coefs[2:h+1] .^ 2) * σ2 for h in 1:h_max]
    return 1 .- forecast_var
end

# Simplified arma fitting and core ARARMA entry point can be filled in similarly
# Here we define only the integration block for brevity
function ararma(y::Vector{Float64}; max_ar_depth::Int=26, max_lag::Int=40, p::Int=4, q::Int=1)
    Y = copy(y)
    Ψ = [1.0]

    for _ in 1:3
        n = length(y)
        ϕs = map(τ -> sum(y[(τ + 1):n] .* y[1:(n - τ)]) / sum(y[1:(n - τ)].^2), 1:15)
        errs = map(τ -> sum((y[(τ + 1):n] - ϕs[τ] * y[1:(n - τ)]).^2) / sum(y[(τ + 1):n].^2), 1:15)
        τ = argmin(errs)

        if errs[τ] <= 8/n || (ϕs[τ] >= 0.93 && τ > 2)
            y = y[(τ + 1):n] .- ϕs[τ] * y[1:(n - τ)]
            Ψ = [Ψ; zeros(τ)] .- ϕs[τ] .* [zeros(τ); Ψ]
        else
            break
        end
    end

    Sbar = mean(y)
    X = y .- Sbar
    n = length(X)
    gamma = map(i -> sum((X[1:(n - i)] .- mean(X)) .* (X[(i + 1):n] .- mean(X))) / n, 0:max_lag)

    aic, cat = compute_aic_cat(gamma, 10, n)
    best_order = argmin(aic) + 1

    # (Assume best_lag and best_ϕ are determined from gamma like before)
    # Dummy best_ϕ and lag for example
    best_ϕ = randn(best_order)
    best_lag = (1, 2, 3, 4)
    σ2 = gamma[1] - dot(best_ϕ, gamma[2:best_order+1])

    pvh_vals = compute_pvh(best_ϕ, 20, σ2)

    return ArarmaModel(Ψ, Sbar, gamma, best_ϕ, σ2, best_lag, Y, zeros(q), q, p, pvh_vals, aic, cat)
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

forecast(model, 12)