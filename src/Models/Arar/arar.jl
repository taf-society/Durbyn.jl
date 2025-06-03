struct ARAR
    Ψ::AbstractArray
    Sbar::Float64
    gamma::AbstractArray
    best_ϕ::AbstractArray
    σ2::Float64
    best_lag::Tuple{Int, Int, Int, Int}
    y::AbstractArray
end

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

function residuals(model::ARAR)
    y = model.y
    fits = fitted(model)
    return y .- fits
end

function forecast(model::ARAR, h::Int, level::Vector{Int}=[80, 95])
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