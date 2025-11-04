using Test
using Durbyn
using Random
import Durbyn.Generics: Forecast, forecast
using Durbyn.Utils: NamedMatrix

@testset "Durbyn.Arima - forecast with xreg columns starting with `ma`" begin
    Random.seed!(123)

    n = 120
    idx = 1:n
    temperature = -(15 .+ 8 .* sin.(2π .* idx ./ 12) .+ 0.5 .* randn(n))
    marketing = -(0.3 .+ 0.15 .* sin.(2π .* idx ./ 6) .+ 0.05 .* randn(n))
    sales = 120 .+ 1.5 .* temperature .+ 30 .* marketing .+ randn(n)

    xreg = NamedMatrix(hcat(temperature, marketing), ["temperature", "marketing"])
    fit = auto_arima(sales, 12, xreg = xreg)

    n_ahead = 24
    future_idx = (n + 1):(n + n_ahead)
    future_temp = -(15 .+ 8 .* sin.(2π .* future_idx ./ 12))
    future_marketing = -(0.3 .+ 0.15 .* sin.(2π .* future_idx ./ 6))
    future_xreg = NamedMatrix(hcat(future_temp, future_marketing), ["temperature", "marketing"])

    fc = forecast(fit; h = n_ahead, xreg = future_xreg)
    @test fc isa Forecast
    @test length(fc.mean) == n_ahead
end

@testset "FittedArima - forecast injects drift column automatically" begin
    Random.seed!(321)

    n = 120
    idx = 1:n
    temperature = -(15 .+ 8 .* sin.(2π .* idx ./ 12) .+ 0.5 .* randn(n))
    marketing = -(0.3 .+ 0.15 .* sin.(2π .* idx ./ 6) .+ 0.05 .* randn(n))
    sales = 120 .+ 1.5 .* temperature .+ 30 .* marketing .+ randn(n)

    data = (sales = sales, temperature = temperature, marketing = marketing)
    spec = ArimaSpec(@formula(sales = d(0) + p(0, 2) + q(0, 2) + temperature + marketing))
    fitted = fit(spec, data, m = 12)

    n_ahead = 24
    future_idx = (n + 1):(n + n_ahead)
    future_temp = -(15 .+ 8 .* sin.(2π .* future_idx ./ 12))
    future_marketing = -(0.3 .+ 0.15 .* sin.(2π .* future_idx ./ 6))
    newdata = (temperature = future_temp, marketing = future_marketing)

    fc = forecast(fitted; h = n_ahead, newdata = newdata)
    @test fc isa Forecast
    @test length(fc.mean) == n_ahead
end
