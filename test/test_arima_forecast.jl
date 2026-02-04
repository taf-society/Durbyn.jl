using Test
using Durbyn
using Random
import Durbyn.Generics: Forecast, forecast
using Durbyn.Utils: NamedMatrix
import Durbyn.Arima: predict_arima, arima_rjh, PDQ

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

@testset "predict_arima - xreg validation" begin
    # Bug fix: providing newxreg when model has no xreg should throw clear error
    Random.seed!(456)
    y = randn(50)
    fit = arima_rjh(y, 1; order=PDQ(1,0,0))

    # Model was fit without xreg, so providing newxreg should error
    newxreg = NamedMatrix(randn(5,1), ["x1"])
    @test_throws ArgumentError predict_arima(fit, 5; newxreg=newxreg)
end

@testset "forecast - plain matrix xreg uses training column names" begin
    # Bug fix: plain matrix should use training column names for positional matching
    Random.seed!(789)

    n = 100
    x1 = randn(n)
    x2 = randn(n)
    y = 50.0 .+ 2.0 .* x1 .+ 3.0 .* x2 .+ randn(n)

    xreg_train = NamedMatrix(hcat(x1, x2), ["temp", "promo"])
    fit = auto_arima(y, 12; xreg=xreg_train)

    # Forecast with plain matrix (positional matching)
    h = 10
    future_xreg = randn(h, 2)
    fc = forecast(fit; h=h, xreg=future_xreg)

    @test fc isa Forecast
    @test length(fc.mean) == h
end

@testset "forecast - plain matrix xreg column count validation" begin
    # Bug fix: plain matrix with wrong column count should throw clear error
    Random.seed!(101)

    n = 100
    x1 = randn(n)
    x2 = randn(n)
    y = 50.0 .+ 2.0 .* x1 .+ 3.0 .* x2 .+ randn(n)

    xreg_train = NamedMatrix(hcat(x1, x2), ["temp", "promo"])
    fit = auto_arima(y, 12; xreg=xreg_train)

    # Forecast with wrong number of columns should error
    h = 10
    wrong_xreg = randn(h, 3)  # 3 columns instead of 2
    @test_throws ArgumentError forecast(fit; h=h, xreg=wrong_xreg)
end

@testset "Residual semantics - fitted ≈ y - residuals" begin
    # Bug fix: residuals should be raw innovations, not standardized
    Random.seed!(202)
    y = randn(80)
    fit = arima_rjh(y, 1; order=PDQ(1,0,0))

    # For a well-fitted model, fitted + residuals should approximately equal y
    # (within numerical tolerance, accounting for conditioning period)
    n_cond = fit.n_cond
    y_subset = fit.y[(n_cond+1):end]
    fitted_subset = fit.fitted[(n_cond+1):end]
    resid_subset = fit.residuals[(n_cond+1):end]

    reconstructed = fitted_subset .+ resid_subset
    @test isapprox(reconstructed, y_subset, rtol=0.1)
end
