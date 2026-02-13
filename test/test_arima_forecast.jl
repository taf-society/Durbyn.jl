using Test
using Durbyn
using Random
import Durbyn.Generics: Forecast, forecast, fitted, residuals
using Durbyn.Utils: NamedMatrix
import Durbyn.Arima: predict_arima, arima_rjh, arima, ArimaFit, PDQ

# AirPassengers data (48 values) for seasonal ARIMA tests
const AP_LONG = Float64[
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
]

# ── PDQ validation ──────────────────────────────────────────────────────────
@testset "PDQ struct - valid construction" begin
    @test PDQ(1, 0, 0) isa PDQ
    @test PDQ(0, 0, 0) isa PDQ
    @test PDQ(2, 1, 2) isa PDQ
end

@testset "PDQ struct - invalid (negative) arguments" begin
    @test_throws ArgumentError PDQ(-1, 0, 0)
    @test_throws ArgumentError PDQ(0, -1, 0)
    @test_throws ArgumentError PDQ(0, 0, -1)
end

# ── Basic arima() fitting ───────────────────────────────────────────────────
@testset "arima() - non-seasonal AR(1,0,0)" begin
    Random.seed!(42)
    y = randn(100)
    fit = arima(y, 1; order=PDQ(1,0,0))

    @test fit isa ArimaFit
    @test "ar1" in fit.coef.colnames
    @test isfinite(fit.sigma2)
    @test isfinite(fit.loglik)
    @test isfinite(fit.aic)
    @test occursin("ARIMA", fit.method)
end

@testset "arima() - ARMA(1,0,1)" begin
    Random.seed!(43)
    y = randn(100)
    fit = arima(y, 1; order=PDQ(1,0,1))

    @test fit isa ArimaFit
    @test "ar1" in fit.coef.colnames
    @test "ma1" in fit.coef.colnames
end

@testset "arima() - ARIMA(0,1,1) with differencing" begin
    Random.seed!(44)
    y = cumsum(randn(100))
    fit = arima(y, 1; order=PDQ(0,1,1))

    @test fit isa ArimaFit
    @test fit.n_cond >= 0
    @test length(fit.fitted) == length(y)
    @test length(fit.residuals) == length(y)
end

@testset "arima() - seasonal ARIMA(1,0,0)(1,0,0)[12]" begin
    fit = arima(AP_LONG, 12; order=PDQ(1,0,0), seasonal=PDQ(1,0,0))
    @test fit isa ArimaFit
    @test any(c -> contains(c, "ar"), fit.coef.colnames)
end

@testset "ArimaFit struct fields" begin
    Random.seed!(45)
    y = randn(80)
    fit = arima(y, 1; order=PDQ(1,0,0))

    @test length(fit.arma) == 7
    @test fit.nobs == length(y) || fit.nobs <= length(y)
    @test fit.convergence_code isa Bool
end

# ── Estimation methods ──────────────────────────────────────────────────────
@testset "arima() - CSS-ML method (default)" begin
    Random.seed!(50)
    y = randn(80)
    fit = arima(y, 1; order=PDQ(1,0,0), method="CSS-ML")
    @test occursin("ARIMA", fit.method)
    @test isfinite(fit.aic)
end

@testset "arima() - ML method" begin
    Random.seed!(51)
    y = randn(80)
    fit = arima(y, 1; order=PDQ(1,0,0), method="ML")
    @test occursin("ARIMA", fit.method)
    @test isfinite(fit.aic)
end

@testset "arima() - CSS method" begin
    Random.seed!(52)
    y = randn(80)
    fit = arima(y, 1; order=PDQ(1,0,0), method="CSS")
    @test occursin("ARIMA", fit.method)
    @test fit.aic === nothing
end

# ── arima_rjh() wrapper ────────────────────────────────────────────────────
@testset "arima_rjh() - basic fit" begin
    Random.seed!(60)
    y = randn(100)
    fit = arima_rjh(y, 1; order=PDQ(1,0,0))
    @test fit isa ArimaFit
    @test isfinite(fit.sigma2)
end

@testset "arima_rjh() - include_drift with d=1" begin
    Random.seed!(61)
    y = cumsum(randn(100)) .+ collect(1:100) * 0.1
    fit = arima_rjh(y, 1; order=PDQ(0,1,0), include_drift=true)
    @test fit isa ArimaFit
    @test "drift" in fit.coef.colnames
end

@testset "arima_rjh() - lambda Box-Cox" begin
    Random.seed!(62)
    y = exp.(randn(100))  # positive data for Box-Cox
    fit = arima_rjh(y, 1; order=PDQ(1,0,0), lambda=0.0)
    @test fit.lambda == 0.0
end

@testset "arima_rjh() - model refit" begin
    Random.seed!(63)
    y1 = randn(100)
    fit1 = arima_rjh(y1, 1; order=PDQ(1,0,0))

    Random.seed!(64)
    y2 = randn(100)
    # Known issue: refit path has a BoundsError with fixed parameter indexing
    @test_broken (arima_rjh(y2, 1; model=fit1); true)
end

# ── forecast() options ──────────────────────────────────────────────────────
@testset "forecast(ArimaFit) - basic" begin
    Random.seed!(70)
    y = randn(100)
    fit = arima_rjh(y, 1; order=PDQ(1,0,0))
    fc = forecast(fit; h=10)

    @test fc isa Forecast
    @test length(fc.mean) == 10
    @test all(isfinite, fc.mean)
end

@testset "forecast(ArimaFit) - custom levels" begin
    Random.seed!(71)
    y = randn(100)
    fit = arima_rjh(y, 1; order=PDQ(1,0,0))
    fc = forecast(fit; h=5, level=[90, 99])

    @test fc.level == [90, 99]
    # upper/lower are h×L matrices (5×2)
    @test size(fc.upper, 2) == 2
    @test size(fc.lower, 2) == 2
end

@testset "forecast(ArimaFit) - fan=true" begin
    Random.seed!(72)
    y = randn(100)
    fit = arima_rjh(y, 1; order=PDQ(1,0,0))
    fc = forecast(fit; h=5, fan=true)

    @test length(fc.level) > 2
end

@testset "forecast(ArimaFit) - bootstrap=true" begin
    Random.seed!(73)
    y = randn(100)
    fit = arima_rjh(y, 1; order=PDQ(1,0,0))
    # Known issue: bootstrap forecast requires simulate() which is not yet implemented
    @test_broken (forecast(fit; h=5, bootstrap=true, npaths=500); true)
end

# ── fitted() and residuals() generics ──────────────────────────────────────
@testset "fitted() and residuals() on ArimaFit" begin
    Random.seed!(80)
    y = randn(80)
    fit = arima_rjh(y, 1; order=PDQ(1,0,0))

    f = fitted(fit)
    r = residuals(fit)
    @test length(f) == length(y)
    @test length(r) == length(y)
    @test f isa AbstractVector
    @test r isa AbstractVector
end

# ── Existing tests (preserved verbatim) ────────────────────────────────────
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
