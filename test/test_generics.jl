using Test
using Durbyn
using Random
import Durbyn.Generics: Forecast, forecast, fitted, residuals
import Durbyn.Arima: arima_rjh, ArimaFit, PDQ
import Durbyn.Naive: naive, NaiveFit

# ── Forecast struct construction ────────────────────────────────────────────
@testset "Forecast struct - manual construction" begin
    fc = Forecast(
        nothing,           # model
        "Test Method",     # method
        [1.0, 2.0, 3.0],  # mean
        [80, 95],          # level
        [10.0, 20.0],      # x
        [[1.5, 2.5, 3.5], [2.0, 3.0, 4.0]],  # upper
        [[0.5, 1.5, 2.5], [0.0, 1.0, 2.0]],  # lower
        [9.5, 19.5],       # fitted
        [0.5, 0.5],        # residuals
    )

    @test fc isa Forecast
    @test fc.method == "Test Method"
    @test fc.mean == [1.0, 2.0, 3.0]
    @test fc.level == [80, 95]
    @test fc.x == [10.0, 20.0]
    @test length(fc.upper) == 2
    @test length(fc.lower) == 2
    @test fc.fitted == [9.5, 19.5]
    @test fc.residuals == [0.5, 0.5]
    @test fc.model === nothing
end

@testset "Forecast struct - show method" begin
    fc = Forecast(
        nothing,
        "TestShow",
        [1.0, 2.0, 3.0],
        [80, 95],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        [[1.5, 2.5, 3.5], [2.0, 3.0, 4.0]],
        [[0.5, 1.5, 2.5], [0.0, 1.0, 2.0]],
        [9.5, 19.5, 29.5, 39.5, 49.5, 59.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )

    buf = IOBuffer()
    show(buf, fc)
    output = String(take!(buf))

    @test occursin("TestShow", output)
    @test occursin("3 steps", output)
end

# ── Forecast from real models ───────────────────────────────────────────────
@testset "Forecast from naive model" begin
    Random.seed!(100)
    y = randn(50)
    fit = naive(y)
    fc = forecast(fit; h=10)

    @test fc isa Forecast
    @test length(fc.mean) == 10
    @test fc.level == [80, 95]
    @test length(fc.upper) == 2
    @test length(fc.lower) == 2
    @test length(fc.fitted) == length(y)
    @test length(fc.residuals) == length(y)
    @test length(fc.x) == length(y)
    @test !isempty(fc.method)
end

@testset "Forecast from arima_rjh model" begin
    Random.seed!(101)
    y = randn(100)
    fit = arima_rjh(y, 1; order=PDQ(1,0,0))
    fc = forecast(fit; h=12)

    @test fc isa Forecast
    @test length(fc.mean) == 12
    @test all(isfinite, fc.mean)
    @test length(fc.upper) >= 1
    @test length(fc.lower) >= 1
    @test length(fc.fitted) == length(y)
    @test length(fc.residuals) == length(y)
    @test !isempty(fc.method)
end

# ── fitted() and residuals() dispatch ───────────────────────────────────────
@testset "fitted() and residuals() on ArimaFit" begin
    Random.seed!(110)
    y = randn(80)
    fit = arima_rjh(y, 1; order=PDQ(1,0,0))

    f = fitted(fit)
    r = residuals(fit)
    @test f isa AbstractVector
    @test r isa AbstractVector
    @test length(f) == length(y)
    @test length(r) == length(y)
end

@testset "NaiveFit direct field access" begin
    Random.seed!(111)
    y = randn(50)
    fit = naive(y)

    # NaiveFit stores fitted and residuals as fields directly
    @test length(fit.fitted) == length(y)
    @test length(fit.residuals) == length(y)
    @test fit.fitted isa AbstractVector
    @test fit.residuals isa AbstractVector
end

@testset "Forecast from naive - fitted and residuals propagate" begin
    Random.seed!(112)
    y = randn(50)
    fit = naive(y)
    fc = forecast(fit; h=5)

    # Forecast struct should carry forward fitted/residuals from the model
    @test length(fc.fitted) == length(y)
    @test length(fc.residuals) == length(y)
end
