using Test
using Durbyn
using Durbyn.Bats: bats, BATSModel
import Durbyn.Generics: Forecast, forecast, fitted

@testset "BATS with air_passengers (default)" begin
    ap = air_passengers()
    @test length(ap) == 144
    @test all(isfinite, ap)

    fit = bats(ap, 12)
    @test fit isa BATSModel
    @test length(fit.fitted_values) == length(ap)
    @test length(fit.errors) == length(ap)
    @test all(isfinite, fit.fitted_values)
    @test all(isfinite, fit.errors)
    @test isfinite(fit.AIC)

    fitted_view = fitted(fit)
    @test fitted_view === fit.fitted_values

    fc = forecast(fit; h = 12, level = [80, 95])
    @test fc isa Forecast
    @test length(fc.mean) == 12
    @test size(fc.upper) == (12, 2)
    @test size(fc.lower) == (12, 2)
    @test all(isfinite, fc.mean)
    @test all(isfinite, vec(fc.upper))
    @test all(isfinite, vec(fc.lower))

    # Confidence interval ordering: lower < mean < upper
    @test all(fc.lower[:, 1] .< fc.mean)
    @test all(fc.mean .< fc.upper[:, 1])
    # Wider interval contains narrower: 80 inside 95
    @test all(fc.lower[:, 2] .<= fc.lower[:, 1])
    @test all(fc.upper[:, 1] .<= fc.upper[:, 2])
end

@testset "BATS no Box-Cox, no ARMA" begin
    ap = air_passengers()
    fit = bats(ap, 12; use_box_cox=false, use_arma_errors=false)
    @test fit isa BATSModel
    @test isnothing(fit.lambda)
    @test isnothing(fit.ar_coefficients)
    @test isnothing(fit.ma_coefficients)
    @test fit.seasonal_periods == [12]
    @test all(isfinite, fit.fitted_values)
    @test isfinite(fit.AIC)

    fc = forecast(fit; h=12)
    @test length(fc.mean) == 12
    @test all(isfinite, fc.mean)
end

@testset "BATS with Box-Cox, no ARMA" begin
    ap = air_passengers()
    fit = bats(ap, 12; use_box_cox=true, use_arma_errors=false)
    @test fit isa BATSModel
    @test !isnothing(fit.lambda)
    @test isfinite(fit.lambda)
    @test all(isfinite, fit.fitted_values)
    @test isfinite(fit.AIC)

    fc = forecast(fit; h=12)
    @test length(fc.mean) == 12
    @test all(isfinite, fc.mean)
end

@testset "BATS with ARMA errors, no Box-Cox" begin
    ap = air_passengers()
    fit = bats(ap, 12; use_box_cox=false, use_arma_errors=true)
    @test fit isa BATSModel
    @test isnothing(fit.lambda)
    @test all(isfinite, fit.fitted_values)
    @test isfinite(fit.AIC)

    fc = forecast(fit; h=12)
    @test length(fc.mean) == 12
    @test all(isfinite, fc.mean)
end

@testset "BATS with damped trend" begin
    ap = air_passengers()
    fit = bats(ap, 12; use_damped_trend=true, use_box_cox=false, use_arma_errors=false)
    @test fit isa BATSModel
    @test all(isfinite, fit.fitted_values)

    fc = forecast(fit; h=24)
    @test length(fc.mean) == 24
    @test all(isfinite, fc.mean)
end

@testset "BATS refit with model kwarg" begin
    ap = air_passengers()
    fit1 = bats(ap[1:120], 12; use_arma_errors=false)
    fit2 = bats(ap, 12; model=fit1)

    @test fit2 isa BATSModel
    @test length(fit2.fitted_values) == 144
    @test all(isfinite, fit2.fitted_values)
    @test isfinite(fit2.variance)

    # Parameters frozen from old model
    @test fit2.alpha == fit1.alpha
    @test fit2.lambda == fit1.lambda
    @test fit2.seasonal_periods == fit1.seasonal_periods

    # Forecasting works
    fc = forecast(fit2; h=12)
    @test length(fc.mean) == 12
    @test all(isfinite, fc.mean)
end

@testset "BATS refit with Box-Cox model" begin
    ap = air_passengers()
    fit1 = bats(ap[1:120], 12; use_box_cox=true, use_arma_errors=false)
    fit2 = bats(ap, 12; model=fit1)

    @test fit2 isa BATSModel
    @test fit2.lambda == fit1.lambda
    @test length(fit2.fitted_values) == 144
    @test all(isfinite, fit2.fitted_values)

    fc = forecast(fit2; h=12)
    @test all(isfinite, fc.mean)
end

@testset "BATS warns on Box-Cox with non-positive data" begin
    data = [-1.0; collect(1.0:99.0)]
    @test_logs (:warn, r"non-positive") match_mode=:any bats(data; use_box_cox=true)
end

@testset "BATS nonseasonal" begin
    ap = air_passengers()
    fit = bats(ap)
    @test fit isa BATSModel
    fc = forecast(fit; h=10)
    @test length(fc.mean) == 10
    @test all(isfinite, fc.mean)
end

@testset "BATS forecast with fan" begin
    ap = air_passengers()
    fit = bats(ap, 12; use_box_cox=false, use_arma_errors=false)
    fc = forecast(fit; h=12, fan=true)
    @test fc isa Forecast
    @test length(fc.mean) == 12
    # fan produces levels 51:3:99 → 17 levels
    @test size(fc.upper, 2) == 17
    @test size(fc.lower, 2) == 17
end

@testset "BATS with short data" begin
    # Short enough to trigger constant model path
    y = collect(1.0:30.0)
    fit = bats(y, 12; use_box_cox=false, use_arma_errors=false)
    @test fit isa BATSModel
    fc = forecast(fit; h=5)
    @test length(fc.mean) == 5
end
