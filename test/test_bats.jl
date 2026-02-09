using Test
using Durbyn
using Durbyn.Bats: bats, BATSModel
import Durbyn.Generics: Forecast, forecast, fitted

@testset "BATS with air_passengers" begin
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
end
