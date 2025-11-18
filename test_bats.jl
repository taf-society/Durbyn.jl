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
