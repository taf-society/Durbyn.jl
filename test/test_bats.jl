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

@testset "BATS parity: no Box-Cox, no trend, no ARMA" begin
    ap = air_passengers()
    fit = bats(ap, 12; use_box_cox=false, use_trend=false, use_arma_errors=false)

    @test fit.alpha ≈ 1.204409639474027 rtol=1e-6
    @test fit.gamma_values[1] ≈ -0.020768738557064 rtol=1e-4
    @test fit.AIC ≈ 1555.678560109123509 rtol=1e-6
    @test fit.variance ≈ 277.321283192424005 rtol=1e-6
    @test isnothing(fit.beta)
    @test isnothing(fit.damping_parameter)

    expected_fc = [438.498411519184060, 430.923054886217926, 464.069321947585536,
                   457.989419800910412, 461.042801288024521, 498.574465570803341,
                   535.795117729898948, 532.894378123716365, 482.054231690987251,
                   444.890729435936294, 408.045106600487713, 435.678221626298637,
                   438.498411519184060, 430.923054886217926, 464.069321947585536,
                   457.989419800910412, 461.042801288024521, 498.574465570803341,
                   535.795117729898948, 532.894378123716365, 482.054231690987251,
                   444.890729435936294, 408.045106600487713, 435.678221626298637]
    fc = forecast(fit; h=24)
    @test fc.mean ≈ expected_fc rtol=1e-6
end

@testset "BATS parity: no Box-Cox, trend + damped, no ARMA" begin
    ap = air_passengers()
    fit = bats(ap, 12; use_box_cox=false, use_trend=true, use_damped_trend=true, use_arma_errors=false)

    @test fit.alpha ≈ 1.306915464360444 rtol=1e-6
    @test fit.beta ≈ -0.126848346969942 rtol=1e-4
    @test fit.damping_parameter ≈ 0.800000073472066 rtol=1e-4
    @test fit.gamma_values[1] ≈ -0.027656243695200 rtol=1e-4
    @test fit.AIC ≈ 1560.057516483699828 rtol=1e-6
    @test fit.variance ≈ 274.216915002459075 rtol=1e-6

    expected_fc = [445.698590387200227, 436.793284276621648, 470.135368260069697,
                   462.800108206447248, 466.105811444294886, 502.816295091337054,
                   539.552713014055712, 535.786155143030896, 484.477320483752635,
                   447.366685466786805, 409.117776747840026, 437.171373006457713,
                   445.432680027271942, 436.580555969142040, 469.965185598456458,
                   462.663962064652992, 465.996894520856529, 502.729161544584031,
                   539.483006170251429, 535.730389662865946, 484.432708095523424,
                   447.330995552925629, 409.089224814128841, 437.148531457390959]
    fc = forecast(fit; h=24)
    @test fc.mean ≈ expected_fc rtol=1e-6
end

@testset "BATS parity: Box-Cox, trend + damped, no ARMA" begin
    ap = air_passengers()
    fit = bats(ap, 12; use_box_cox=true, use_trend=true, use_damped_trend=true, use_arma_errors=false)

    @test !isnothing(fit.lambda)
    @test fit.lambda ≈ 0.001452554850448 rtol=0.1
    @test fit.alpha ≈ 0.843284820906740 rtol=0.01
    @test fit.beta ≈ 0.033641672890850 rtol=0.01
    @test fit.damping_parameter ≈ 0.994477807394488 rtol=0.001
    @test fit.AIC ≈ 1402.161640106883851 rtol=0.001

    expected_fc = [444.648315793595771, 437.035121749424206, 505.556675564460079,
                   492.738245325743833, 493.725132247300508, 558.995470149387074,
                   625.081622161088262, 622.629497573427329, 542.220960107442693,
                   479.906742716094016, 416.525378694914195, 467.863769496029875]
    fc = forecast(fit; h=12)
    @test fc.mean ≈ expected_fc rtol=0.01
end
