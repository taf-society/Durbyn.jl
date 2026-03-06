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
    @test !isnothing(fit.aic)

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
    @test !isnothing(fit.aic)

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
    @test !isnothing(fit.aic)

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
    @test !isnothing(fit.aic)

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

    @test fit.alpha ≈ 1.2046239081449244 rtol=1e-6
    @test fit.gamma_values[1] ≈ -0.020739740472257875 rtol=1e-4
    @test fit.aic ≈ 1555.6785579735697 rtol=1e-6
    @test fit.variance ≈ 277.321279079684 rtol=1e-6
    @test isnothing(fit.beta)
    @test isnothing(fit.damping_parameter)

    expected_fc = [438.50893142394364, 430.9315157962287, 464.0763787448749,
                   457.9988458283522, 461.0507945424869, 498.5824444505813,
                   535.8037027341932, 532.9041101847045, 482.0638426042857,
                   444.89762721793767, 408.05428707778117, 435.6852042981058,
                   438.50893142394364, 430.9315157962287, 464.0763787448749,
                   457.9988458283522, 461.0507945424869, 498.5824444505813,
                   535.8037027341932, 532.9041101847045, 482.0638426042857,
                   444.89762721793767, 408.05428707778117, 435.6852042981058]
    fc = forecast(fit; h=24)
    @test fc.mean ≈ expected_fc rtol=1e-6
end

@testset "BATS parity: no Box-Cox, trend + damped, no ARMA" begin
    ap = air_passengers()
    fit = bats(ap, 12; use_box_cox=false, use_trend=true, use_damped_trend=true, use_arma_errors=false)

    @test fit.alpha ≈ 1.267243183113559 rtol=1e-6
    @test fit.beta ≈ -0.10165490374570549 rtol=1e-4
    @test fit.damping_parameter ≈ 0.8000052043547048 rtol=1e-4
    @test fit.gamma_values[1] ≈ -0.030383646970758817 rtol=1e-4
    @test fit.aic ≈ 1560.270958302632 rtol=1e-6
    @test fit.variance ≈ 274.62367025028016 rtol=1e-6

    expected_fc = [444.55770034010226, 435.875170769237, 469.6748933683165,
                   462.13038353395217, 465.76281031273476, 502.5747931400765,
                   539.286269044737, 535.4345214509591, 484.1817954652309,
                   447.49213323288876, 408.9525243477131, 437.3292924664919,
                   445.3598482757269, 436.5168932923991, 470.1882747265978,
                   462.5410912923959, 466.09137865695857, 502.83764952544175,
                   539.4965555210271, 535.6027517263966, 484.3163805611109,
                   447.59980201002134, 409.0386599297657, 437.3982013804141]
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
    @test fit.aic ≈ 1402.161640106883851 rtol=0.001

    expected_fc = [444.648315793595771, 437.035121749424206, 505.556675564460079,
                   492.738245325743833, 493.725132247300508, 558.995470149387074,
                   625.081622161088262, 622.629497573427329, 542.220960107442693,
                   479.906742716094016, 416.525378694914195, 467.863769496029875]
    fc = forecast(fit; h=12)
    @test fc.mean ≈ expected_fc rtol=0.01
end
