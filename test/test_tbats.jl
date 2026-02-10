using Test
using Durbyn
using Durbyn.Tbats: tbats, TBATSModel
import Durbyn.Generics: Forecast, forecast, fitted

@testset "TBATS with air_passengers (default)" begin
    ap = air_passengers()

    fit = tbats(ap, 12)
    @test fit isa TBATSModel
    @test length(fit.fitted_values) == 144
    @test length(fit.errors) == 144
    @test all(isfinite, fit.fitted_values)
    @test all(isfinite, fit.errors)
    @test isfinite(fit.AIC)
    @test fit.seasonal_periods == [12]

    fitted_view = fitted(fit)
    @test fitted_view === fit.fitted_values

    fc = forecast(fit; h=12, level=[80, 95])
    @test fc isa Forecast
    @test length(fc.mean) == 12
    @test size(fc.upper) == (12, 2)
    @test size(fc.lower) == (12, 2)
    @test all(isfinite, fc.mean)
    @test all(isfinite, vec(fc.upper))
    @test all(isfinite, vec(fc.lower))

    # Confidence interval ordering
    @test all(fc.lower[:, 1] .< fc.mean)
    @test all(fc.mean .< fc.upper[:, 1])
    @test all(fc.lower[:, 2] .<= fc.lower[:, 1])
    @test all(fc.upper[:, 1] .<= fc.upper[:, 2])
end

@testset "TBATS no Box-Cox, no ARMA" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=false, use_arma_errors=false)
    @test fit isa TBATSModel
    @test isnothing(fit.lambda)
    @test isnothing(fit.ar_coefficients)
    @test isnothing(fit.ma_coefficients)
    @test fit.seasonal_periods == [12]
    @test !isnothing(fit.k_vector)
    @test all(isfinite, fit.fitted_values)
    @test isfinite(fit.AIC)

    fc = forecast(fit; h=12)
    @test length(fc.mean) == 12
    @test all(isfinite, fc.mean)
end

@testset "TBATS with Box-Cox, no ARMA" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=true, use_arma_errors=false)
    @test fit isa TBATSModel
    @test !isnothing(fit.lambda)
    @test isfinite(fit.lambda)
    @test all(isfinite, fit.fitted_values)
    @test isfinite(fit.AIC)

    fc = forecast(fit; h=12)
    @test length(fc.mean) == 12
    @test all(isfinite, fc.mean)
end

@testset "TBATS with ARMA errors, no Box-Cox" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=false, use_arma_errors=true)
    @test fit isa TBATSModel
    @test isnothing(fit.lambda)
    @test all(isfinite, fit.fitted_values)
    @test isfinite(fit.AIC)

    fc = forecast(fit; h=12)
    @test length(fc.mean) == 12
    @test all(isfinite, fc.mean)
end

@testset "TBATS with damped trend" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_damped_trend=true, use_box_cox=false, use_arma_errors=false)
    @test fit isa TBATSModel
    @test all(isfinite, fit.fitted_values)

    fc = forecast(fit; h=24)
    @test length(fc.mean) == 24
    @test all(isfinite, fc.mean)
end

@testset "TBATS refit with model kwarg" begin
    ap = air_passengers()
    fit1 = tbats(ap[1:120], 12; use_arma_errors=false)
    fit2 = tbats(ap, 12; model=fit1)

    @test fit2 isa TBATSModel
    @test length(fit2.fitted_values) == 144
    @test all(isfinite, fit2.fitted_values)
    @test isfinite(fit2.variance)

    # Parameters frozen from old model
    @test fit2.alpha == fit1.alpha
    @test fit2.lambda == fit1.lambda
    @test fit2.seasonal_periods == fit1.seasonal_periods
    @test fit2.k_vector == fit1.k_vector

    # Forecasting works
    fc = forecast(fit2; h=12)
    @test length(fc.mean) == 12
    @test all(isfinite, fc.mean)
end

@testset "TBATS refit with Box-Cox model" begin
    ap = air_passengers()
    fit1 = tbats(ap[1:120], 12; use_box_cox=true, use_arma_errors=false)
    fit2 = tbats(ap, 12; model=fit1)

    @test fit2 isa TBATSModel
    @test fit2.lambda == fit1.lambda
    @test length(fit2.fitted_values) == 144
    @test all(isfinite, fit2.fitted_values)

    fc = forecast(fit2; h=12)
    @test all(isfinite, fc.mean)
end

@testset "TBATS warns on Box-Cox with non-positive data" begin
    data = [-1.0; collect(1.0:99.0)]
    @test_logs (:warn, r"non-positive") match_mode=:any tbats(data; use_box_cox=true)
end

@testset "TBATS nonseasonal" begin
    ap = air_passengers()
    # tbats() with no seasonal period may return BATSModel if nonseasonal wins
    fit = tbats(ap)
    @test fit isa Union{TBATSModel, Durbyn.Bats.BATSModel}
    fc = forecast(fit; h=10)
    @test length(fc.mean) == 10
    @test all(isfinite, fc.mean)
end

@testset "TBATS with non-integer seasonal period" begin
    # Simulate weekly data with yearly seasonality (52.18 weeks/year)
    n = 200
    t = 1:n
    y = 100.0 .+ 10.0 .* sin.(2π .* t ./ 52.18) .+ randn(n)

    fit = tbats(y, 52.18; use_box_cox=false, use_arma_errors=false)
    @test fit isa TBATSModel
    @test fit.seasonal_periods == [52.18]
    @test all(isfinite, fit.fitted_values)
    @test isfinite(fit.AIC)

    fc = forecast(fit; h=26)
    @test length(fc.mean) == 26
    @test all(isfinite, fc.mean)
end

@testset "TBATS with multiple seasonal periods" begin
    # Daily data with weekly + yearly seasonality
    n = 400
    t = 1:n
    y = 100.0 .+ 5.0 .* sin.(2π .* t ./ 7) .+ 10.0 .* sin.(2π .* t ./ 365.25) .+ randn(n)

    fit = tbats(y, [7.0, 365.25]; use_box_cox=false, use_arma_errors=false)
    @test fit isa TBATSModel
    @test fit.seasonal_periods == [7.0, 365.25]
    @test length(fit.k_vector) == 2
    @test all(isfinite, fit.fitted_values)

    fc = forecast(fit; h=14)
    @test length(fc.mean) == 14
    @test all(isfinite, fc.mean)
end

@testset "TBATS with user-specified k" begin
    ap = air_passengers()

    # Scalar k applied to single period
    fit = tbats(ap, 12; k=3, use_box_cox=false, use_arma_errors=false)
    @test fit isa TBATSModel
    @test fit.k_vector == [3]
    @test all(isfinite, fit.fitted_values)

    # Vector k for multiple periods
    n = 400
    t = 1:n
    y = 100.0 .+ 5.0 .* sin.(2π .* t ./ 7) .+ 10.0 .* sin.(2π .* t ./ 365.25) .+ randn(n)
    fit2 = tbats(y, [7.0, 365.25]; k=[2, 5], use_box_cox=false, use_arma_errors=false)
    @test fit2 isa TBATSModel
    @test fit2.k_vector == [2, 5]

    # Invalid k: exceeds max Fourier order
    @test_throws ArgumentError tbats(ap, 12; k=10)

    # Invalid k: wrong length
    @test_throws ArgumentError tbats(y, [7.0, 365.25]; k=[2])

    # Invalid k: zero
    @test_throws ArgumentError tbats(ap, 12; k=0)
end

@testset "TBATS forecast with fan" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=false, use_arma_errors=false)
    fc = forecast(fit; h=12, fan=true)
    @test fc isa Forecast
    @test length(fc.mean) == 12
    @test size(fc.upper, 2) == 17
    @test size(fc.lower, 2) == 17
end

@testset "TBATS default forecast horizon" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=false, use_arma_errors=false)

    # Default h should be 2 * max(seasonal_periods) = 24
    fc = forecast(fit)
    @test length(fc.mean) == 24
end

@testset "TBATS model descriptor" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=false, use_arma_errors=false)
    desc = fit.method
    @test occursin("TBATS", desc)
    @test occursin("<12", desc) || occursin("<12.0", desc)

    # Non-integer period shows in descriptor when seasonal model wins
    n = 400
    t = 1:n
    y = 100.0 .+ 10.0 .* sin.(2π .* t ./ 52.18) .+ randn(n)
    fit2 = tbats(y, 52.18; use_box_cox=false, use_arma_errors=false)
    @test occursin("52.18", fit2.method)
end

@testset "TBATS with short data" begin
    # Short data may cause tbats to fall back to BATSModel (nonseasonal wins)
    y = collect(1.0:30.0)
    fit = tbats(y, 12; use_box_cox=false, use_arma_errors=false)
    @test fit isa Union{TBATSModel, Durbyn.Bats.BATSModel}
    fc = forecast(fit; h=5)
    @test length(fc.mean) == 5
end
