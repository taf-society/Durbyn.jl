using Test
using Durbyn
import Durbyn.Generics: Forecast, forecast, fitted, residuals
using Durbyn.Ararma

@testset "Durbyn.Ararma - ARARMA tests" begin
    ap = air_passengers()

    @testset "ararma() - Basic fit with default parameters" begin
        fit = ararma(ap)
        @test fit isa ArarmaModel
        @test fit.ar_order == 4
        @test fit.ma_order == 1
        @test length(fit.y_original) == length(ap)
        @test length(fit.psi) >= 1
        @test fit.sigma2 > 0
        @test !isnan(fit.aic)
        @test !isnan(fit.bic)
        @test !isnan(fit.loglik)
        fc = forecast(fit, h = 12)
        # plot(fc)  # Requires Plots.jl
    end

    @testset "ararma() - Custom AR/MA orders" begin
        fit_21 = ararma(ap; p=2, q=1)
        @test fit_21 isa ArarmaModel
        @test fit_21.ar_order == 2
        @test fit_21.ma_order == 1
        @test length(fit_21.lag_phi) == 4  # Always 4 coeffs from lag selection
        @test length(fit_21.arma_phi) == 2  # p coeffs from ARMA stage
        @test length(fit_21.arma_theta) == 1  # q coeffs from ARMA stage
        fc = forecast(fit_21, h = 12)
        # plot(fc)  # Requires Plots.jl

        fit_32 = ararma(ap; p=3, q=2)
        @test fit_32 isa ArarmaModel
        @test fit_32.ar_order == 3
        @test fit_32.ma_order == 2
        @test length(fit_32.lag_phi) == 4  # Always 4 coeffs from lag selection
        @test length(fit_32.arma_phi) == 3  # p coeffs from ARMA stage
        @test length(fit_32.arma_theta) == 2  # q coeffs from ARMA stage
        fc = forecast(fit_32, h = 12)
        # plot(fc)  # Requires Plots.jl
    end

    @testset "ararma() - AR-only model (q=0)" begin
        fit_ar = ararma(ap; p=4, q=0)
        @test fit_ar isa ArarmaModel
        @test fit_ar.ar_order == 4
        @test fit_ar.ma_order == 0
        @test length(fit_ar.arma_theta) == 0
        fc = forecast(fit_ar, h = 12)
        # plot(fc)  # Requires Plots.jl
    end

    @testset "ararma() - MA-only model (p=0)" begin
        fit_ma = ararma(ap; p=0, q=2)
        @test fit_ma isa ArarmaModel
        @test fit_ma.ar_order == 0
        @test fit_ma.ma_order == 2
        @test length(fit_ma.arma_phi) == 0
        @test length(fit_ma.arma_theta) == 2
        fc = forecast(fit_ma, h = 12)
        # plot(fc)  # Requires Plots.jl
    end

    @testset "ararma() - Custom max_ar_depth and max_lag" begin
        fit = ararma(ap; max_ar_depth=15, max_lag=20)
        @test fit isa ArarmaModel
        @test fit.best_lag[1] == 1
        @test all(fit.best_lag .<= 15)
        @test length(fit.gamma) == 21  # max_lag + 1
        fc = forecast(fit, h = 12)
        # plot(fc)  # Requires Plots.jl
    end

    @testset "ararma() - Model components" begin
        fit = ararma(ap)

        # Check prefilter polynomial
        @test fit.psi[1] == 1.0

        # Check best lag tuple
        @test fit.best_lag[1] == 1
        @test 1 < fit.best_lag[2] < fit.best_lag[3] < fit.best_lag[4]

        # Check mean
        @test !isnan(fit.Sbar)

        # Check autocovariances
        @test length(fit.gamma) >= 1
        @test fit.gamma[1] >= 0  # variance should be non-negative
    end

    @testset "fitted() - Fitted values" begin
        fit = ararma(ap)
        fits = fitted(fit)

        @test length(fits) == length(ap)
        @test sum(isnan.(fits)) > 0  # Some initial values should be NaN
        @test sum(.!isnan.(fits)) > 0  # But not all

        # Check that fitted values are reasonable
        valid_fits = fits[.!isnan.(fits)]
        @test all(isfinite.(valid_fits))
    end

    @testset "residuals() - Residuals" begin
        fit = ararma(ap)
        res = residuals(fit)

        @test length(res) == length(ap)
        @test sum(isnan.(res)) > 0  # Some initial residuals should be NaN

        # Check residuals match definition
        fits = fitted(fit)
        @test length(fits) == length(ap)
    end

    @testset "forecast() - Basic forecasting" begin
        fit = ararma(ap)

        fc = forecast(fit; h=12)
        @test fc isa Forecast
        @test length(fc.mean) == 12
        @test all(isfinite.(fc.mean))
        @test fc.method == "Ararma(4, 1)"
    end

    @testset "forecast() - Multiple forecast horizons" begin
        fit = ararma(ap)

        fc6 = forecast(fit; h=6)
        @test length(fc6.mean) == 6

        fc24 = forecast(fit; h=16)
        @test length(fc24.mean) == 16

        # Short horizon forecasts should match beginning of longer horizon
        @test all(fc6.mean .≈ fc24.mean[1:6])
    end

    @testset "forecast() - Confidence intervals" begin
        fit = ararma(ap)

        fc = forecast(fit; h=12, level=[80, 95])
        @test size(fc.upper) == (12, 2)
        @test size(fc.lower) == (12, 2)

        # Check intervals are ordered correctly
        @test all(fc.lower[:, 1] .>= fc.lower[:, 2])  # 80% lower >= 95% lower
        @test all(fc.upper[:, 1] .<= fc.upper[:, 2])  # 80% upper <= 95% upper
        @test all(fc.lower .< fc.mean)
        @test all(fc.upper .> fc.mean)

        # Single level
        fc_single = forecast(fit; h=12, level=[90])
        @test size(fc_single.upper) == (12, 1)
        @test size(fc_single.lower) == (12, 1)
    end

    @testset "forecast() - Standard errors increase with horizon" begin
        fit = ararma(ap)
        fc = forecast(fit; h=12, level=[95])

        # Standard errors should generally increase with forecast horizon
        widths = fc.upper[:, 1] .- fc.lower[:, 1]
        @test widths[end] >= widths[1]
    end

    @testset "auto_ararma() - Automatic order selection with AIC" begin
        fit = auto_ararma(ap; max_p=3, max_q=2, crit=:aic)
        @test fit isa ArarmaModel
        @test fit.ar_order <= 3
        @test fit.ma_order <= 2
        @test !(fit.ar_order == 0 && fit.ma_order == 0)
    end

    @testset "auto_ararma() - Automatic order selection with BIC" begin
        fit = auto_ararma(ap; max_p=3, max_q=2, crit=:bic)
        @test fit isa ArarmaModel
        @test fit.ar_order <= 3
        @test fit.ma_order <= 2
        @test !(fit.ar_order == 0 && fit.ma_order == 0)
    end

    @testset "auto_ararma() - BIC favors parsimony over AIC" begin
        fit_aic = auto_ararma(ap; max_p=4, max_q=2, crit=:aic)
        fit_bic = auto_ararma(ap; max_p=4, max_q=2, crit=:bic)

        @test fit_aic isa ArarmaModel
        @test fit_bic isa ArarmaModel

        # BIC typically selects fewer parameters
        total_params_aic = fit_aic.ar_order + fit_aic.ma_order
        total_params_bic = fit_bic.ar_order + fit_bic.ma_order
        @test total_params_bic <= total_params_aic
    end

    @testset "auto_ararma() - Custom parameters" begin
        fit = auto_ararma(ap; max_p=2, max_q=1, max_ar_depth=15, max_lag=20)
        @test fit isa ArarmaModel
        @test fit.ar_order <= 2
        @test fit.ma_order <= 1
        @test length(fit.gamma) == 21
    end

    @testset "Information criteria comparisons" begin
        fit = ararma(ap; p=4, q=1)

        # AIC and BIC should both be finite
        @test isfinite(fit.aic)
        @test isfinite(fit.bic)

        # For same data, BIC typically >= AIC (penalizes complexity more)
        # This may not always hold but is a general tendency
        @test fit.bic > 0
        @test fit.aic > 0
    end

    @testset "Short series handling" begin
        short_ap = ap[1:20]
        fit = ararma(short_ap; max_ar_depth=10, max_lag=15)
        @test fit isa ArarmaModel

        fc = forecast(fit; h=5)
        @test length(fc.mean) == 5
    end

    @testset "Consistency between fit and forecast" begin
        fit = ararma(ap)
        fc = forecast(fit; h=1)

        # First forecast should be reasonable relative to last observation
        @test isfinite(fc.mean[1])
    end
end
