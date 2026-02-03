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
    end

    @testset "ararma() - Custom AR/MA orders" begin
        fit_21 = ararma(ap; p=2, q=1)
        @test fit_21 isa ArarmaModel
        @test fit_21.ar_order == 2
        @test fit_21.ma_order == 1
        @test length(fit_21.lag_phi) == 4
        @test length(fit_21.arma_phi) == 2
        @test length(fit_21.arma_theta) == 1
        fc = forecast(fit_21, h = 12)

        fit_32 = ararma(ap; p=3, q=2)
        @test fit_32 isa ArarmaModel
        @test fit_32.ar_order == 3
        @test fit_32.ma_order == 2
        @test length(fit_32.lag_phi) == 4
        @test length(fit_32.arma_phi) == 3
        @test length(fit_32.arma_theta) == 2
        fc = forecast(fit_32, h = 12)
    end

    @testset "ararma() - AR-only model (q=0)" begin
        fit_ar = ararma(ap; p=4, q=0)
        @test fit_ar isa ArarmaModel
        @test fit_ar.ar_order == 4
        @test fit_ar.ma_order == 0
        @test length(fit_ar.arma_theta) == 0
        fc = forecast(fit_ar, h = 12)
    end

    @testset "ararma() - MA-only model (p=0)" begin
        fit_ma = ararma(ap; p=0, q=2)
        @test fit_ma isa ArarmaModel
        @test fit_ma.ar_order == 0
        @test fit_ma.ma_order == 2
        @test length(fit_ma.arma_phi) == 0
        @test length(fit_ma.arma_theta) == 2
        fc = forecast(fit_ma, h = 12)
    end

    @testset "ararma() - Custom max_ar_depth and max_lag" begin
        fit = ararma(ap; max_ar_depth=15, max_lag=20)
        @test fit isa ArarmaModel
        @test fit.best_lag[1] == 1
        @test all(fit.best_lag .<= 15)
        @test length(fit.gamma) == 21
        fc = forecast(fit, h = 12)
    end

    @testset "ararma() - Model components" begin
        fit = ararma(ap)

        @test fit.psi[1] == 1.0

        @test fit.best_lag[1] == 1
        @test 1 < fit.best_lag[2] < fit.best_lag[3] < fit.best_lag[4]

        @test !isnan(fit.Sbar)

        @test length(fit.gamma) >= 1
        @test fit.gamma[1] >= 0
    end

    @testset "fitted() - Fitted values" begin
        fit = ararma(ap)
        fits = fitted(fit)

        @test length(fits) == length(ap)
        @test sum(isnan.(fits)) > 0  # Some initial values should be NaN
        @test sum(.!isnan.(fits)) > 0

        valid_fits = fits[.!isnan.(fits)]
        @test all(isfinite.(valid_fits))
    end

    @testset "residuals() - Residuals" begin
        fit = ararma(ap)
        res = residuals(fit)

        @test length(res) == length(ap)
        @test sum(isnan.(res)) > 0

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

        @test all(fc6.mean .≈ fc24.mean[1:6])
    end

    @testset "forecast() - Confidence intervals" begin
        fit = ararma(ap)

        fc = forecast(fit; h=12, level=[80, 95])
        @test size(fc.upper) == (12, 2)
        @test size(fc.lower) == (12, 2)

        @test all(fc.lower[:, 1] .>= fc.lower[:, 2])
        @test all(fc.upper[:, 1] .<= fc.upper[:, 2])
        @test all(fc.lower .< fc.mean)
        @test all(fc.upper .> fc.mean)

        fc_single = forecast(fit; h=12, level=[90])
        @test size(fc_single.upper) == (12, 1)
        @test size(fc_single.lower) == (12, 1)
    end

    @testset "forecast() - Standard errors increase with horizon" begin
        fit = ararma(ap)
        fc = forecast(fit; h=12, level=[95])

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

        @test isfinite(fit.aic)
        @test isfinite(fit.bic)

        @test fit.bic > 0
        @test fit.aic > 0
    end

    @testset "Short series handling" begin
        # Use stationary random data to avoid aggressive prefiltering
        using Random
        Random.seed!(42)
        short_series = randn(30)
        # Use smaller p/q for short series to ensure enough residuals
        fit = ararma(short_series; max_ar_depth=8, max_lag=12, p=2, q=1)
        @test fit isa ArarmaModel

        fc = forecast(fit; h=5)
        @test length(fc.mean) == 5
    end

    @testset "Consistency between fit and forecast" begin
        fit = ararma(ap)
        fc = forecast(fit; h=1)

        @test isfinite(fc.mean[1])
    end

    # =========================================================================
    # Bug Fix Tests (Issues #1-6)
    # =========================================================================

    @testset "Issue #1: MA affects forecasts" begin
        # Forecasts should differ with MA terms vs without
        fit_ar_only = ararma(ap; p=2, q=0)
        fit_with_ma = ararma(ap; p=2, q=2)

        fc_ar = forecast(fit_ar_only; h=6)
        fc_ma = forecast(fit_with_ma; h=6)

        # MA component should affect at least the first q forecasts
        # Only test if: (1) theta is non-zero and (2) ARMA is usable (stable/invertible)
        theta_nonzero = any(fit_with_ma.arma_theta .!= 0.0)
        arma_usable = Durbyn.Ararma.is_arma_usable(fit_with_ma.arma_phi, fit_with_ma.arma_theta)
        if theta_nonzero && arma_usable
            @test fc_ar.mean != fc_ma.mean
        end
    end

    @testset "Issue #2: Formula min bounds respected" begin
        data = (y = ap,)

        # Test that p(2,4) never returns p < 2
        fit = @formula(y = p(2, 4) + q(0, 2)) |> f -> ararma(f, data)
        @test fit.ar_order >= 2
        @test fit.ar_order <= 4

        # Test that q(1,2) never returns q < 1
        fit2 = @formula(y = p(0, 3) + q(1, 2)) |> f -> ararma(f, data)
        @test fit2.ma_order >= 1
        @test fit2.ma_order <= 2
    end

    @testset "Issue #3: max_lag validation" begin
        # max_ar_depth < 4 should throw
        @test_throws ArgumentError ararma(ap; max_ar_depth=3, max_lag=20)

        # When max_ar_depth > max_lag, max_ar_depth gets clamped (with warning)
        # This should NOT throw, but warn and clamp
        fit = ararma(ap; max_ar_depth=30, max_lag=20)
        @test fit isa ArarmaModel
    end

    @testset "Issue #4: Short series auto-adjustment" begin
        # Very short series (n=12) should not crash and should auto-adjust parameters
        short_series = ap[1:12]
        fit = ararma(short_series; p=2, q=1)
        @test fit isa ArarmaModel
        @test length(fit.y_original) == 12

        fc = forecast(fit; h=3)
        @test length(fc.mean) == 3
        @test all(isfinite.(fc.mean))
    end

    @testset "Issue #5: Ill-conditioned fallback" begin
        # Near-constant series should use fallback lag without crashing
        near_constant = ones(50) .+ randn(50) .* 1e-10
        fit = ararma(near_constant; p=1, q=0)
        @test fit isa ArarmaModel
        # Should get a valid best_lag tuple
        @test fit.best_lag[1] == 1
        @test fit.best_lag[2] > 1
    end

    @testset "Issue #6: Stability functions" begin
        # Import stability functions for direct testing
        ar_stable = Durbyn.Ararma.ar_stable
        ma_invertible = Durbyn.Ararma.ma_invertible

        # Test AR stability
        @test ar_stable(Float64[])  # Empty is stable
        @test ar_stable([0.0, 0.0])  # All zeros is stable
        @test ar_stable([0.5])  # |root| > 1 for 1-0.5z
        @test ar_stable([0.3, 0.2])  # Small coeffs
        @test !ar_stable([1.5])  # Root inside unit circle

        # Test MA invertibility
        @test ma_invertible(Float64[])  # Empty is invertible
        @test ma_invertible([0.0, 0.0])  # All zeros is invertible
        @test ma_invertible([0.5])  # |root| > 1 for 1+0.5z
        @test ma_invertible([0.3, 0.2])  # Small coeffs
        @test !ma_invertible([1.5])  # Root inside unit circle
    end

    @testset "auto_ararma() - min bounds validation" begin
        # Test that min_p/min_q are respected
        fit = auto_ararma(ap; min_p=2, max_p=4, min_q=1, max_q=2)
        @test fit.ar_order >= 2
        @test fit.ma_order >= 1

        # Test invalid bounds throw errors
        @test_throws ArgumentError auto_ararma(ap; min_p=5, max_p=3)
        @test_throws ArgumentError auto_ararma(ap; min_q=3, max_q=1)
    end

    @testset "auto_ararma() - respects search range" begin
        # With min_p=2, should never select p=0 or p=1
        fit = auto_ararma(ap; min_p=2, max_p=3, min_q=0, max_q=1, crit=:bic)
        @test fit.ar_order >= 2
    end

    # =========================================================================
    # Additional Validation Tests
    # =========================================================================

    @testset "Parameter validation - max_ar_depth and max_lag minimums" begin
        # max_ar_depth < 4 should throw
        @test_throws ArgumentError ararma(ap; max_ar_depth=3, max_lag=10)

        # max_lag < 4 should throw
        @test_throws ArgumentError ararma(ap; max_ar_depth=4, max_lag=3)

        # Series with n < 5 should throw
        @test_throws ArgumentError ararma([1.0, 2.0, 3.0, 4.0])
    end

    @testset "Parameter validation - max_lag vs series length" begin
        # max_lag exceeding series length should be clamped (with warning)
        short = ap[1:10]
        # This should work (clamped) rather than error
        fit = ararma(short; max_ar_depth=4, max_lag=50)
        @test fit isa ArarmaModel
        # gamma length should be clamped to n
        @test length(fit.gamma) <= length(short)
    end

    @testset "Parameter validation - negative min_p/min_q" begin
        @test_throws ArgumentError auto_ararma(ap; min_p=-1, max_p=4)
        @test_throws ArgumentError auto_ararma(ap; min_q=-2, max_q=2)
    end

    @testset "Constant series handling" begin
        # Perfectly constant series should not crash (uses fallback with zero coefficients)
        constant_series = ones(50)
        # This might warn but should not throw
        fit = ararma(constant_series; p=1, q=0)
        @test fit isa ArarmaModel
    end

    @testset "Parameter validation - negative p/q" begin
        @test_throws ArgumentError ararma(ap; p=-1, q=1)
        @test_throws ArgumentError ararma(ap; p=2, q=-1)
    end

    @testset "Formula precedence over kwargs" begin
        data = (y = ap,)

        # When formula specifies p(), kwargs for p should be ignored
        fit = @formula(y = p(2, 4) + q(0, 2)) |> f -> ararma(f, data; min_p=0, max_p=10)
        @test fit.ar_order >= 2
        @test fit.ar_order <= 4

        # In fixed-order mode, min_p/max_p/crit kwargs should be ignored (not error)
        fit_fixed = @formula(y = p(3) + q(1)) |> f -> ararma(f, data; min_p=0, max_p=5, crit=:bic)
        @test fit_fixed.ar_order == 3
        @test fit_fixed.ma_order == 1

        # When formula omits p(), kwargs CAN override defaults
        # Formula only specifies q(1,2), so max_p kwarg should be used
        fit_override = @formula(y = q(1, 2)) |> f -> ararma(f, data; max_p=2)
        @test fit_override.ar_order <= 2  # respects max_p kwarg
        @test fit_override.ma_order >= 1  # respects formula's q(1,2)
    end

    @testset "fit_arma residual length validation" begin
        # Directly test fit_arma with too-short residuals
        short_resid = [1.0, 2.0, 3.0]  # length 3

        # ARMA(4,0) needs at least 5 observations
        @test_throws ArgumentError Durbyn.Ararma.fit_arma(4, 0, short_resid)

        # ARMA(0,4) needs at least 5 observations
        @test_throws ArgumentError Durbyn.Ararma.fit_arma(0, 4, short_resid)

        # ARMA(2,2) needs at least 3 observations - this should work
        result = Durbyn.Ararma.fit_arma(2, 2, short_resid)
        @test length(result[1]) == 2  # phi
        @test length(result[2]) == 2  # theta
    end
end
