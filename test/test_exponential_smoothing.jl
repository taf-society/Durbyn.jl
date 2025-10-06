using Test
using Durbyn
import Durbyn.Generics: Forecast, forecast
using Durbyn.ExponentialSmoothing

@testset "Durbyn.ExponentialSmoothing Tests" begin
    ap = air_passengers()
    @testset "ets() - ETS State Space Model" begin
        @testset "Basic automatic model selection (ZZZ)" begin
            fit = ets(ap, 12, "ZZZ")
            fc = forecast(fit, h = 5, level=[50, 60, 70, 80, 90])
            plot(fc)
            @test fit isa ETS
            @test fit.m == 12
            @test length(fit.fitted) == length(ap)
            @test length(fit.residuals) == length(ap)
            @test fit.aic ≈ 1442.809116061896
            @test fit.bic ≈ 1493.295942154688
            @test fit.aicc ≈ 1447.666258919039
            @test fc.mean == [442.86762657690826, 416.72087316234746, 
            469.0511903879788, 490.3502542340624, 502.0331066405476]
        end

        @testset "Specific model structures" begin
            # Additive error, additive trend, no season
            fit_aan = ets(ap, 12, "AAN")
            fc_aan = forecast(fit_aan, h = 5)
            plot(fc_aan)
            @test fc_aan.mean == [433.9422439065297, 435.91790638016016, 
            437.89356885379067, 439.8692313274211, 441.84489380105157]
            @test fit_aan isa ETS

            # Additive error, additive trend, additive season
            fit_aaa = ets(ap, 12, "AAA")
            fc_aaa = forecast(fit_aaa, h = 5)
            plot(fc_aaa)
            @test fc_aaa.mean == [453.5774393666588, 429.2731911239944, 
            466.44373223545244, 502.6987535852552, 511.8096155802117]
            @test fit_aaa isa ETS

            # Multiplicative error, additive trend, multiplicative season
            fit_mam = ets(ap, 12, "MAM")
            fc_mam = forecast(fit_mam, h = 5)
            #plot(fc_mam)
            @test fit_mam isa ETS
            @test fc_mam.mean == [442.86762657690826, 416.72087316234746, 469.0511903879788,
             490.3502542340624, 502.0331066405476]
        end

        @testset "Damped trend" begin
            fit_damped = ets(ap, 12, "AAN"; damped=true)
            fc_damped = forecast(fit_damped, h = 3)
            #plot(fc_damped)
            @test fit_damped isa ETS
            @test fit_damped.par["phi"] ==  0.9741021384906694
            @test fc_damped.mean == [432.03818371264083, 432.0917828086241, 432.1439938026426]

            # Auto-selection with damping consideration
            fit_damped_auto = ets(ap, 12, "ZZZ"; damped=nothing)
            fc_damped_auto = forecast(fit_damped_auto, h = 3)
            # plot(fc_damped_auto)
            @test fit_damped_auto isa ETS
        end

        @testset "Fixed smoothing parameters" begin
            fit_fixed = ets(ap, 12, "AAN"; alpha=0.3, beta=0.1)
            fc__fixed = forecast(fit_fixed, h = 12)
            #plot(fc__fixed)
            @test fit_fixed isa ETS
            @test fit_fixed.par["alpha"] ≈ 0.9993573983379229
            @test fit_fixed.par["beta"] ≈ 0.00017930074082943281
        end

        @testset "Box-Cox transformation" begin
            fit_lambda = ets(ap, 12, "AAN"; lambda=0.5)
            fc_lambda = forecast(fit_lambda, h = 3)
            # plot(fc_lambda)
            @test fit_lambda isa ETS
            @test fit_lambda.lambda == 0.5

            # Auto lambda selection
            fit_auto_lambda = ets(ap, 12, "AAN"; lambda="auto")
            fc_auto_lambda = forecast(fit_auto_lambda, h = 3)
            # plot(fc_auto_lambda)
            @test fit_auto_lambda isa ETS
            @test fit_auto_lambda.lambda !== nothing
            fc_auto_lambda.mean ≈ [446.43081823715215, 448.20671894484923, 449.9917769985243]
        end

        @testset "Information criteria selection" begin
            fit_aic = ets(ap, 12, "ZZZ"; ic="aic")
            @test fit_aic.aic ≈ 1442.809116061896

            fit_bic = ets(ap, 12, "ZZZ"; ic="bic")
            @test fit_bic.bic ≈ 1493.295942154688

            fit_aicc = ets(ap, 12, "ZZZ"; ic="aicc")
            @test fit_aicc.aicc ≈ 1447.666258919039
        end

        @testset "Forecasting from ETS model" begin
            fit = ets(ap, 12, "AAA")
            fc = forecast(fit; h=3)
            #plot(fc)
            @test fc isa Forecast
            @test fc.mean ≈ [453.5774393666588, 429.2731911239944, 466.44373223545244]

            fc_levels = forecast(fit; h=12, level=[80, 95])
            @test size(fc_levels.upper) == (12, 2)
            @test size(fc_levels.lower) == (12, 2)
        end

        @testset "Constant series handling" begin
            const_series = fill(100.0, 50)
            fit_const = ets(const_series, 12, "ZZZ")
            fc_const = forecast(fit_const, h = 5)
            #plot(fc_const)
            @test fc_const.mean ≈ [100.0, 100.0, 100.0, 100.0, 100.0]
            @test fit_const isa SES  # Should fall back to SES
        end
    end

    @testset "ses() - Simple Exponential Smoothing" begin
        @testset "Optimal initialization" begin
            fit = ses(ap, 12)
            fc = forecast(fit, h = 4)
            fit2 = ses(ap)
            fc2 = forecast(fit2, h = 4)
            fc.mean == fc2.mean
            @test fit isa SES
            @test fit.method == "Simple Exponential Smoothing"
            @test length(fit.fitted) == length(ap)
            @test length(fit.residuals) == length(ap)
            @test !isnan(fit.aic)
            @test !isnan(fit.mse)
            @test fc.mean ≈ [431.9929197152949, 431.9929197152949, 431.9929197152949, 431.9929197152949]
        end

        @testset "Simple initialization" begin
            fit_simple = ses(ap, 12; initial="simple")
            fc_simple = forecast(fit_simple, h = 4)
            @test fc_simple.mean ≈ [459.4035874451524, 459.4035874451524, 459.4035874451524, 459.4035874451524]
            @test fit_simple isa SES
            @test isnan(fit_simple.aic)  # Simple init doesn't compute IC
        end

        @testset "Fixed initial alpha parameter" begin
            fit_alpha = ses(ap, 12; alpha=0.2)
            #fc = forecast(fit_alpha, h = 12)
            #plot(fc)
            @test fit_alpha isa SES
            @test fit_alpha.par["alpha"] ≈ 0.9998967227442795
        end

        @testset "Box-Cox transformation" begin
            fit_lambda = ses(ap, 12; lambda=0.5, biasadj=true)
            #fc = forecast(fit_lambda, h = 12)
            #plot(fc)
            @test fit_lambda isa SES
            @test fit_lambda.lambda == 0.5
            @test fit_lambda.biasadj == true
        end

        @testset "Forecasting from SES" begin
            fit = ses(ap, 12)
            fc = forecast(fit; h=4)
            @test fc isa Forecast
            @test fc.mean ≈ [431.9929197152949, 431.9929197152949, 431.9929197152949, 431.9929197152949]
        end
    end

    @testset "holt() - Holt's Linear Trend Method" begin
        @testset "Basic Holt method" begin
            fit1 = holt(ap, 12)
            fit2 = holt(ap)
            fc1 = forecast(fit1, h = 4)
            fc2 = forecast(fit2, h = 4)
            @test fc1.mean == fc2.mean 
            # plot(fc2)
            @test fc1.mean ≈ [433.9993718605254, 436.003094616045, 438.00681737156464, 440.01054012708425]
            @test fit1 isa Holt
            @test occursin("Holt's method", fit1.method)
            @test length(fit1.fitted) == length(ap)
        end

        @testset "Damped trend" begin
            fit_damped1 = holt(ap, 12; damped=true)
            fit_damped2 = holt(ap; damped=true)
            fc1 = forecast(fit_damped1, h = 4)
            fc2 = forecast(fit_damped2, h = 4)
            # plot(fc2)
            @test fc1.mean == fc2.mean
            @test fit_damped1 isa Holt
            @test occursin("Damped", fit_damped1.method)
            @test haskey(fit_damped1.par, "phi")
        end

        @testset "Exponential trend" begin
            fit_exp1 = holt(ap, 12; exponential=true)
            fit_exp2 = holt(ap; exponential=true)
            fc1 = forecast(fit_exp1, h = 12)
            fc2 = forecast(fit_exp2, h = 12)
            # plot(fc2)
            @test fc1.mean == fc2.mean
            @test fit_exp1 isa Holt
            @test occursin("exponential trend", fit_exp1.method)
        end

        @testset "Simple initialization" begin
            fit_simple1 = holt(ap, 12; initial="simple")
            fit_simple2 = holt(ap; initial="simple")
            fc1 = forecast(fit_simple1, h = 12)
            fc2 = forecast(fit_simple2, h = 12)
            plot(fc1)
            plot(fc2)
            @test all(fc1.mean - fc2.mean .< 1)
            @test fit_simple1 isa Holt
            @test fit_simple2 isa Holt
        end

        @testset "Fixed parameters" begin
            fit_fixed = holt(ap, 12; alpha=0.3, beta=0.1)
            #plot(forecast(fit_fixed, h = 12))
            @test fit_fixed isa Holt
            @test fit_fixed.par["alpha"] ≈ 0.9998702179600357
            @test fit_fixed.par["beta"] ≈ 0.00013310303558172737
        end

        @testset "Short series error" begin
            @test_throws ArgumentError holt([100.0], 1)
        end

        @testset "Forecasting from Holt" begin
            fit = holt(ap, 12)
            fc = forecast(fit; h=12)
            plot(fc)
            @test fc isa Forecast
            @test length(fc.mean) == 12
            # Holt forecasts should show linear trend
            @test !all(fc.mean .≈ fc.mean[1])
        end
    end

    @testset "holt_winters() - Holt-Winters Seasonal Method" begin
        @testset "Additive seasonality" begin
            fit = holt_winters(ap, 12; seasonal="additive")
            fc = forecast(fit, h = 4)
            #plot(fc)
            @test fc.mean ≈ [453.5774393666588, 429.2731911239944, 466.44373223545244, 502.6987535852552]
            @test fit isa HoltWinters
            @test occursin("additive", fit.method)
            @test length(fit.fitted) == length(ap)
        end

        @testset "Multiplicative seasonality" begin
            fit = holt_winters(ap, 12; seasonal="multiplicative")
            fc = forecast(fit, h = 4)
            # plot(fc)
            @test fc.mean ≈ [432.1676153635917, 406.0390668265751, 443.83818710265473, 475.9148322339634]
            @test fit isa HoltWinters
            @test occursin("multiplicative", fit.method)
        end

        @testset "Damped trend" begin
            fit_damped = holt_winters(ap, 12; damped=true)
            fc = forecast(fit_damped, h = 4)
            plot(fc)
            @test fc.mean ≈ [451.28599287660575, 429.1690101784728, 472.0045688929445, 497.5134710351016]
            @test fit_damped isa HoltWinters
            @test occursin("Damped", fit_damped.method)
        end

        @testset "Exponential trend with multiplicative season" begin
            fit = holt_winters(ap, 12; seasonal="multiplicative", exponential=false)
            fc = forecast(fit, h = 12)
            plot(fc)
            @test fc.mean
            @test fit isa HoltWinters
            @test occursin("exponential trend", fit.method)
        end

        @testset "Invalid combinations" begin
            # Additive seasonality with exponential trend is forbidden
            @test_throws ArgumentError holt_winters(ap, 12; seasonal="additive", exponential=true)
        end

        @testset "Simple initialization" begin
            fit_simple = holt_winters(ap, 12; initial="simple")
            @test fit_simple isa HoltWinters
        end

        @testset "Fixed parameters" begin
            fit_fixed = holt_winters(ap, 12; alpha=0.3, beta=0.1, gamma=0.2)
            @test fit_fixed isa HoltWinters
            @test fit_fixed.par["alpha"] ≈ 0.3
            @test fit_fixed.par["beta"] ≈ 0.1
            @test fit_fixed.par["gamma"] ≈ 0.2
        end

        @testset "Frequency validation" begin
            @test_throws ArgumentError holt_winters(ap, 1)  # m <= 1
        end

        @testset "Insufficient data error" begin
            short_series = ap[1:10]
            @test_throws ArgumentError holt_winters(short_series, 12)
        end

        @testset "Forecasting from Holt-Winters" begin
            fit = holt_winters(ap, 12)
            fc = forecast(fit; h=24)
            @test fc isa Forecast
            @test length(fc.mean) == 24

            fc_levels = forecast(fit; h=12, level=[90])
            @test size(fc_levels.upper) == (12, 1)
        end
    end

    @testset "croston() - Croston's Method for Intermittent Demand" begin
        @testset "Regular intermittent series" begin
            intermittent = [0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 7.0, 0.0, 0.0, 4.0]
            fit = croston(intermittent, 1)
            @test fit isa CrostonFit
            @test fit.type == CrostonFour
            @test fit.m == 1
        end

        @testset "All zeros (type 1)" begin
            zeros_series = zeros(10)
            fit = croston(zeros_series, 1)
            @test fit isa CrostonFit
            @test fit.type == CrostonOne
        end

        @testset "Single non-zero value (type 2)" begin
            single_val = [0.0, 0.0, 5.0, 0.0, 0.0]
            fit = croston(single_val, 1)
            @test fit isa CrostonFit
            @test fit.type == CrostonTwo
        end

        @testset "Fixed alpha parameter" begin
            intermittent = [0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 7.0, 0.0, 0.0, 4.0]
            fit = croston(intermittent, 1; alpha=0.2)
            @test fit isa CrostonFit
        end

        @testset "Forecasting from Croston" begin
            intermittent = [0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 7.0, 0.0, 0.0, 4.0]
            fit = croston(intermittent, 1)
            fc = forecast(fit, 6)
            @test fc isa CrostonForecast
            @test length(fc.mean) == 6
            @test fc.method == "Croston's Method"
        end

        @testset "Fitted values" begin
            intermittent = [0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 7.0, 0.0, 0.0, 4.0]
            fit = croston(intermittent, 1)
            fits = fitted(fit)
            @test length(fits) == length(intermittent)
        end
    end

    @testset "Model Comparison and Integration" begin
        @testset "Consistency across methods" begin
            # SES via ets should match ses()
            fit_ets = ets(ap, 12, "ANN")
            fit_ses = ses(ap, 12)

            @test fit_ets isa ETS
            @test fit_ses isa SES
            # Both should produce similar fitted values
            @test cor(fit_ets.fitted, fit_ses.fitted) > 0.99
        end

        @testset "Holt via ets should match holt()" begin
            fit_ets = ets(ap, 12, "AAN")
            fit_holt = holt(ap, 12)

            @test fit_ets isa ETS
            @test fit_holt isa Holt
            @test cor(fit_ets.fitted, fit_holt.fitted) > 0.99
        end

        @testset "Holt-Winters via ets should match holt_winters()" begin
            fit_ets = ets(ap, 12, "AAA")
            fit_hw = holt_winters(ap, 12)

            @test fit_ets isa ETS
            @test fit_hw isa HoltWinters
            @test cor(fit_ets.fitted, fit_hw.fitted) > 0.99
        end
    end
end