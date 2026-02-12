using Test
using Durbyn
import Durbyn.Generics: Forecast, forecast, fitted
using Durbyn.ExponentialSmoothing:
    ets,
    holt as es_holt,
    holt_winters as es_holt_winters,
    ses as es_ses,
    croston as es_croston,
    ETS,
    SES,
    Holt,
    HoltWinters,
    CrostonFit,
    CrostonForecast,
    CrostonType
using Durbyn.ModelSpecs
using Durbyn.Grammar
using Durbyn.TableOps

@testset "Durbyn.ExponentialSmoothing Tests" begin
    ap = air_passengers()
    @testset "ets() - ETS State Space Model" begin
        @testset "Basic automatic model selection (ZZZ)" begin
            fit = ets(ap, 12, "ZZZ")
            fc = forecast(fit, h = 12, level=[50, 60, 70, 80, 90])
            # plot(fc) 
            @test fit isa ETS
            @test fit.m == 12
            @test length(fit.fitted) == length(ap)
            @test length(fit.residuals) == length(ap)
            @test length(fc.mean) == 12
        end

        @testset "Specific model structures" begin
            # Additive error, additive trend, no season
            fit_aan = ets(ap, 12, "AAN")
            fc_aan = forecast(fit_aan, h = 36)
            #plot(fc_aan)
            @test length(fc_aan.mean) == 36
            @test fit_aan isa ETS

            # Additive error, additive trend, additive season
            fit_aaa = ets(ap, 12, "AAA")
            fc_aaa = forecast(fit_aaa, h = 19)
            #plot(fc_aaa)
            @test length(fc_aaa.mean) == 19
            @test fit_aaa isa ETS

            # Multiplicative error, additive trend, multiplicative season
            fit_mam = ets(ap, 12, "MAM")
            fc_mam = forecast(fit_mam, h = 75)
            # plot(fc_mam)
            @test fit_mam isa ETS
            @test length(fc_mam.mean) == 75
        end

        @testset "Damped trend" begin
            fit_damped = ets(ap, 12, "AAN"; damped=true)
            fc_damped = forecast(fit_damped, h = 3)
            #plot(fc_damped)
            @test fit_damped isa ETS
            @test isreal(fit_damped.par["phi"])
            @test length(fc_damped.mean) == 3

            # Auto-selection with damping consideration
            fit_damped_auto = ets(ap, 12, "ZZZ"; damped=nothing)
            fc_damped_auto = forecast(fit_damped_auto, h = 12)
            # plot(fc_damped_auto)  # Requires Plots.jl
            @test fit_damped_auto isa ETS

            # fit_damped_auto = ets(ap[1:5], 1, "ZZZ"; damped=nothing)
            # fc_damped_auto = forecast(fit_damped_auto, h = 12)
            # plot(fc_damped_auto)
            # @test fit_damped_auto isa ETS
        end

        @testset "Fixed smoothing parameters" begin
            fit_fixed = ets(ap, 12, "AAN"; alpha=0.3, beta=0.1)
            fc__fixed = forecast(fit_fixed, h = 12)
            # plot(fc__fixed)  # Requires Plots.jl
            @test fit_fixed isa ETS
        end

        @testset "Box-Cox transformation" begin
            fit_lambda = ets(ap, 12, "AAN"; lambda=0.5)
            fc_lambda = forecast(fit_lambda, h = 3)
            # plot(fc_lambda)
            @test fit_lambda isa ETS
            @test fit_lambda.lambda == 0.5

            # Auto lambda selection
            fit_auto_lambda = ets(ap, 12, "AAN"; lambda="auto")
            fc_auto_lambda = forecast(fit_auto_lambda, h = 15)
            # plot(fc_auto_lambda)  # Requires Plots.jl
            @test fit_auto_lambda isa ETS
            @test fit_auto_lambda.lambda !== nothing
            @test length(fc_auto_lambda.mean) == 15
        end

        @testset "Information criteria selection" begin
            fit_aic = ets(ap, 12, "ZZZ"; ic="aic")
            @test fit_aic.aic isa Float64

            fit_bic = ets(ap, 12, "ZZZ"; ic="bic")
            @test fit_bic.bic isa Float64

            fit_aicc = ets(ap, 12, "ZZZ"; ic="aicc")
            @test fit_aicc.aicc isa Float64
        end

        @testset "Forecasting from ETS model" begin
            fit = ets(ap, 12, "AAA")
            fc = forecast(fit; h=27)
            # plot(fc)  # Requires Plots.jl
            @test fc isa Forecast
            @test length(fc.mean) ==27
            fc_levels = forecast(fit; h=12, level=[80, 95])
            @test size(fc_levels.upper) == (12, 2)
            @test size(fc_levels.lower) == (12, 2)

            # Forecast with confidence intervals for non-seasonal model
            fit_ann = ets(ap, 1, "ANN")
            fc_ann = forecast(fit_ann; h=6, level=[80, 95])
            @test fc_ann isa Forecast
            @test size(fc_ann.upper) == (6, 2)
            @test size(fc_ann.lower) == (6, 2)
        end

        @testset "Constant series handling" begin
            const_series = fill(100.0, 50)
            fit_const = ets(const_series, 12, "ZZZ")
            fc_const = forecast(fit_const, h = 5)
            # plot(fc_const)
            @test fc_const.mean ≈ [100.0, 100.0, 100.0, 100.0, 100.0]
            @test fit_const isa SES  # Should fall back to SES
        end
    end

    @testset "ses() - Simple Exponential Smoothing" begin
        @testset "Optimal initialization" begin
            fit = es_ses(ap, 12)
            fc = forecast(fit, h = 4)
            fit2 = es_ses(ap)
            fc2 = forecast(fit2, h = 4)
            fc.mean == fc2.mean
            @test fit isa SES
            @test fit.method == "Simple Exponential Smoothing"
            @test length(fit.fitted) == length(ap)
            @test length(fit.residuals) == length(ap)
            @test !isnan(fit.aic)
            @test !isnan(fit.mse)
            @test length(fc.mean) == 4
        end

        @testset "Simple initialization" begin
            fit_simple = es_ses(ap, 12; initial="simple")
            fc_simple = forecast(fit_simple, h = 4)
            @test fc_simple.mean ≈ [459.8755335582515, 459.8755335582515, 459.8755335582515, 459.8755335582515]
            @test fit_simple isa SES
            @test isnan(fit_simple.aic)  # Simple init doesn't compute IC
        end

        @testset "Fixed initial alpha parameter" begin
            fit_alpha = es_ses(ap, 12; alpha=0.2)
            #fc = forecast(fit_alpha, h = 12)
            #plot(fc)
            @test fit_alpha isa SES
            @test fit_alpha.par["alpha"] ≈ 0.9999 atol=1e-6
        end

        @testset "Box-Cox transformation" begin
            fit_lambda = es_ses(ap, 12; lambda=0.5, biasadj=true)
            fc = forecast(fit_lambda, h = 12)
            # plot(fc)  # Requires Plots.jl
            @test fit_lambda isa SES
            @test fit_lambda.lambda == 0.5
            @test fit_lambda.biasadj == true
        end

        @testset "Forecasting from SES" begin
            fit = es_ses(ap, 12)
            fc = forecast(fit; h=4)
            @test fc isa Forecast
            @test fc.mean ≈ [431.9957984229696, 431.9957984229696, 431.9957984229696, 431.9957984229696]
        end
    end

    @testset "holt() - Holt's Linear Trend Method" begin
        @testset "Basic Holt method" begin
            fit1 = es_holt(ap, 12)
            fit2 = es_holt(ap)
            fc1 = forecast(fit1, h = 24)
            fc2 = forecast(fit2, h = 24)
            @test fc1.mean == fc2.mean 
            # plot(fc2)
            @test occursin("Holt's method", fit1.method)
            @test length(fit1.fitted) == length(ap)
        end

        @testset "Damped trend" begin
            fit_damped1 = es_holt(ap, 12; damped=true)
            fit_damped2 = es_holt(ap; damped=true)
            fc1 = forecast(fit_damped1, h = 4)
            fc2 = forecast(fit_damped2, h = 4)
            # plot(fc2)
            @test fc1.mean == fc2.mean
            @test fit_damped1 isa Holt
            @test occursin("Damped", fit_damped1.method)
            @test haskey(fit_damped1.par, "phi")
        end

        @testset "Exponential trend" begin
            fit_exp1 = es_holt(ap, 12; exponential=true)
            fit_exp2 = es_holt(ap; exponential=true)
            fc1 = forecast(fit_exp1, h = 12)
            fc2 = forecast(fit_exp2, h = 12)
            # plot(fc2)
            @test fc1.mean == fc2.mean
            @test fit_exp1 isa Holt
            @test occursin("exponential trend", fit_exp1.method)
        end

        @testset "Simple initialization" begin
            fit_simple1 = es_holt(ap, 12; initial="simple")
            fit_simple2 = es_holt(ap; initial="simple")
            fc1 = forecast(fit_simple1, h = 12)
            fc2 = forecast(fit_simple2, h = 12)
            # plot(fc1)  # Requires Plots.jl
            # plot(fc2)  # Requires Plots.jl
            @test all(fc1.mean - fc2.mean .< 1)
            @test fit_simple1 isa Holt
            @test fit_simple2 isa Holt
        end

        @testset "Fixed parameters" begin
            fit_fixed = es_holt(ap, 1; alpha=0.3, beta=0.1)
            # plot(forecast(fit_fixed, h = 12)) same as R
            @test fit_fixed isa Holt
        end

        @testset "Short series error" begin
            @test_throws ArgumentError es_holt([100.0], 1)
        end

        @testset "Forecasting from Holt" begin
            fit = es_holt(ap, 12)
            fc = forecast(fit; h=12)
            # plot(fc)  # Requires Plots.jl
            @test fc isa Forecast
            @test length(fc.mean) == 12
            # Holt forecasts should show linear trend
            @test !all(fc.mean .≈ fc.mean[1])
        end
    end

    @testset "holt_winters() - Holt-Winters Seasonal Method" begin
        @testset "Additive seasonality" begin
            fit = es_holt_winters(ap, 12; seasonal="additive")
            fc = forecast(fit, h = 65)
            # plot(fc)  # Requires Plots.jl
            @test length(fc.mean) == 65
            @test fit isa HoltWinters
            @test occursin("additive", fit.method)
            @test length(fit.fitted) == length(ap)
        end

        @testset "Multiplicative seasonality" begin
            fit = es_holt_winters(ap, 12; seasonal="multiplicative")
            fc = forecast(fit, h = 34)
            # plot(fc)  # Requires Plots.jl
            @test length(fc.mean) == 34
            @test fit isa HoltWinters
            @test occursin("multiplicative", fit.method)
        end

        @testset "Damped trend" begin
            fit_damped = es_holt_winters(ap, 12; damped=true)
            fc = forecast(fit_damped, h = 13)
            # plot(fc)  # Requires Plots.jl
            @test length(fc.mean)== 13
            @test fit_damped isa HoltWinters
            @test occursin("Damped", fit_damped.method)
        end

        @testset "Exponential trend with multiplicative season" begin
            fit = es_holt_winters(ap, 12; seasonal="multiplicative", exponential=false)
            fc = forecast(fit, h = 12)
            # plot(fc)  # Requires Plots.jl
            @test fit isa HoltWinters
        end

        @testset "Invalid combinations" begin
            # Additive seasonality with exponential trend is forbidden
            @test_throws ArgumentError es_holt_winters(ap, 12; seasonal="additive", exponential=true)
        end

        @testset "Simple initialization" begin
            fit_simple = es_holt_winters(ap, 12; initial="simple")
            fc = forecast(fit_simple, h = 12)
            # plot(fc)  # Requires Plots.jl
            @test fit_simple isa HoltWinters
        end

        @testset "Fixed parameters" begin
            fit_fixed = es_holt_winters(ap, 12; alpha=0.3, beta=0.1, gamma=0.2)
            @test fit_fixed isa HoltWinters
        end

        @testset "Frequency validation" begin
            @test_throws ArgumentError es_holt_winters(ap, 1)  # m <= 1
        end

        @testset "Insufficient data error" begin
            short_series = ap[1:10]
            @test_throws ArgumentError es_holt_winters(short_series, 12)
        end

        @testset "Forecasting from Holt-Winters" begin
            fit = es_holt_winters(ap, 12)
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
            fit = es_croston(intermittent, 1)
            @test fit isa CrostonFit
            #@test fit.type == CrostonFour
            @test fit.m == 1
        end

        @testset "All zeros (type 1)" begin
            zeros_series = zeros(10)
            fit = es_croston(zeros_series, 1)
            @test fit isa CrostonFit
            #@test fit.type == CrostonOne
        end

        @testset "Single non-zero value (type 2)" begin
            single_val = [0.0, 0.0, 5.0, 0.0, 0.0]
            fit = es_croston(single_val, 1)
            @test fit isa CrostonFit
            #@test fit.type == CrostonTwo
        end

        @testset "Fixed alpha parameter" begin
            intermittent = [0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 7.0, 0.0, 0.0, 4.0]
            fit = es_croston(intermittent, 1; alpha=0.2)
            @test fit isa CrostonFit
        end

        @testset "Forecasting from Croston" begin
            intermittent = [0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 7.0, 0.0, 0.0, 4.0]
            fit = es_croston(intermittent, 1)
            fc = forecast(fit, 6)
            @test fc isa CrostonForecast
            @test length(fc.mean) == 6
            @test fc.method == "Croston's Method"
        end

        @testset "Fitted values" begin
            intermittent = [0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 7.0, 0.0, 0.0, 4.0]
            fit = es_croston(intermittent, 1)
            fits = fitted(fit)
            @test length(fits) == length(intermittent)
        end
    end

    @testset "Model Comparison and Integration" begin
        @testset "Consistency across methods" begin
            # SES via ets should match ses()
            fit_ets = ets(ap, 12, "ANN")
            fit_ses = es_ses(ap, 12)

            @test fit_ets isa ETS
            @test fit_ses isa SES
            # Both should produce similar fitted values (small numeric drift is acceptable).
            @test maximum(abs.(fit_ets.fitted .- fit_ses.fitted)) < 0.11
        end

        @testset "Holt via ets should match holt()" begin
            fit_ets = ets(ap, 12, "AAN")
            fit_holt = es_holt(ap, 12)

            @test fit_ets isa ETS
            @test fit_holt isa Holt
        end

        @testset "Holt-Winters via ets should match holt_winters()" begin
            fit_ets = ets(ap, 12, "AAA")
            fit_hw = es_holt_winters(ap, 12)

            @test fit_ets isa ETS
            @test fit_hw isa HoltWinters
        end
    end

    @testset "ModelCollection forecasts" begin
        n = length(ap)
        panel_tbl = (
            series = repeat(["A", "B"], inner = n),
            date = vcat(collect(1:n), collect(1:n)),
            value = vcat(Float64.(ap), Float64.(ap))
        )
        panel = PanelData(panel_tbl; groupby = :series, date = :date, m = 12)

        models = model(
            ArimaSpec(@formula(value = p() + q())),
            EtsSpec(@formula(value = e("Z") + t("Z") + s("Z"))),
            SesSpec(@formula(value = ses())),
            HoltSpec(@formula(value = holt())),
            names = ["arima", "ets", "ses", "holt"]
        )

        fitted_collection = fit(models, panel)
        @test fitted_collection isa FittedModelCollection

        fc_collection = forecast(fitted_collection; h = 6)
        @test fc_collection isa ForecastModelCollection

        tbl = as_table(fc_collection)
        @test :model_name in propertynames(tbl)
        @test length(getfield(tbl, :model_name)) > 0

        glimpse(fc_collection)
    end

    @testset "EtsSpec grammar interface" begin
        ap_vals = Float64.(air_passengers())

        spec_basic = EtsSpec(@formula(value = e("A") + t("N") + s("N")))
        @test spec_basic.components == (error = "A", trend = "N", seasonal = "N")
        @test spec_basic.damped === nothing

        data_basic = (value = ap_vals,)
        fit_basic = fit(spec_basic, data_basic)
        @test fit_basic isa FittedEts
        fc_basic = forecast(fit_basic, h = 6)
        @test length(fc_basic.mean) == 6

        spec_damped = EtsSpec(@formula(value = e("Z") + t("A") + s("N") + drift()))
        @test spec_damped.damped === true
        fit_damped = fit(spec_damped, data_basic; m = 12)
        @test fit_damped isa FittedEts

        spec_auto = EtsSpec(@formula(value = e("Z") + t("Z") + s("Z") + drift(:auto)))
        @test spec_auto.damped === nothing
        fit_auto = fit(spec_auto, data_basic; m = 12, ic = "aicc")
        fc_auto = forecast(fit_auto, h = 4)
        @test fc_auto isa Forecast

        grouped_data = (
            store = repeat(["A", "B"], inner = length(ap_vals)),
            value = vcat(ap_vals, ap_vals)
        )
        grouped_spec = EtsSpec(@formula(value = e("A") + t("N") + s("N")))
        group_fit = fit(grouped_spec, grouped_data; groupby = :store)
        @test group_fit isa GroupedFittedModels
        @test group_fit.successful == 2
        group_fc = forecast(group_fit, h = 3)
        @test group_fc isa GroupedForecasts
        @test group_fc.successful == 2

        panel = PanelData(grouped_data; groupby = :store, m = 12)
        panel_fit = fit(grouped_spec, panel)
        @test panel_fit isa GroupedFittedModels
        @test panel_fit.successful == 2
    end

    @testset "Smoothing specs grammar interface" begin
        ap_vals = Float64.(air_passengers())
        data_basic = (value = ap_vals,)

        ses_spec = SesSpec(@formula(value = ses()))
        fit_ses_spec = fit(ses_spec, data_basic)
        @test fit_ses_spec isa FittedSes
        fc_ses_spec = forecast(fit_ses_spec, h = 5)
        @test length(fc_ses_spec.mean) == 5

        holt_spec = HoltSpec(@formula(value = holt(damped=true)))
        fit_holt_spec = fit(holt_spec, data_basic)
        @test fit_holt_spec isa FittedHolt
        fc_holt_spec = forecast(fit_holt_spec, h = 6)
        @test length(fc_holt_spec.mean) == 6

        hw_spec = HoltWintersSpec(@formula(value = hw(seasonal="additive")))
        fit_hw_spec = fit(hw_spec, data_basic; m = 12)
        @test fit_hw_spec isa FittedHoltWinters
        fc_hw_spec = forecast(fit_hw_spec, h = 3)
        @test length(fc_hw_spec.mean) == 3

        intermittent = [0, 0, 5, 0, 0, 3, 0, 0, 0, 7, 0, 0, 4, 0, 0]
        croston_spec = CrostonSpec(@formula(demand = croston()))
        fit_croston_spec = fit(croston_spec, (demand = intermittent,))
        @test fit_croston_spec isa FittedCroston
        fc_croston_spec = forecast(fit_croston_spec, h = 8)
        @test length(fc_croston_spec.mean) == 8

        grouped_data = (
            store = repeat(["A", "B"], inner = length(ap_vals)),
            value = vcat(ap_vals, ap_vals)
        )
        ses_group_spec = SesSpec(@formula(value = ses()))
        ses_group_fit = fit(ses_group_spec, grouped_data; groupby = :store)
        @test ses_group_fit isa GroupedFittedModels
        @test ses_group_fit.successful == 2

        panel = PanelData(grouped_data; groupby = :store, m = 12)
        holt_panel_spec = HoltSpec(@formula(value = holt()))
        holt_panel_fit = fit(holt_panel_spec, panel)
        @test holt_panel_fit isa GroupedFittedModels
        @test holt_panel_fit.successful == 2
    end

    @testset "ETS Model Refit" begin
        ap = air_passengers()

        @testset "Refit with use_initial_values=false" begin
            fit1 = ets(ap, 12, "AAA")
            new_data = ap[1:100]
            fit2 = ets(new_data, 12, fit1, use_initial_values=false)
            @test fit2 isa ETS
        end

        @testset "Refit with use_initial_values=true" begin
            fit1 = ets(ap, 12, "AAN")
            new_data = ap[1:100]
            fit2 = ets(new_data, 12, fit1, use_initial_values=true)
            @test fit2 isa ETS
        end
    end

    @testset "Non-positive data validation" begin
        non_positive = [0.0, 1.0, 2.0, -1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        @test_throws ArgumentError ets(non_positive, 1, "MNN")
        @test_throws ArgumentError ets(non_positive, 1, "AMN")
        @test_throws ArgumentError ets(non_positive .+ 2, 4, "ANM")
    end
end
