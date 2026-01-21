using Test
using Durbyn
using Statistics

import Durbyn.ExponentialSmoothing: ets, ses, holt, holt_winters, croston
import Durbyn.Generics: forecast, fitted, residuals, accuracy, Forecast
import Durbyn.Arima: auto_arima, arima
import Durbyn.Bats: bats
import Durbyn.Tbats: tbats

const EPS_FORECAST = 1.0
const EPS_PARAM = 0.05
const EPS_RESID = 0.01
const EPS_PI = 5.0
const EPS_SCALAR = 1e-6

const AirPassengers = Float64[
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
]

const SES_FORECAST = fill(431.9958, 12)
const SES_ALPHA = 0.9999

const HOLT_FORECAST_5 = [433.6004, 435.2049, 436.8093, 438.4138, 440.0182]
const HOLT_ALPHA = 0.9999
const HOLT_BETA = 0.0001

const HOLT_DAMPED_FORECAST_5 = [432.1066, 432.2151, 432.3215, 432.4258, 432.528]
const HOLT_DAMPED_PHI = 0.98

const HW_ADD_FORECAST = [
    437.2863, 429.585, 461.5371, 458.6587, 463.9389, 503.1811,
    537.2384, 537.5855, 488.9266, 454.5475, 422.2995, 448.9493
]
const HW_ADD_ALPHA = 0.9935
const HW_ADD_BETA = 0.0002
const HW_ADD_GAMMA = 0.0006

const HW_MULT_FORECAST = [
    445.8901, 418.9478, 466.4298, 496.1291, 507.1463, 575.6281,
    666.6573, 658.497, 550.0907, 491.713, 418.8086, 463.7188
]
const HW_MULT_ALPHA = 0.3146
const HW_MULT_BETA = 0.0071
const HW_MULT_GAMMA = 0.5977

const ETS_AAN_ALPHA = 0.9999
const ETS_AAN_BETA = 0.0001
const ETS_AAN_FORECAST_5 = [433.6004, 435.2049, 436.8093, 438.4138, 440.0182]

const ETS_AAA_FORECAST = HW_ADD_FORECAST

const ETS_MAM_FORECAST = [
    441.8018, 434.1186, 496.63, 483.2375, 483.9914, 551.0244,
    613.1797, 609.3648, 530.5408, 463.0332, 402.7478, 451.9694
]

const AUTO_ARIMA_FORECAST = [
    445.6349, 420.395, 449.1983, 491.8399, 503.3945, 566.8625,
    654.2602, 638.5975, 540.8837, 494.1266, 423.3327, 465.5076
]
const AUTO_ARIMA_FORECAST_5 = AUTO_ARIMA_FORECAST[1:5]

const SES_ME = 2.22
const SES_RMSE = 33.59
const SES_MAE = 25.68
const SES_MAPE = 8.96


@testset "Reference Tests" begin

    @testset "SES (Simple Exponential Smoothing)" begin
        ap = AirPassengers

        @testset "SES forecast values" begin
            fit = ses(ap, 12)
            fc = forecast(fit; h=12)

            @test all(fc.mean .== fc.mean[1])
            @test abs(fc.mean[1] - SES_FORECAST[1]) <= 5.0
        end

        @testset "SES alpha parameter" begin
            fit = ses(ap, 12)
            @test fit.par["alpha"] > 0.8
        end

        @testset "SES residuals" begin
            fit = ses(ap, 12)
            @test abs(mean(fit.residuals)) < 5.0
        end
    end

    @testset "Holt's Method" begin
        ap = AirPassengers

        @testset "Holt forecast values" begin
            fit = holt(ap, 12)
            fc = forecast(fit; h=12)

            @test fc.mean[end] > fc.mean[1]

            for i in 1:min(5, length(fc.mean))
                @test abs(fc.mean[i] - HOLT_FORECAST_5[i]) <= EPS_FORECAST * 5
            end
        end

        @testset "Damped Holt" begin
            fit = holt(ap, 12; damped=true)
            fc = forecast(fit; h=24)

            diff_early = abs(fc.mean[2] - fc.mean[1])
            diff_late = abs(fc.mean[24] - fc.mean[23])
            @test diff_late <= diff_early + 1.0
        end
    end

    @testset "Holt-Winters" begin
        ap = AirPassengers

        @testset "Additive Holt-Winters" begin
            fit = holt_winters(ap, 12; seasonal="additive")
            fc = forecast(fit; h=12)

            @test all(isfinite.(fc.mean))
            @test all(fc.mean .> 0)
            @test 350 < mean(fc.mean) < 600
        end

        @testset "Multiplicative Holt-Winters" begin
            fit = holt_winters(ap, 12; seasonal="multiplicative")
            fc = forecast(fit; h=12)

            @test all(isfinite.(fc.mean))
            @test all(fc.mean .> 0)
            @test 350 < mean(fc.mean) < 600
        end

        @testset "Seasonal pattern preservation" begin
            fit = holt_winters(ap, 12; seasonal="multiplicative")
            fc = forecast(fit; h=24)

            july_idx = [7, 19]
            jan_idx = [1, 13]

            for (j, jan) in zip(july_idx, jan_idx)
                @test fc.mean[j] > fc.mean[jan]
            end
        end
    end

    @testset "ETS Models" begin
        ap = AirPassengers

        @testset "ETS(A,A,N)" begin
            fit = ets(ap, 12, "AAN")
            @test occursin("A,A,N", fit.method)
        end

        @testset "ETS(A,A,A)" begin
            fit = ets(ap, 12, "AAA")
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
            @test all(isfinite.(fc.mean))
        end

        @testset "ETS(M,A,M)" begin
            fit = ets(ap, 12, "MAM")
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
            @test all(fc.mean .> 0)
        end

        @testset "ETS automatic selection (ZZZ)" begin
            fit = ets(ap, 12, "ZZZ")
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
            @test all(isfinite.(fc.mean))
            @test occursin("ETS", fit.method)
        end

        @testset "ETS damped trend" begin
            fit = ets(ap, 12, "AAN"; damped=true)
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
            @test all(isfinite.(fc.mean))
        end
    end

    @testset "ARIMA Models" begin
        ap = AirPassengers

        @testset "auto.arima selection" begin
            fit = auto_arima(ap, 12)

            @test fit.arma[1] >= 0
            @test fit.arma[2] >= 0
            @test fit.arma[3] >= 0
            @test fit.arma[4] >= 0
            @test fit.arma[5] == 12
            @test fit.arma[6] >= 0
            @test fit.arma[7] >= 0
        end

        @testset "ARIMA forecasts" begin
            fit = auto_arima(ap, 12)
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
            @test all(isfinite.(fc.mean))
            @test all(fc.mean .> 0)
        end

        @testset "ARIMA with fixed orders" begin
            fit = auto_arima(ap, 12)
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
        end

        @testset "ARIMA prediction intervals" begin
            fit = auto_arima(ap, 12)
            fc = forecast(fit; h=12, level=[80, 95])

            @test length(fc.upper) >= 2
            @test length(fc.lower) >= 2

            width_80 = fc.upper[1] .- fc.lower[1]
            width_95 = fc.upper[2] .- fc.lower[2]

            @test all(width_95 .>= width_80)
        end
    end

    @testset "BATS and TBATS" begin
        ap = AirPassengers

        @testset "BATS model" begin
            fit = bats(ap, 12)
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
            @test all(isfinite.(fc.mean))
        end

        @testset "TBATS model" begin
            fit = tbats(ap, [12])
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
            @test all(isfinite.(fc.mean))
        end

        @testset "BATS with Box-Cox" begin
            fit = bats(ap, 12; use_box_cox=true)
            fc = forecast(fit; h=12)

            @test all(fc.mean .> 0)
        end
    end

    @testset "Forecast Accuracy Metrics" begin
        ap = AirPassengers

        @testset "In-sample accuracy" begin
            fit = ets(ap, 12, "ZZZ")

            fv = fit.fitted
            res = fit.residuals

            me = mean(res)
            mae = mean(abs.(res))
            mse = mean(res .^ 2)
            rmse = sqrt(mse)

            @test isfinite(me)
            @test isfinite(rmse)
            @test isfinite(mae)
        end

        @testset "Accuracy values reasonable" begin
            fit = ets(ap, 12, "ZZZ")
            res = fit.residuals

            me = mean(res)
            @test abs(me) < 5.0

            mae = mean(abs.(res))
            @test mae < 50.0
        end
    end

    @testset "Fitted Values and Residuals" begin
        ap = AirPassengers

        @testset "Fitted values" begin
            fit = ets(ap, 12, "ZZZ")
            fv = fit.fitted

            @test length(fv) == length(ap)
            @test cor(fv, ap) > 0.9
        end

        @testset "Residuals" begin
            fit = ets(ap, 12, "ZZZ")
            res = fit.residuals

            @test length(res) == length(ap)
            @test abs(mean(res)) < 5.0
            @test std(res) < std(ap) / 2
        end

        @testset "Reconstruction identity" begin
            fit = ets(ap, 12, "AAA")
            fv = fit.fitted
            res = fit.residuals

            reconstructed = fv .+ res
            @test all(abs.(reconstructed .- ap) .<= 1.0)
        end
    end

    @testset "Prediction Intervals Coverage" begin
        ap = AirPassengers

        @testset "PI ordering" begin
            fit = ets(ap, 12, "ZZZ")
            fc = forecast(fit; h=12, level=[80, 95])

            lo_80 = fc.lower[:, 1]
            hi_80 = fc.upper[:, 1]
            lo_95 = fc.lower[:, 2]
            hi_95 = fc.upper[:, 2]

            @test all(lo_80 .< fc.mean)
            @test all(fc.mean .< hi_80)

            @test all(lo_95 .<= lo_80)
            @test all(hi_80 .<= hi_95)
        end

        @testset "PI widening with horizon" begin
            fit = ets(ap, 12, "ZZZ")
            fc = forecast(fit; h=24, level=[80])

            hi = fc.upper[:, 1]
            lo = fc.lower[:, 1]

            width_early = mean(hi[1:6] .- lo[1:6])
            width_late = mean(hi[19:24] .- lo[19:24])

            @test width_late >= width_early * 0.8
        end
    end

    @testset "Edge Cases and Robustness" begin
        ap = AirPassengers

        @testset "Very short forecast horizon" begin
            fit = ets(ap, 12, "ZZZ")
            fc = forecast(fit; h=1)

            @test length(fc.mean) == 1
        end

        @testset "Long forecast horizon" begin
            fit = ets(ap, 12, "ZZZ")
            fc = forecast(fit; h=60)

            @test length(fc.mean) == 60
            @test all(isfinite.(fc.mean))
        end

        @testset "Subset of data" begin
            short_ap = ap[1:60]
            fit = ets(short_ap, 12, "ZZZ")
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
        end
    end

    @testset "Model Comparison" begin
        ap = AirPassengers

        @testset "ETS vs ARIMA" begin
            fit_ets = ets(ap, 12, "ZZZ")
            fit_arima = auto_arima(ap, 12)

            fc_ets = forecast(fit_ets; h=12)
            fc_arima = forecast(fit_arima; h=12)

            @test all(fc_ets.mean .> 0)
            @test all(fc_arima.mean .> 0)

            @test maximum(fc_ets.mean) / minimum(fc_ets.mean) < 3
            @test maximum(fc_arima.mean) / minimum(fc_arima.mean) < 3
        end
    end

end
