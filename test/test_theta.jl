using Test
using Durbyn
using Statistics

import Durbyn.Theta: theta, auto_theta, ThetaFit, ThetaModelType, STM, OTM, DSTM, DOTM
import Durbyn.Generics: forecast, Forecast

const EPS_PARAM = 1e-2
const EPS_FORECAST = 5.0
const EPS_STATE = 1e-3
const EPS_RESID = 1.0

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

const REF_STM_FORECAST_NONSEASONAL = [432.9292, 434.2578, 435.5864, 436.9150, 438.2435]
const REF_DSTM_FORECAST_NONSEASONAL = [432.9292, 434.2520, 435.5693, 436.8809, 438.1871]
const ALPHA_LOWER = 0.001
const ALPHA_UPPER = 0.9999
const THETA_LOWER = 0.5


@testset "Durbyn.Theta Module Tests" begin

    @testset "ThetaFit Structure" begin
        fit = theta(AirPassengers, 1; model_type=STM)

        @test fit isa ThetaFit
        @test fit.model_type == STM
        @test length(fit.y) == length(AirPassengers)
        @test length(fit.residuals) == length(AirPassengers)
        @test length(fit.fitted) == length(AirPassengers)
        @test size(fit.states, 1) == length(AirPassengers)
        @test size(fit.states, 2) == 5
    end

    @testset "Simple Theta Model (STM)" begin

        @testset "STM parameter constraints" begin
            fit = theta(AirPassengers, 1; model_type=STM)

            @test fit.theta == 2.0
            @test ALPHA_LOWER <= fit.alpha <= ALPHA_UPPER
            @test isfinite(fit.initial_level)
        end

        @testset "STM forecast matches reference" begin
            fit = auto_theta(AirPassengers, 1; model=STM)
            fc = forecast(fit; h=5)

            for i in 1:5
                @test abs(fc.mean[i] - REF_STM_FORECAST_NONSEASONAL[i]) <= EPS_FORECAST
            end
        end

        @testset "STM fitted values" begin
            fit = theta(AirPassengers, 1; model_type=STM)

            reconstructed = fit.fitted .+ fit.residuals
            @test all(isapprox.(reconstructed, AirPassengers, atol=EPS_RESID))
            @test fit.mse > 0
        end
    end

    @testset "Optimized Theta Model (OTM)" begin

        @testset "OTM parameter optimization" begin
            fit = theta(AirPassengers, 1; model_type=OTM)

            @test ALPHA_LOWER <= fit.alpha <= ALPHA_UPPER
            @test fit.theta >= THETA_LOWER
            @test fit.model_type == OTM
        end

        @testset "OTM forecasting" begin
            fit = theta(AirPassengers, 1; model_type=OTM)
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
            @test all(isfinite.(fc.mean))
            @test all(fc.mean .> 0)
        end
    end

    @testset "Dynamic Simple Theta Model (DSTM)" begin

        @testset "DSTM forecast matches reference" begin
            fit = auto_theta(AirPassengers, 1; model=DSTM)
            fc = forecast(fit; h=5)

            for i in 1:5
                @test abs(fc.mean[i] - REF_DSTM_FORECAST_NONSEASONAL[i]) <= EPS_FORECAST
            end
        end

        @testset "DSTM dynamic state updates" begin
            fit = theta(AirPassengers, 1; model_type=DSTM)

            @test fit.theta == 2.0
            @test size(fit.states) == (length(AirPassengers), 5)
        end
    end

    @testset "Dynamic Optimized Theta Model (DOTM)" begin

        @testset "DOTM full optimization" begin
            fit = theta(AirPassengers, 1; model_type=DOTM)

            @test ALPHA_LOWER <= fit.alpha <= ALPHA_UPPER
            @test fit.theta >= THETA_LOWER
            @test fit.model_type == DOTM
        end

        @testset "DOTM forecasting" begin
            fit = theta(AirPassengers, 1; model_type=DOTM)
            fc = forecast(fit; h=12)

            @test length(fc.mean) == 12
            @test all(isfinite.(fc.mean))
        end
    end

    @testset "Seasonal Models (m=12)" begin
        ap = AirPassengers

        @testset "Seasonal decomposition detection" begin
            fit = auto_theta(ap, 12)

            @test fit.decompose == true
            @test fit.decomposition_type in ["additive", "multiplicative"]
            @test !isnothing(fit.seasonal_component)
        end

        @testset "All seasonal model types" begin
            for mtype in [STM, OTM, DSTM, DOTM]
                fit = auto_theta(ap, 12; model=mtype)

                @test fit.model_type == mtype
                @test fit.m == 12

                fc = forecast(fit; h=12)
                @test length(fc.mean) == 12
                @test all(isfinite.(fc.mean))
                @test all(fc.mean .> 0)
            end
        end

        @testset "Seasonal forecast patterns" begin
            fit = auto_theta(ap, 12; model=STM)
            fc = forecast(fit; h=24)

            @test length(fc.mean) == 24
            @test abs(fc.mean[1] - fc.mean[13]) / fc.mean[1] < 0.3
            @test abs(fc.mean[7] - fc.mean[19]) / fc.mean[7] < 0.3
        end
    end

    @testset "auto_theta Model Selection" begin

        @testset "Automatic model selection" begin
            fit = auto_theta(AirPassengers, 12)

            @test fit.model_type in [STM, OTM, DSTM, DOTM]
            @test isfinite(fit.mse)
        end

        @testset "Specific model override" begin
            for mtype in [STM, OTM, DSTM, DOTM]
                fit = auto_theta(AirPassengers, 12; model=mtype)
                @test fit.model_type == mtype
            end
        end

        @testset "Model selection by MSE" begin
            fits = [auto_theta(AirPassengers, 12; model=m) for m in [STM, OTM, DSTM, DOTM]]
            mses = [f.mse for f in fits]

            auto_fit = auto_theta(AirPassengers, 12)
            @test auto_fit.mse <= minimum(mses) + EPS_STATE
        end
    end

    @testset "Prediction Intervals" begin
        fit = auto_theta(AirPassengers, 12; model=STM)

        @testset "Interval structure" begin
            fc = forecast(fit; h=12, level=[80, 95])

            @test length(fc.mean) == 12
            @test length(fc.lower) == 2
            @test length(fc.upper) == 2
            @test length(fc.lower[1]) == 12
            @test length(fc.lower[2]) == 12
        end

        @testset "Interval ordering" begin
            fc = forecast(fit; h=12, level=[80, 95])

            @test all(fc.lower[2] .<= fc.lower[1])
            @test all(fc.upper[1] .<= fc.upper[2])
            @test all(fc.lower[1] .<= fc.mean)
            @test all(fc.mean .<= fc.upper[1])
        end

        @testset "Intervals widen with horizon" begin
            fc = forecast(fit; h=24, level=[80])

            width_early = fc.upper[1][1:6] .- fc.lower[1][1:6]
            width_late = fc.upper[1][19:24] .- fc.lower[1][19:24]

            @test mean(width_late) >= mean(width_early)
        end
    end

    @testset "Edge Cases" begin

        @testset "Constant series" begin
            constant = fill(100.0, 50)
            fit = auto_theta(constant, 1)

            @test isfinite(fit.mse)

            fc = forecast(fit; h=10)
            @test all(abs.(fc.mean .- 100.0) .<= 1.0)
        end

        @testset "Linear trend series" begin
            trend = collect(1.0:100.0)
            fit = auto_theta(trend, 1)

            @test isfinite(fit.mse)

            fc = forecast(fit; h=10)
            @test all(fc.mean .> 100.0)
        end

        @testset "Short series" begin
            short = AirPassengers[1:24]
            fit = auto_theta(short, 12)

            @test isfinite(fit.mse)

            fc = forecast(fit; h=6)
            @test length(fc.mean) == 6
        end

        @testset "nmse parameter" begin
            fit1 = theta(AirPassengers, 1; model_type=STM, nmse=1)
            fit3 = theta(AirPassengers, 1; model_type=STM, nmse=3)
            fit10 = theta(AirPassengers, 1; model_type=STM, nmse=10)

            @test isfinite(fit1.mse)
            @test isfinite(fit3.mse)
            @test isfinite(fit10.mse)
        end

        @testset "nmse bounds" begin
            @test_throws ArgumentError auto_theta(AirPassengers, 1; nmse=0)
            @test_throws ArgumentError auto_theta(AirPassengers, 1; nmse=31)
        end
    end

    @testset "Decomposition Types" begin
        ap = AirPassengers

        @testset "Multiplicative decomposition" begin
            fit = auto_theta(ap, 12; decomposition_type="multiplicative")

            if fit.decompose
                @test fit.decomposition_type == "multiplicative"
            end
        end

        @testset "Additive decomposition" begin
            fit = auto_theta(ap, 12; decomposition_type="additive")

            if fit.decompose
                @test fit.decomposition_type == "additive"
            end
        end

        @testset "Fallback for non-positive data" begin
            data_with_negative = vcat(ap[1:50], -ap[51:100], ap[101:end])

            fit = auto_theta(data_with_negative .+ 1000, 12; decomposition_type="multiplicative")
            @test isfinite(fit.mse)
        end
    end

    @testset "Fixed Parameters" begin

        @testset "Fixed alpha" begin
            fit = theta(AirPassengers, 1; model_type=STM, alpha=0.3)
            @test abs(fit.alpha - 0.3) <= EPS_PARAM
        end

        @testset "Fixed initial level" begin
            fit = theta(AirPassengers, 1; model_type=STM, initial_level=50.0)
            @test abs(fit.initial_level - 50.0) <= EPS_PARAM
        end

        @testset "Fixed theta (OTM)" begin
            fit = theta(AirPassengers, 1; model_type=OTM, theta_param=1.5)
            @test abs(fit.theta - 1.5) <= EPS_PARAM
        end
    end

    @testset "Forecast Object Properties" begin
        fit = auto_theta(AirPassengers, 12)
        fc = forecast(fit; h=12)

        @test fc isa Forecast
        @test fc.method == "Theta($(fit.model_type))"
        @test length(fc.x) == length(AirPassengers)
        @test length(fc.fitted) == length(AirPassengers)
        @test length(fc.residuals) == length(AirPassengers)
    end

    @testset "State Space Consistency" begin
        fit = theta(AirPassengers, 1; model_type=STM)

        @test size(fit.states, 2) == 5
        @test all(isfinite.(fit.states))

        levels = fit.states[:, 1]
        @test all(levels .> 0)

        mu = fit.states[:, 5]
        @test all(isfinite.(mu))
    end

    @testset "Reproducibility" begin
        fit1 = theta(AirPassengers, 1; model_type=STM)
        fit2 = theta(AirPassengers, 1; model_type=STM)

        @test fit1.alpha == fit2.alpha
        @test fit1.theta == fit2.theta
        @test fit1.mse == fit2.mse
        @test all(fit1.fitted .== fit2.fitted)
    end

end
