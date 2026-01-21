using Test
using Durbyn
using Statistics

import Durbyn.Naive: meanf, MeanFit, forecast
import Durbyn.Stats: box_cox, inv_box_cox

const EPS_SCALAR = 1e-4
const EPS_PI = 1e-2

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

const REFERENCE_MEAN_AP = 280.2986
const REFERENCE_SD_AP = 119.9663


@testset "Durbyn.Naive (meanf) Module Tests" begin

    @testset "MeanFit Structure" begin
        fit = meanf(AirPassengers, 12)

        @test fit isa MeanFit
        @test length(fit.fitted) == length(AirPassengers)
        @test length(fit.residuals) == length(AirPassengers)
        @test fit.m == 12
    end

    @testset "Mean Calculation" begin
        fit = meanf(AirPassengers, 12)

        @test abs(fit.mu - REFERENCE_MEAN_AP) <= EPS_SCALAR
        @test abs(fit.sd - REFERENCE_SD_AP) <= 0.01
    end

    @testset "Fitted Values" begin
        fit = meanf(AirPassengers, 12)
        @test all(fit.fitted .== fit.mu)
    end

    @testset "Residuals" begin
        fit = meanf(AirPassengers, 12)

        expected_residuals = AirPassengers .- fit.mu
        @test all(abs.(fit.residuals .- expected_residuals) .<= EPS_SCALAR)
        @test abs(sum(fit.residuals)) <= EPS_SCALAR * length(fit.residuals)
    end

    @testset "Forecast Point Estimates" begin
        fit = meanf(AirPassengers, 12)

        @testset "Default horizon" begin
            fc = forecast(fit)
            @test length(fc["mean"]) == 10
            @test all(fc["mean"] .== fit.mu)
        end

        @testset "Custom horizon" begin
            fc = forecast(fit, 24)
            @test length(fc["mean"]) == 24
            @test all(fc["mean"] .== fit.mu)
        end

        @testset "Short horizon" begin
            fc = forecast(fit, 1)
            @test length(fc["mean"]) == 1
            @test fc["mean"][1] == fit.mu
        end
    end

    @testset "Prediction Intervals" begin
        fit = meanf(AirPassengers, 12)

        @testset "Default levels" begin
            fc = forecast(fit, 10, [80.0, 95.0])

            @test size(fc["lower"]) == (10, 2)
            @test size(fc["upper"]) == (10, 2)
        end

        @testset "Interval properties" begin
            fc = forecast(fit, 10, [80.0, 95.0])

            @test all(fc["lower"][:, 1] .< fc["mean"])
            @test all(fc["lower"][:, 2] .< fc["mean"])
            @test all(fc["upper"][:, 1] .> fc["mean"])
            @test all(fc["upper"][:, 2] .> fc["mean"])

            width_80 = fc["upper"][:, 1] .- fc["lower"][:, 1]
            width_95 = fc["upper"][:, 2] .- fc["lower"][:, 2]
            @test all(width_95 .> width_80)
        end

        @testset "Symmetric intervals" begin
            fc = forecast(fit, 10, [80.0])

            lower_dist = fc["mean"] .- fc["lower"][:, 1]
            upper_dist = fc["upper"][:, 1] .- fc["mean"]
            @test all(abs.(lower_dist .- upper_dist) .<= EPS_SCALAR)
        end

        @testset "Single level" begin
            fc = forecast(fit, 10, [90.0])
            @test size(fc["lower"]) == (10, 1)
            @test size(fc["upper"]) == (10, 1)
        end
    end

    @testset "Fan mode" begin
        fit = meanf(AirPassengers, 12)
        fc = forecast(fit, 10, [80.0], true)

        @test haskey(fc, "level")
        @test haskey(fc, "lower")
        @test haskey(fc, "upper")
    end

    @testset "Level normalization" begin
        fit = meanf(AirPassengers, 12)

        fc1 = forecast(fit, 10, [0.80, 0.95])
        @test fc1["level"] == [80.0, 95.0]
    end

    @testset "Box-Cox Transformation" begin
        fit = meanf(AirPassengers, 12, 0.0)

        @test fit.lambda == 0.0
        @test isapprox(fit.mu, mean(log.(AirPassengers)), atol=EPS_SCALAR)

        fc = forecast(fit, 10, [80.0])

        @test all(fc["mean"] .> 0)
        @test all(fc["lower"] .> 0)
        @test all(fc["upper"] .> 0)
    end

    @testset "Simple Test Data" begin
        simple_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        fit = meanf(simple_data, 1)

        @test fit.mu == 3.0
        @test fit.sd ≈ std(simple_data) atol=EPS_SCALAR

        fc = forecast(fit, 3)
        @test all(fc["mean"] .== 3.0)
    end

    @testset "Constant Series" begin
        constant = fill(10.0, 50)
        fit = meanf(constant, 1)

        @test fit.mu == 10.0
        @test fit.sd == 0.0
        @test all(fit.residuals .== 0.0)
    end

    @testset "Edge Cases" begin
        @testset "Short series" begin
            short = [1.0, 2.0, 3.0]
            fit = meanf(short, 1)
            @test fit.mu == 2.0

            fc = forecast(fit, 5)
            @test length(fc["mean"]) == 5
        end

        @testset "Different frequencies" begin
            data = AirPassengers[1:48]

            fit_1 = meanf(data, 1)
            fit_4 = meanf(data, 4)
            fit_12 = meanf(data, 12)

            @test abs(fit_1.mu - fit_4.mu) <= EPS_SCALAR
            @test abs(fit_1.mu - fit_12.mu) <= EPS_SCALAR
        end
    end

    @testset "Forecast Output Structure" begin
        fit = meanf(AirPassengers, 12)
        fc = forecast(fit, 10, [80.0, 95.0])

        @test haskey(fc, "mean")
        @test haskey(fc, "lower")
        @test haskey(fc, "upper")
        @test haskey(fc, "level")
        @test haskey(fc, "m")

        @test fc["m"] == 12
    end

    @testset "Reproducibility" begin
        fit1 = meanf(AirPassengers, 12)
        fit2 = meanf(AirPassengers, 12)

        @test fit1.mu == fit2.mu
        @test fit1.sd == fit2.sd
        @test all(fit1.fitted .== fit2.fitted)
        @test all(fit1.residuals .== fit2.residuals)
    end

    @testset "Missing Values Handling" begin
        data = collect(skipmissing(AirPassengers))
        fit = meanf(data, 12)

        @test isfinite(fit.mu)
        @test isfinite(fit.sd)
    end

    @testset "Reference Comparison" begin
        fit = meanf(AirPassengers, 12)
        fc = forecast(fit, 12)

        @test all(abs.(fc["mean"] .- REFERENCE_MEAN_AP) .<= EPS_SCALAR)

        n = length(AirPassengers)
        fc_pi = forecast(fit, 12, [80.0, 95.0])

        width_80 = fc_pi["upper"][1, 1] - fc_pi["lower"][1, 1]
        @test 300 < width_80 < 320
    end

end
