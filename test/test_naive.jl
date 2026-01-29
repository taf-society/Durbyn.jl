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
            @test length(fc.mean) == 10
            @test all(fc.mean .== fit.mu_original)
        end

        @testset "Custom horizon" begin
            fc = forecast(fit, 24)
            @test length(fc.mean) == 24
            @test all(fc.mean .== fit.mu_original)
        end

        @testset "Short horizon" begin
            fc = forecast(fit, 1)
            @test length(fc.mean) == 1
            @test fc.mean[1] == fit.mu_original
        end
    end

    @testset "Prediction Intervals" begin
        fit = meanf(AirPassengers, 12)

        @testset "Default levels" begin
            fc = forecast(fit, 10, [80.0, 95.0])

            @test length(fc.lower) == 2
            @test length(fc.upper) == 2
            @test length(fc.lower[1]) == 10
            @test length(fc.upper[1]) == 10
        end

        @testset "Interval properties" begin
            fc = forecast(fit, 10, [80.0, 95.0])

            @test all(fc.lower[1] .< fc.mean)
            @test all(fc.lower[2] .< fc.mean)
            @test all(fc.upper[1] .> fc.mean)
            @test all(fc.upper[2] .> fc.mean)

            width_80 = fc.upper[1] .- fc.lower[1]
            width_95 = fc.upper[2] .- fc.lower[2]
            @test all(width_95 .> width_80)
        end

        @testset "Symmetric intervals" begin
            fc = forecast(fit, 10, [80.0])

            lower_dist = fc.mean .- fc.lower[1]
            upper_dist = fc.upper[1] .- fc.mean
            @test all(abs.(lower_dist .- upper_dist) .<= EPS_SCALAR)
        end

        @testset "Single level" begin
            fc = forecast(fit, 10, [90.0])
            @test length(fc.lower) == 1
            @test length(fc.upper) == 1
        end
    end

    @testset "Fan mode" begin
        fit = meanf(AirPassengers, 12)
        fc = forecast(fit, 10, [80.0], true)

        @test !isempty(fc.level)
        @test !isempty(fc.lower)
        @test !isempty(fc.upper)
    end

    @testset "Level normalization" begin
        fit = meanf(AirPassengers, 12)

        fc1 = forecast(fit, 10, [0.80, 0.95])
        @test fc1.level == [80.0, 95.0]
    end

    @testset "Box-Cox Transformation" begin
        fit = meanf(AirPassengers, 12, 0.0)

        @test fit.lambda == 0.0
        @test isapprox(fit.mu, mean(log.(AirPassengers)), atol=EPS_SCALAR)

        fc = forecast(fit, 10, [80.0])

        @test all(fc.mean .> 0)
        @test all(fc.lower[1] .> 0)
        @test all(fc.upper[1] .> 0)
    end

    @testset "Simple Test Data" begin
        simple_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        fit = meanf(simple_data, 1)

        @test fit.mu == 3.0
        @test fit.sd ≈ std(simple_data) atol=EPS_SCALAR

        fc = forecast(fit, 3)
        @test all(fc.mean .== 3.0)
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
            @test length(fc.mean) == 5
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

        @test !isempty(fc.mean)
        @test !isempty(fc.lower)
        @test !isempty(fc.upper)
        @test !isempty(fc.level)

        @test fc.model.m == 12
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

        # mu_original is back-transformed, so compare that
        @test all(abs.(fc.mean .- fit.mu_original) .<= EPS_SCALAR)

        n = length(AirPassengers)
        fc_pi = forecast(fit, 12, [80.0, 95.0])

        width_80 = fc_pi.upper[1][1] - fc_pi.lower[1][1]
        @test 300 < width_80 < 320
    end

    @testset "Bootstrap Intervals" begin
        fit = meanf(AirPassengers, 12)

        # Test bootstrap path doesn't error and produces valid intervals
        fc_boot = forecast(fit, 10, [80.0, 95.0], false, true, 1000)

        @test !isempty(fc_boot.mean)
        @test !isempty(fc_boot.lower)
        @test !isempty(fc_boot.upper)

        @test length(fc_boot.mean) == 10
        @test length(fc_boot.lower) == 2
        @test length(fc_boot.lower[1]) == 10

        # Intervals should bracket the mean
        @test all(fc_boot.lower[1] .< fc_boot.mean)
        @test all(fc_boot.upper[1] .> fc_boot.mean)
    end

    @testset "MeanFit Structure Updated" begin
        fit = meanf(AirPassengers, 12)

        # Test new fields exist
        @test hasfield(MeanFit, :x)
        @test hasfield(MeanFit, :mu)
        @test hasfield(MeanFit, :mu_original)
        @test hasfield(MeanFit, :sd)
        @test hasfield(MeanFit, :n)

        @test fit.n == length(AirPassengers)
        @test length(fit.x) == fit.n
    end

    @testset "Box-Cox Scale Consistency" begin
        fit = meanf(AirPassengers, 12, 0.0)

        # mu is on transformed scale (log)
        @test isapprox(fit.mu, mean(log.(AirPassengers)), atol=EPS_SCALAR)

        # sd is on transformed scale
        @test isapprox(fit.sd, std(log.(AirPassengers)), atol=EPS_SCALAR)

        # fitted values are on original scale (back-transformed)
        @test all(fit.fitted .> 0)  # Back-transformed, should be positive
        @test isapprox(fit.fitted[1], fit.mu_original, atol=EPS_SCALAR)

        # R's meanf: residuals stay on TRANSFORMED scale (res = x - fits where x is transformed)
        # residuals[i] = log(y[i]) - mu_trans
        @test isapprox(fit.residuals[1], log(AirPassengers[1]) - fit.mu, atol=EPS_SCALAR)
    end

    @testset "meanf with n == 1" begin
        single_obs = [100.0]
        fit = meanf(single_obs, 1)

        @test fit.n == 1
        @test fit.mu == 100.0
        @test fit.sd == 0.0  # sd of single observation

        # Forecast should work but produce infinite intervals (like R)
        fc = forecast(fit, 5, [80.0, 95.0])
        @test length(fc.mean) == 5
        @test all(fc.mean .== 100.0)

        # Intervals should be infinite (TDist(0) not defined, so we use Inf)
        @test all(fc.lower[1] .== -Inf)
        @test all(fc.upper[1] .== Inf)
    end

    @testset "meanf with n == 1 and lambda < 0" begin
        # With lambda < 0, infinite upper bounds become NaN after inv_box_cox
        # This is correct mathematical behavior - the Box-Cox transform domain is bounded
        single_obs = [100.0]
        fit = meanf(single_obs, 1, -0.5)

        @test fit.n == 1
        @test !isnothing(fit.lambda)
        @test fit.lambda == -0.5

        fc = forecast(fit, 5, [80.0, 95.0])
        @test length(fc.mean) == 5
        @test all(isfinite.(fc.mean))

        # Lower bounds: -Inf on transformed scale → 0 on original scale (for lambda < 0)
        @test all(fc.lower[1] .== 0.0)

        # Upper bounds: +Inf on transformed scale → NaN (exceeds inv_box_cox domain)
        # This is correct behavior matching R's InvBoxCox
        @test all(isnan.(fc.upper[1]))
    end

end

# =============================================================================
# Tests for naive, snaive, rw functions
# =============================================================================

import Durbyn.Naive: naive, snaive, rw, rwf, NaiveFit

@testset "Durbyn.Naive (naive/snaive/rw) Module Tests" begin

    @testset "NaiveFit Structure" begin
        fit = naive(AirPassengers)

        @test fit isa NaiveFit
        @test length(fit.fitted) == length(AirPassengers)
        @test length(fit.residuals) == length(AirPassengers)
        @test fit.lag == 1
        @test fit.method == "Naive method"
    end

    @testset "Naive Fitted Values" begin
        fit = naive(AirPassengers)

        # First fitted value should be missing
        @test ismissing(fit.fitted[1])

        # Fitted values should be lagged original values
        for t in 2:length(AirPassengers)
            @test fit.fitted[t] == AirPassengers[t-1]
        end
    end

    @testset "Naive Residuals" begin
        fit = naive(AirPassengers)

        @test ismissing(fit.residuals[1])
        for t in 2:length(AirPassengers)
            @test fit.residuals[t] ≈ AirPassengers[t] - AirPassengers[t-1] atol=EPS_SCALAR
        end
    end

    @testset "Naive Forecast" begin
        fit = naive(AirPassengers, 12)
        fc = forecast(fit, h=10)

        # All forecasts should equal last observation
        @test all(fc.mean .== AirPassengers[end])

        # Prediction intervals should widen with horizon
        @test fc.lower[1][1] > fc.lower[1][10]  # 80% lower bound
        @test fc.upper[1][1] < fc.upper[1][10]  # 80% upper bound
    end

    @testset "Seasonal Naive (snaive)" begin
        fit = snaive(AirPassengers, 12)

        @test fit isa NaiveFit
        @test fit.lag == 12
        @test fit.m == 12
        @test fit.method == "Seasonal naive method"

        # First m fitted values should be missing
        for t in 1:12
            @test ismissing(fit.fitted[t])
        end

        # Fitted values should be lagged by m
        for t in 13:length(AirPassengers)
            @test fit.fitted[t] == AirPassengers[t-12]
        end
    end

    @testset "Seasonal Naive Forecast" begin
        fit = snaive(AirPassengers, 12)
        fc = forecast(fit, h=24)

        # Forecasts should cycle through last m values
        last_season = AirPassengers[end-11:end]
        for i in 1:12
            @test fc.mean[i] == last_season[i]
            @test fc.mean[i+12] == last_season[i]
        end
    end

    @testset "Random Walk without Drift" begin
        fit = rw(AirPassengers)

        @test fit isa NaiveFit
        @test fit.method == "Random walk method"
        @test isnothing(fit.drift)
        @test isnothing(fit.drift_se)

        # Should be equivalent to naive
        fit_naive = naive(AirPassengers)
        @test fit.fitted[2:end] == fit_naive.fitted[2:end]
    end

    @testset "Random Walk with Drift" begin
        fit = rw(AirPassengers, drift=true)

        @test fit.method == "Random walk with drift"
        @test !isnothing(fit.drift)
        @test !isnothing(fit.drift_se)

        # Drift should be (last - first) / (n - 1)
        n = length(AirPassengers)
        expected_drift = (AirPassengers[n] - AirPassengers[1]) / (n - 1)
        @test fit.drift ≈ expected_drift atol=EPS_SCALAR

        # Forecasts should include drift trend
        fc = forecast(fit, h=10)
        for i in 1:10
            expected = AirPassengers[end] + i * fit.drift
            @test fc.mean[i] ≈ expected atol=EPS_SCALAR
        end
    end

    @testset "rwf alias" begin
        fit1 = rw(AirPassengers, drift=true)
        fit2 = rwf(AirPassengers, drift=true)

        @test fit1.drift == fit2.drift
        @test fit1.sigma2 == fit2.sigma2
    end

    @testset "Minimum Length Error (n == 1)" begin
        single_obs = [100.0]

        @test_throws ArgumentError naive(single_obs)
        @test_throws ArgumentError rw(single_obs)
    end

    @testset "Minimum Length Error for snaive (n <= m)" begin
        short_series = [1.0, 2.0, 3.0]

        @test_throws ArgumentError snaive(short_series, 4)
        @test_throws ArgumentError snaive(short_series, 3)  # n must be > m, not >= m
    end

    @testset "Box-Cox Transformation" begin
        fit = naive(AirPassengers, lambda=0.0)

        @test fit.lambda == 0.0
        @test !isnothing(fit.y_transformed)

        # Transformed data should be log
        @test fit.y_transformed ≈ log.(AirPassengers) atol=EPS_SCALAR

        # Fitted values should be back-transformed
        @test all(skipmissing(fit.fitted) .> 0)

        fc = forecast(fit, h=10)
        @test all(fc.mean .> 0)
        @test all(fc.lower[1] .> 0)
        @test all(fc.upper[1] .> 0)
    end

    @testset "Box-Cox Bias Adjustment" begin
        # Use data with clear trend for noticeable difference
        y = 10.0 .+ collect(1.0:50.0) .+ 0.5 .* randn(50)
        y = abs.(y)  # Ensure positive for Box-Cox

        fit_no_bias = naive(y, lambda=0.5, biasadj=false)
        fit_with_bias = naive(y, lambda=0.5, biasadj=true)

        @test fit_no_bias.biasadj == false
        @test fit_with_bias.biasadj == true

        # Forecasts with bias adjustment should generally be slightly higher
        fc_no_bias = forecast(fit_no_bias, h=10)
        fc_with_bias = forecast(fit_with_bias, h=10)

        # Both should produce valid forecasts
        @test all(isfinite.(fc_no_bias.mean))
        @test all(isfinite.(fc_with_bias.mean))

        # Bias-adjusted forecasts typically differ from non-adjusted
        # (may be higher or lower depending on data, but should differ)
        @test fc_no_bias.mean != fc_with_bias.mean
    end

    @testset "Box-Cox with snaive" begin
        fit = snaive(AirPassengers, 12, lambda=0.0, biasadj=true)

        @test fit.lambda == 0.0
        @test fit.biasadj == true

        fc = forecast(fit, h=24)
        @test all(fc.mean .> 0)
    end

    @testset "Box-Cox with rw drift" begin
        fit = rw(AirPassengers, drift=true, lambda=0.0)

        @test fit.lambda == 0.0
        @test !isnothing(fit.drift)

        fc = forecast(fit, h=12)
        @test all(fc.mean .> 0)
        @test all(isfinite.(fc.mean))
    end

    @testset "Prediction Interval Properties" begin
        fit = naive(AirPassengers, 12)
        fc = forecast(fit, h=12, level=[80, 95])

        # Lower < mean < upper
        @test all(fc.lower[1] .< fc.mean)
        @test all(fc.lower[2] .< fc.mean)
        @test all(fc.mean .< fc.upper[1])
        @test all(fc.mean .< fc.upper[2])

        # 95% interval wider than 80%
        width_80 = fc.upper[1] .- fc.lower[1]
        width_95 = fc.upper[2] .- fc.lower[2]
        @test all(width_95 .> width_80)

        # Intervals widen with horizon (naive variance = h * sigma2)
        @test width_80[1] < width_80[12]
        @test width_95[1] < width_95[12]
    end

    @testset "Seasonal Naive Prediction Intervals" begin
        fit = snaive(AirPassengers, 12)
        fc = forecast(fit, h=24, level=[80, 95])

        # Intervals should step up at seasonal boundaries
        width_80 = fc.upper[1] .- fc.lower[1]

        # Within first season (h=1 to 12), widths should be constant
        @test all(width_80[1:12] .≈ width_80[1])

        # Second season (h=13 to 24) should have wider intervals
        @test width_80[13] > width_80[12]
        @test all(width_80[13:24] .≈ width_80[13])
    end

    @testset "Constant Series" begin
        constant = fill(10.0, 50)
        fit = naive(constant)

        @test fit.sigma2 == 0.0
        @test all(skipmissing(fit.residuals) .== 0.0)

        fc = forecast(fit, h=10)
        @test all(fc.mean .== 10.0)
    end

    @testset "Reproducibility" begin
        fit1 = naive(AirPassengers, 12)
        fit2 = naive(AirPassengers, 12)

        @test fit1.sigma2 == fit2.sigma2
        @test all(skipmissing(fit1.fitted) .== skipmissing(fit2.fitted))
    end

    @testset "Missing Values Support" begin
        @testset "naive with missing values" begin
            y_with_missing = Union{Float64, Missing}[1.0, 2.0, missing, 4.0, 5.0, missing, 7.0, 8.0]
            fit = naive(y_with_missing)

            @test fit isa NaiveFit
            @test length(fit.x) == 8
            @test isnan(fit.x[3])  # missing converted to NaN
            @test isnan(fit.x[6])

            # R's lagwalk fill strategy: only fill lagged values at positions where y is NA
            # y[4] is NOT NA, so lagged[4] = y[3] = NA stays missing
            @test ismissing(fit.fitted[4])
            # Fitted at t=5 should use t=4 which is 4.0
            @test fit.fitted[5] == 4.0
            # y[7] is NOT NA, so lagged[7] = y[6] = NA stays missing
            @test ismissing(fit.fitted[7])

            fc = forecast(fit, h=5)
            @test all(isfinite.(fc.mean))
        end

        @testset "snaive with missing values" begin
            y_with_missing = Union{Float64, Missing}[missing, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            fit = snaive(y_with_missing, 4)

            @test fit isa NaiveFit
            # First 4 fitted values are missing (lag period)
            @test all(ismissing.(fit.fitted[1:4]))
            # R's fill strategy: fill lagged at positions where y is NA
            # y[5] is NOT NA, so lagged[5] = y[1] = NA stays missing
            @test ismissing(fit.fitted[5])
            # Fitted at t=6 uses t=2 which is 2.0
            @test fit.fitted[6] == 2.0
        end

        @testset "rw with missing values" begin
            y_with_missing = Union{Float64, Missing}[1.0, missing, 3.0, 4.0, 5.0]
            fit = rw(y_with_missing, drift=true)

            @test fit isa NaiveFit
            @test !isnothing(fit.drift)
            # Drift computed from first to last non-missing
            @test isfinite(fit.drift)

            fc = forecast(fit, h=5)
            @test all(isfinite.(fc.mean))
        end

        @testset "Error on insufficient non-missing data" begin
            y_mostly_missing = Union{Float64, Missing}[missing, missing, 1.0, missing]
            @test_throws ArgumentError naive(y_mostly_missing)  # only 1 non-missing

            y_snaive_insufficient = Union{Float64, Missing}[1.0, 2.0, missing, missing, missing]
            @test_throws ArgumentError snaive(y_snaive_insufficient, 4)  # 2 non-missing <= m=4
        end
    end

    @testset "Trailing Missing Values" begin
        @testset "naive with trailing missings" begin
            # Last 2 observations are missing
            y = Union{Float64, Missing}[1.0, 2.0, 3.0, 4.0, 5.0, missing, missing]
            fit = naive(y)

            # Forecast should use last non-missing value (5.0)
            fc = forecast(fit, h=5)
            @test all(fc.mean .== 5.0)
            @test all(isfinite.(fc.mean))
        end

        @testset "snaive with trailing missings" begin
            # Seasonal period 3, with missing in last season
            y = Union{Float64, Missing}[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, missing, 9.0]
            fit = snaive(y, 3)

            fc = forecast(fit, h=6)
            # Position 1: should use y[7]=7.0 (or y[4]=4.0 as fallback)
            # Position 2: should use y[8]=missing, fall back to y[5]=5.0
            # Position 3: should use y[9]=9.0
            @test fc.mean[1] == 7.0
            @test fc.mean[2] == 5.0  # Fallback for missing position
            @test fc.mean[3] == 9.0
            @test all(isfinite.(fc.mean))
        end

        @testset "rw with drift and trailing missings" begin
            y = Union{Float64, Missing}[1.0, 2.0, 3.0, 4.0, 5.0, missing, missing]
            fit = rw(y, drift=true)

            # Forecast should use last non-missing value
            fc = forecast(fit, h=5)
            @test all(isfinite.(fc.mean))
            # First forecast should be close to 5.0 + drift
            @test fc.mean[1] > 5.0  # Positive drift expected
        end
    end

    @testset "Single Residual (n_valid == 2)" begin
        @testset "naive with 2 observations" begin
            y = [10.0, 20.0]
            fit = naive(y)

            # Only 1 residual (at t=2), res = 20 - 10 = 10
            # R uses MSE = mean(res^2) = 100, not centered variance (which would be 0)
            @test fit.sigma2 == 100.0

            fc = forecast(fit, h=5)
            @test all(fc.mean .== 20.0)
            # With sigma2 > 0, intervals are now wider (matching R behavior)
            @test all(fc.lower[1] .< fc.mean)
            @test all(fc.upper[1] .> fc.mean)
        end

        @testset "snaive with m+1 observations" begin
            y = [1.0, 2.0, 3.0, 4.0, 5.0]  # m=4, so only 1 residual at t=5
            fit = snaive(y, 4)

            # res = 5 - 1 = 4, MSE = 16
            @test fit.sigma2 == 16.0
            fc = forecast(fit, h=4)
            @test all(isfinite.(fc.mean))
        end

        @testset "rw with drift and 2 observations" begin
            y = [10.0, 20.0]
            fit = rw(y, drift=true)

            # Drift should be (20 - 10) / 1 = 10
            @test fit.drift == 10.0
            # With drift, R uses (n-1) divisor via var(corrected=true)
            # Only 1 residual, so sigma2 = var([0.0], corrected=true) which is NaN or 0
            # But with MSE approach for drift case, it's the variance of residuals
            # residual = 20 - (10 + 10) = 0, so sigma2 = var([0], corrected=true) = 0
            @test fit.sigma2 == 0.0
            # drift_se should also be 0 (or very small)
            @test fit.drift_se == 0.0

            fc = forecast(fit, h=3)
            @test fc.mean[1] ≈ 30.0 atol=EPS_SCALAR  # 20 + 1*10
            @test fc.mean[2] ≈ 40.0 atol=EPS_SCALAR  # 20 + 2*10
            @test fc.mean[3] ≈ 50.0 atol=EPS_SCALAR  # 20 + 3*10
        end
    end

    @testset "Seasonal Position with All NaN Values" begin
        # Test case: all values at position 2 in the season are missing
        # m=3, positions are: 1, 2, 3, 1, 2, 3, 1, 2, 3
        # indices:           1, 2, 3, 4, 5, 6, 7, 8, 9
        # Make position 2 (indices 2, 5, 8) all NaN
        y = Union{Float64, Missing}[1.0, missing, 3.0, 4.0, missing, 6.0, 7.0, missing, 9.0]
        fit = snaive(y, 3)

        fc = forecast(fit, h=6)

        # Position 1 (forecast indices 1, 4): should use y[7]=7.0
        @test fc.mean[1] == 7.0
        @test fc.mean[4] == 7.0

        # Position 2 (forecast indices 2, 5): all values are missing
        # Should return NaN for these positions
        @test isnan(fc.mean[2])
        @test isnan(fc.mean[5])

        # Position 3 (forecast indices 3, 6): should use y[9]=9.0
        @test fc.mean[3] == 9.0
        @test fc.mean[6] == 9.0
    end

end
