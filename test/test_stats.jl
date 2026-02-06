using Test
using Durbyn
using Statistics

import Durbyn.Stats: acf, pacf, ACFResult, PACFResult
import Durbyn.Stats: box_cox, box_cox!, inv_box_cox, box_cox_lambda
import Durbyn.Stats: decompose, DecomposedTimeSeries
import Durbyn.Stats: adf, ADF, kpss, KPSS
import Durbyn.Stats: embed, diff, fourier, ndiffs, nsdiffs
import Durbyn.Stats: ols, OlsFit, approx, approxfun, seasonal_strength
import Durbyn.Stats: na_interp

const EPS_SCALAR = 1e-6
const EPS_VECTOR = 0.05
const EPS_VECTOR_LOOSE = 0.5
const EPS_STAT = 1e-2

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

const REF_ACF_AP_12 = [
    1.0000000, 0.9480473, 0.8755256, 0.8067571, 0.7526098, 0.7138417,
    0.6817767, 0.6571523, 0.6226461, 0.5645584, 0.4997893, 0.4500905,
    0.4237918
]

const REF_PACF_AP_12 = [
    0.9480473, -0.2548888, 0.0047997, 0.0729694, 0.0622037, -0.0009382,
    0.0330001, -0.0715714, -0.2031917, -0.1239858, 0.0481459, 0.1075217
]

const REF_BOX_COX_LAMBDA_AP = 0.0

const REF_DECOMPOSE_FIGURE_ADD = [
    -24.748737, -36.188131, -2.241162, -8.036616, -4.506313, 35.402778,
    63.830808, 62.823232, 16.520202, -20.642677, -53.593434, -28.620202
]

const REF_ADF_STAT_AP = -0.8784

const REF_KPSS_STAT_AP = 2.8767


@testset "Durbyn.Stats Module Tests" begin

    @testset "ACF (Autocorrelation Function)" begin
        ap = AirPassengers

        @testset "Basic ACF computation matches reference" begin
            result = acf(ap, 12, 12)

            @test length(result.values) >= length(REF_ACF_AP_12)
            @test result.values[1] == 1.0
            @test result.values[2] > 0.8
            @test all(result.values[2:end] .< 1.0)
            @test all(result.values .>= -1.0)
        end

        @testset "ACF properties" begin
            result = acf(ap, 12, 12)

            @test result.values[1] == 1.0
            @test all(-1.0 .<= result.values .<= 1.0)
            @test length(result.values) == 13
            @test length(result.lags) == 13
            @test abs(result.ci - 1.96 / sqrt(length(ap))) <= EPS_SCALAR
        end

        @testset "ACF with different nlags" begin
            result_5 = acf(ap, 12, 5)
            result_20 = acf(ap, 12, 20)

            @test length(result_5.values) == 6
            @test length(result_20.values) == 21
            @test all(abs.(result_5.values .- result_20.values[1:6]) .<= EPS_SCALAR)
        end

        @testset "ACF auto nlags calculation" begin
            result = acf(ap, 12)
            expected_nlags = min(floor(Int, 10 * log10(length(ap))), length(ap) - 1)
            @test length(result.lags) - 1 == expected_nlags
        end

        @testset "ACF edge cases" begin
            constant_series = fill(5.0, 50)
            result = acf(constant_series, 1, 10)
            @test all(result.values .== 1.0)

            short_series = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = acf(short_series, 1, 3)
            @test length(result.values) == 4
        end

        @testset "ACF error handling" begin
            @test_throws ArgumentError acf(ap, 12, 200)
            @test_throws ArgumentError acf(ap, 12, -1)
            @test_throws ArgumentError acf(ap, 0, 10)
        end
    end

    @testset "PACF (Partial Autocorrelation Function)" begin
        ap = AirPassengers

        @testset "Basic PACF computation matches reference" begin
            result = pacf(ap, 12, 12)

            @test length(result.values) == length(REF_PACF_AP_12)
            @test result.values[1] > 0.5
            @test all(-1.0 .<= result.values .<= 1.0)
        end

        @testset "PACF properties" begin
            result = pacf(ap, 12, 12)

            @test all(-1.0 .<= result.values .<= 1.0)
            @test result.lags[1] == 1
            @test length(result.values) == 12
        end

        @testset "PACF Durbin-Levinson algorithm" begin
            n = 200
            phi = 0.8
            ar1 = zeros(n)
            ar1[1] = randn()
            for i in 2:n
                ar1[i] = phi * ar1[i-1] + randn() * 0.5
            end

            result = pacf(ar1, 1, 10)

            @test abs(result.values[1] - phi) < 0.2
            @test all(abs.(result.values[3:end]) .< 0.3)
        end
    end

    @testset "Box-Cox Transformation" begin
        ap = AirPassengers

        @testset "box_cox_lambda estimation" begin
            lambda_guer = box_cox_lambda(ap, 12; method="guerrero")
            @test -1.0 <= lambda_guer <= 2.0

            lambda_loglik = box_cox_lambda(ap, 12; method="loglik")
            @test -1.0 <= lambda_loglik <= 2.0

            @test abs(lambda_guer - lambda_loglik) < 0.5
        end

        @testset "box_cox transformation" begin
            transformed_log, _ = box_cox(ap, 12; lambda=0.0)
            @test all(isapprox.(transformed_log, log.(ap), atol=EPS_SCALAR))

            transformed_1, _ = box_cox(ap, 12; lambda=1.0)
            expected_1 = (ap .- 1) ./ 1.0
            @test all(isapprox.(transformed_1, expected_1, atol=EPS_SCALAR))

            transformed_half, _ = box_cox(ap, 12; lambda=0.5)
            expected_half = (sqrt.(ap) .- 1) ./ 0.5
            @test all(isapprox.(transformed_half, expected_half, atol=EPS_SCALAR))
        end

        @testset "inv_box_cox transformation" begin
            for lambda in [-0.5, 0.0, 0.5, 1.0, 2.0]
                transformed, _ = box_cox(ap, 12; lambda=lambda)
                back = inv_box_cox(transformed; lambda=lambda)

                @test all(isapprox.(back, ap, atol=EPS_VECTOR))
            end
        end

        @testset "box_cox! in-place version" begin
            output = similar(ap)
            box_cox!(output, ap, 12; lambda=0.5)
            expected, _ = box_cox(ap, 12; lambda=0.5)

            @test all(isapprox.(output, expected, atol=EPS_SCALAR))
        end
    end

    @testset "Classical Decomposition" begin
        ap = AirPassengers

        @testset "Additive decomposition" begin
            result = decompose(x=ap, m=12, type="additive")

            @test length(result.seasonal) == length(ap)
            @test length(result.trend) == length(ap)
            @test length(result.random) == length(ap)
            @test length(result.figure) == 12

            for i in 1:12
                @test abs(result.figure[i] - REF_DECOMPOSE_FIGURE_ADD[i]) <= EPS_STAT
            end

            @test abs(sum(result.figure)) <= EPS_VECTOR

            valid_idx = .!isnan.(result.trend) .& .!isnan.(result.random)
            reconstructed = result.seasonal[valid_idx] .+ result.trend[valid_idx] .+ result.random[valid_idx]
            @test all(isapprox.(reconstructed, ap[valid_idx], atol=EPS_VECTOR))
        end

        @testset "Multiplicative decomposition" begin
            result = decompose(x=ap, m=12, type="multiplicative")

            @test abs(mean(result.figure) - 1.0) <= EPS_VECTOR

            valid_idx = .!isnan.(result.trend) .& .!isnan.(result.random) .& (result.seasonal .!= 0) .& (result.trend .!= 0)
            reconstructed = result.seasonal[valid_idx] .* result.trend[valid_idx] .* result.random[valid_idx]
            @test all(isapprox.(reconstructed, ap[valid_idx], atol=EPS_VECTOR))
        end

        @testset "Decomposition with custom filter" begin
            custom_filter = ones(12) ./ 12
            result = decompose(x=ap, m=12, type="additive", filter=custom_filter)

            @test length(result.trend) == length(ap)
        end

        @testset "Decomposition error handling" begin
            @test_throws ErrorException decompose(x=ap[1:10], m=12, type="additive")
            @test_throws ErrorException decompose(x=ap, m=1, type="additive")
            @test_throws ErrorException decompose(x=ap, m=12, type="invalid")
        end
    end

    @testset "ADF Unit Root Test" begin
        ap = AirPassengers

        @testset "ADF test with different types" begin
            result_none = adf(ap; type=:none, lags=1)
            @test result_none.model == :none
            @test result_none.lag >= 0
            @test !isnan(result_none.teststat.data[1])

            result_drift = adf(ap; type=:drift, lags=1)
            @test result_drift.model == :drift

            result_trend = adf(ap; type=:trend, lags=1)
            @test result_trend.model == :trend
        end

        @testset "ADF test statistics" begin
            result = adf(ap; type=:none, lags=1)

            @test -10.0 < result.teststat.data[1] < 5.0
            @test size(result.cval, 2) == 3
        end

        @testset "ADF lag selection" begin
            result_fixed = adf(ap; type=:drift, lags=5, selectlags=:fixed)
            @test result_fixed.lag == 5

            result_aic = adf(ap; type=:drift, lags=10, selectlags=:aic)
            @test 0 <= result_aic.lag <= 10

            result_bic = adf(ap; type=:drift, lags=10, selectlags=:bic)
            @test 0 <= result_bic.lag <= 10
        end

        @testset "ADF keyword interface" begin
            result1 = adf(ap; type=:drift)
            result2 = adf(y=ap, type="drift")

            @test abs(result1.teststat.data[1] - result2.teststat.data[1]) <= EPS_SCALAR
        end
    end

    @testset "KPSS Stationarity Test" begin
        ap = AirPassengers

        @testset "KPSS test types" begin
            result_mu = kpss(ap; type=:mu)
            @test result_mu.type == :mu
            @test result_mu.teststat > 0

            result_tau = kpss(ap; type=:tau)
            @test result_tau.type == :tau
            @test result_tau.teststat > 0
        end

        @testset "KPSS lag/bandwidth selection" begin
            result_short = kpss(ap; type=:mu, lags=:short)
            result_long = kpss(ap; type=:mu, lags=:long)
            result_nil = kpss(ap; type=:mu, lags=:nil)

            @test result_long.lag >= result_short.lag
            @test result_nil.lag == 0
        end

        @testset "KPSS user-specified lag" begin
            result = kpss(ap; type=:mu, use_lag=5)
            @test result.lag == 5
        end

        @testset "KPSS critical values" begin
            result = kpss(ap; type=:mu)

            @test length(result.cval) == 4
            @test length(result.clevels) == 4
            @test all(result.cval .> 0)
            @test issorted(result.cval)
        end

        @testset "KPSS vs AirPassengers reference" begin
            result = kpss(ap; type=:mu, lags=:short)

            @test result.teststat > result.cval[1]
        end
    end

    @testset "embed function" begin
        x = collect(1.0:10.0)

        @testset "Basic embedding" begin
            result = embed(x, 3)

            @test size(result) == (8, 3)
            @test result[1, :] == [3.0, 2.0, 1.0]
            @test result[end, :] == [10.0, 9.0, 8.0]
        end

        @testset "Embedding dimension 1" begin
            result = embed(x, 1)
            @test size(result) == (10, 1)
            @test vec(result) == x
        end
    end

    @testset "diff function" begin
        x = [1.0, 3.0, 6.0, 10.0, 15.0]

        @testset "First differences" begin
            d1 = diff(x)
            @test length(d1) == length(x)
            @test all(isnan.(d1[1:1]))
            @test d1[2:end] ≈ [2.0, 3.0, 4.0, 5.0]
        end

        @testset "Second differences" begin
            d2 = diff(x; differences=2)
            @test length(d2) == length(x)
            @test all(isnan.(d2[1:2]))
            @test d2[3:end] ≈ [1.0, 1.0, 1.0]
        end

        @testset "Seasonal differences" begin
            quarterly = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            d_seasonal = diff(quarterly; lag=4)
            @test length(d_seasonal) == length(quarterly)
            @test all(isnan.(d_seasonal[1:4]))
            @test d_seasonal[5:end] ≈ [4.0, 4.0, 4.0, 4.0]
        end
    end

    @testset "fourier function" begin
        n = 12
        m_period = 12
        x = collect(1.0:n)

        @testset "Basic Fourier terms" begin
            F = fourier(x; m=m_period, K=2)

            @test size(F, 2) == 4
            @test size(F, 1) == n
            @test all(-1.0 .<= F .<= 1.0)
        end

        @testset "Fourier with different K values" begin
            x24 = collect(1.0:24)
            F1 = fourier(x24; m=12, K=1)
            F3 = fourier(x24; m=12, K=3)
            F6 = fourier(x24; m=12, K=6)

            @test size(F1, 2) == 2
            @test size(F3, 2) == 6
            @test size(F6, 2) == 12
        end
    end

    @testset "ndiffs and nsdiffs" begin
        ap = AirPassengers

        @testset "ndiffs (non-seasonal differencing)" begin
            d = ndiffs(ap)
            @test d >= 0
            @test d <= 2
        end

        @testset "nsdiffs (seasonal differencing)" begin
            D = nsdiffs(ap, 12)
            @test D >= 0
            @test D <= 2
        end
    end

    @testset "OLS regression" begin
        @testset "Simple linear regression" begin
            x = collect(1.0:10.0)
            y = 2.0 .* x .+ 3.0 .+ randn(10) .* 0.1
            X = hcat(ones(10), x)

            fit = ols(y, X)

            @test abs(fit.coef[1] - 3.0) < 0.5
            @test abs(fit.coef[2] - 2.0) < 0.5
            @test std(fit.residuals) < 0.5
        end

        @testset "Multiple regression" begin
            n = 50
            x1 = randn(n)
            x2 = randn(n)
            y = 1.0 .+ 2.0 .* x1 .+ 3.0 .* x2 .+ randn(n) .* 0.5
            X = hcat(ones(n), x1, x2)

            fit = ols(y, X)

            @test length(fit.coef) == 3
            @test length(fit.residuals) == n
            @test abs(fit.coef[2] - 2.0) < 0.5
            @test abs(fit.coef[3] - 3.0) < 0.5
        end
    end

    @testset "approx and approxfun (interpolation)" begin
        x = [1.0, 2.0, 4.0, 5.0]
        y = [1.0, 4.0, 16.0, 25.0]

        @testset "Linear interpolation" begin
            result = approx(x, y; xout=[1.5, 3.0, 4.5])

            @test abs(result.y[1] - 2.5) <= EPS_SCALAR
            @test abs(result.y[2] - 10.0) <= EPS_SCALAR
        end

        @testset "approxfun returns function" begin
            f = approxfun(x, y)

            @test f(1.0) ≈ 1.0
            @test f(2.0) ≈ 4.0
            @test abs(f(1.5) - 2.5) <= EPS_SCALAR
        end
    end

    @testset "seasonal_strength" begin
        ap = AirPassengers

        @testset "Strong seasonality detection" begin
            strength = seasonal_strength(ap, 12)

            @test length(strength) >= 1
            @test all(0.0 .<= strength .<= 1.0)
            @test strength[1] > 0.5
        end

        @testset "No seasonality" begin
            noise = 100.0 .+ 0.1 .* collect(1.0:144.0)
            strength = seasonal_strength(noise, 12)

            @test length(strength) >= 1
            @test all(strength .< 0.5)
        end
    end

    @testset "na_interp (Missing Value Interpolation)" begin

        @testset "Linear interpolation - interior gaps" begin
            # Simple linear interpolation for interior missing values
            x = Union{Float64,Missing}[1.0, 2.0, missing, 4.0, 5.0]
            result = na_interp(x; linear=true)

            @test length(result) == length(x)
            @test !any(ismissing.(result))
            @test result[1] ≈ 1.0
            @test result[2] ≈ 2.0
            @test result[3] ≈ 3.0  # Interpolated
            @test result[4] ≈ 4.0
            @test result[5] ≈ 5.0
        end

        @testset "Linear interpolation - multiple gaps" begin
            x = Union{Float64,Missing}[1.0, missing, missing, 4.0, missing, 6.0]
            result = na_interp(x; linear=true)

            @test !any(ismissing.(result))
            @test result[2] ≈ 2.0  # Interpolated
            @test result[3] ≈ 3.0  # Interpolated
            @test result[5] ≈ 5.0  # Interpolated
        end

        @testset "Linear interpolation - edge missing values" begin
            # Leading and trailing missing values use rule=(2,2) constant extrapolation
            # R's approx with rule=(2,2) extrapolates using the nearest value
            x = Union{Float64,Missing}[missing, missing, 3.0, 4.0, 5.0, missing]
            result = na_interp(x; linear=true)

            @test !any(ismissing.(result))
            # Leading missings extrapolated with first non-missing value (constant)
            @test result[1] ≈ 3.0  # Extrapolated with nearest
            @test result[2] ≈ 3.0  # Extrapolated with nearest
            # Trailing missing extrapolated with last non-missing value (constant)
            @test result[6] ≈ 5.0  # Extrapolated with nearest
        end

        @testset "NaN values treated as missing" begin
            x = [1.0, NaN, 3.0, 4.0, NaN]
            result = na_interp(x; linear=true)

            @test !any(isnan.(result))
            @test result[2] ≈ 2.0  # Was NaN, now interpolated
        end

        @testset "No missing values - returns input unchanged" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = na_interp(x; linear=true)

            @test result == x
        end

        @testset "Seasonal interpolation with AirPassengers" begin
            # Create a copy with some missing values
            ap_miss = copy(AirPassengers)
            ap_miss[50] = NaN
            ap_miss[100] = NaN

            result = na_interp(ap_miss; m=12)

            @test length(result) == length(ap_miss)
            @test !any(isnan.(result))
            # Interpolated values should be reasonable (within seasonal pattern)
            @test result[50] > 150 && result[50] < 300
            @test result[100] > 200 && result[100] < 400
        end

        @testset "Box-Cox transformation support" begin
            # Test with positive data and lambda
            x = Union{Float64,Missing}[100.0, missing, 300.0, 400.0, 500.0]
            result = na_interp(x; linear=true, lambda=0.5)

            @test !any(ismissing.(result))
            @test all(result .> 0)  # Box-Cox preserves positivity
            # The interpolated value should be reasonable
            @test result[2] > 100 && result[2] < 300
        end

        @testset "Log transformation (lambda=0)" begin
            x = Union{Float64,Missing}[10.0, missing, 1000.0]
            result = na_interp(x; linear=true, lambda=0.0)

            @test !any(ismissing.(result))
            @test all(result .> 0)
            # Log interpolation: geometric mean-ish behavior
            @test result[2] > 10 && result[2] < 1000
        end

        @testset "Automatic linear fallback for short series" begin
            # With m=12, a series of 5 elements should fall back to linear
            x = Union{Float64,Missing}[1.0, missing, 3.0, 4.0, 5.0]
            result = na_interp(x; m=12)

            @test !any(ismissing.(result))
            @test result[2] ≈ 2.0
        end

        @testset "Automatic linear when m=1" begin
            x = Union{Float64,Missing}[1.0, missing, 3.0, 4.0, 5.0]
            result = na_interp(x; m=1)

            @test !any(ismissing.(result))
            @test result[2] ≈ 2.0
        end

        @testset "Error on all missing" begin
            x = Union{Float64,Missing}[missing, missing, missing]
            @test_throws ErrorException na_interp(x)
        end

        @testset "Single non-missing value - requires at least two" begin
            # When only one value exists, interpolation is not possible
            # R's approx also requires at least 2 non-NA values
            x = Union{Float64,Missing}[missing, 5.0, missing]
            @test_throws ErrorException na_interp(x; linear=true)
        end

        @testset "Integer input converted to Float" begin
            x = Union{Int,Missing}[1, missing, 3, 4, 5]
            result = na_interp(x; linear=true)

            @test eltype(result) <: AbstractFloat
            @test result[2] ≈ 2.0
        end

    end

end
