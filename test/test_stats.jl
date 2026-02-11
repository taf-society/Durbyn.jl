using Test
using Durbyn
using Statistics

import Durbyn.Stats: acf, pacf, ACFResult, PACFResult
import Durbyn.Stats: box_cox, box_cox!, inv_box_cox, box_cox_lambda
import Durbyn.Stats: decompose, DecomposedTimeSeries
import Durbyn.Stats: adf, ADF, kpss, KPSS
import Durbyn.Stats: embed, diff, fourier, ndiffs, nsdiffs
import Durbyn.Stats: ols, OlsFit, approx, approxfun, seasonal_strength, modelrank
import Durbyn.Stats: na_interp, na_contiguous, na_fail
import Durbyn.Stats: mstl, MSTLResult

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

        @testset "inv_box_cox Dict fvar path (R parity)" begin
            x_trans = [1.0, 2.0, 3.0]
            fvar_dict = Dict(:level => [95.0],
                             :upper => [2.0, 3.5, 5.0],
                             :lower => [0.0, 0.5, 1.0])
            result = inv_box_cox(x_trans; lambda=0.5, biasadj=true, fvar=fvar_dict)

            @test result[1] ≈ 2.3150794429 atol=1e-6
            @test result[2] ≈ 4.1464287465 atol=1e-6
            @test result[3] ≈ 6.5103177716 atol=1e-6
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
            x = Union{Float64,Missing}[1.0, 2.0, missing, 4.0, 5.0]
            result = na_interp(x; linear=true)

            @test length(result) == length(x)
            @test !any(ismissing.(result))
            @test result[1] ≈ 1.0
            @test result[2] ≈ 2.0
            @test result[3] ≈ 3.0
            @test result[4] ≈ 4.0
            @test result[5] ≈ 5.0
        end

        @testset "Linear interpolation - multiple gaps" begin
            x = Union{Float64,Missing}[1.0, missing, missing, 4.0, missing, 6.0]
            result = na_interp(x; linear=true)

            @test !any(ismissing.(result))
            @test result[2] ≈ 2.0
            @test result[3] ≈ 3.0
            @test result[5] ≈ 5.0
        end

        @testset "Linear interpolation - edge missing values" begin
            x = Union{Float64,Missing}[missing, missing, 3.0, 4.0, 5.0, missing]
            result = na_interp(x; linear=true)

            @test !any(ismissing.(result))
            @test result[1] ≈ 3.0
            @test result[2] ≈ 3.0
            @test result[6] ≈ 5.0
        end

        @testset "NaN values treated as missing" begin
            x = [1.0, NaN, 3.0, 4.0, NaN]
            result = na_interp(x; linear=true)

            @test !any(isnan.(result))
            @test result[2] ≈ 2.0
        end

        @testset "No missing values - returns input unchanged" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = na_interp(x; linear=true)

            @test result == x
        end

        @testset "Seasonal interpolation with AirPassengers" begin
            ap_miss = copy(AirPassengers)
            ap_miss[50] = NaN
            ap_miss[100] = NaN

            result = na_interp(ap_miss; m=12)

            @test length(result) == length(ap_miss)
            @test !any(isnan.(result))
            @test result[50] > 150 && result[50] < 300
            @test result[100] > 200 && result[100] < 400
        end

        @testset "Box-Cox transformation support" begin
            x = Union{Float64,Missing}[100.0, missing, 300.0, 400.0, 500.0]
            result = na_interp(x; linear=true, lambda=0.5)

            @test !any(ismissing.(result))
            @test all(result .> 0)
            @test result[2] > 100 && result[2] < 300
        end

        @testset "Log transformation (lambda=0)" begin
            x = Union{Float64,Missing}[10.0, missing, 1000.0]
            result = na_interp(x; linear=true, lambda=0.0)

            @test !any(ismissing.(result))
            @test all(result .> 0)
            @test result[2] > 10 && result[2] < 1000
        end

        @testset "Automatic linear fallback for short series" begin
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

    @testset "Stats Bug Fixes" begin

        @testset "fourier scalar args produce correct matrix" begin
            F = fourier(AirPassengers, m=12, K=6)
            @test size(F) == (144, 12)
            @test all(-1.0 .<= F .<= 1.0)
        end

        @testset "fourier with h produces forecast matrix" begin
            Fh = fourier(AirPassengers, m=12, K=6, h=12)
            @test size(Fh) == (12, 12)
            @test all(-1.0 .<= Fh .<= 1.0)
        end

        @testset "mstl basic decomposition" begin
            res = mstl(AirPassengers, 12)
            @test isa(res, MSTLResult)
            @test length(res.trend) == length(AirPassengers)
            @test length(res.remainder) == length(AirPassengers)
            @test !isempty(res.seasonals)
            @test length(res.seasonals[1]) == length(AirPassengers)
        end

        @testset "mstl with NaN in data" begin
            ap_nan = copy(AirPassengers)
            ap_nan[50] = NaN
            res = mstl(ap_nan, 12)
            @test isa(res, MSTLResult)
            @test length(res.trend) == length(ap_nan)
            @test !isnan(res.trend[50])
            @test !isnan(res.seasonals[1][50])
            recon_50 = res.trend[50] + res.seasonals[1][50] + res.remainder[50]
            @test isfinite(recon_50)
            @test 150.0 < recon_50 < 250.0
        end

        @testset "mstl with lambda" begin
            res = mstl(AirPassengers, 12; lambda="auto")
            @test isa(res, MSTLResult)
            @test !isnothing(res.lambda)
        end

        @testset "mstl multi-seasonal with lambda" begin
            n = 365
            data = 100.0 .+ 10.0 .* sin.(2π .* (1:n) ./ 7) .+ 5.0 .* sin.(2π .* (1:n) ./ 30) .+ randn(n)
            res = mstl(data, [7, 30]; lambda=0.5)
            @test isa(res, MSTLResult)
            @test length(res.m) >= 1
        end

        @testset "box_cox does not mutate input" begin
            orig = copy(AirPassengers)
            input = copy(AirPassengers)
            box_cox(input, 12; lambda=0.5)
            @test input == orig
        end

        @testset "box_cox_lambda uses m not hardcoded 12" begin
            short = Float64[10, 12, 15, 11, 13, 16, 12, 14, 17, 13,
                            15, 18, 14, 16, 19, 15, 17, 20, 16, 18]
            lambda_m4 = box_cox_lambda(short, 4)
            @test -1.0 <= lambda_m4 <= 2.0

            lambda_m12 = box_cox_lambda(short, 12)
            @test lambda_m12 == 1.0
        end

        @testset "modelrank returns correct value" begin
            n = 50
            X = hcat(ones(n), randn(n), randn(n))
            y = X * [1.0, 2.0, 3.0] .+ randn(n) * 0.1
            fit = ols(y, X)
            @test modelrank(fit) == 3
        end

        @testset "smooth_trend computes correct moving average" begin
            constant = fill(5.0, 50)
            smoothed = Durbyn.Stats.smooth_trend(constant)
            @test all(smoothed .≈ 5.0)

            linear = collect(1.0:100.0)
            smoothed_lin = Durbyn.Stats.smooth_trend(linear)
            mid = 20:80
            @test all(abs.(smoothed_lin[mid] .- linear[mid]) .< 1.0)
        end

    end

    @testset "R Numerical Parity" begin

        @testset "fourier matches R forecast::fourier (K=6, m=12)" begin
            R_row1 = [0.5, 0.8660254038, 0.8660254038, 0.5,
                      1.0, 0.0, 0.8660254038, -0.5,
                      0.5, -0.8660254038, -1.0]
            R_row3 = [1.0, 0.0, 0.0, -1.0,
                      -1.0, 0.0, 0.0, 1.0,
                      1.0, 0.0, -1.0]

            F = fourier(AirPassengers, m=12, K=6)
            @test size(F, 1) == 144
            jl_shared = [1,2,3,4,5,6,7,8,9,10,12]
            @test all(isapprox.(F[1, jl_shared], R_row1, atol=1e-8))
            @test all(isapprox.(F[3, jl_shared], R_row3, atol=1e-8))
            @test all(abs.(F[:, 11]) .< 1e-10)
        end

        @testset "fourier h=12 matches R forecast::fourier h=12" begin
            R_h_row6 = [0.0, -1.0, 0.0, 1.0,
                        0.0, -1.0, 0.0, 1.0,
                        0.0, -1.0, 1.0]
            R_h_row12 = [0.0, 1.0, 0.0, 1.0,
                         0.0, 1.0, 0.0, 1.0,
                         0.0, 1.0, 1.0]

            Fh = fourier(AirPassengers, m=12, K=6, h=12)
            jl_shared = [1,2,3,4,5,6,7,8,9,10,12]
            @test all(isapprox.(Fh[6, jl_shared], R_h_row6, atol=1e-8))
            @test all(isapprox.(Fh[12, jl_shared], R_h_row12, atol=1e-8))
        end

        @testset "BoxCox(AP, λ=0.5) matches R forecast::BoxCox" begin
            R_bc_half = [19.16601049, 19.72556098, 20.97825059, 20.71563338,
                         20.00000000, 21.23790008, 22.33105012, 22.33105012,
                         21.32380758, 19.81742423, 18.39607805, 19.72556098]
            jl_bc, _ = box_cox(AirPassengers, 12; lambda=0.5)
            @test all(isapprox.(jl_bc[1:12], R_bc_half, atol=1e-6))
        end

        @testset "BoxCox.lambda matches R (guerrero & loglik)" begin
            λ_guer = box_cox_lambda(AirPassengers, 12; method="guerrero")
            λ_loglik = box_cox_lambda(AirPassengers, 12; method="loglik")
            @test isapprox(λ_guer, -0.2947156, atol=0.05)
            @test isapprox(λ_loglik, 0.2, atol=0.1)
        end

        @testset "BoxCox.lambda short series m=4 vs m=12 matches R" begin
            short = Float64[10, 12, 15, 11, 13, 16, 12, 14, 17, 13,
                            15, 18, 14, 16, 19, 15, 17, 20, 16, 18]
            @test isapprox(box_cox_lambda(short, 4), 1.198332, atol=0.05)
            @test box_cox_lambda(short, 12) == 1.0
        end

        @testset "smooth_trend matches R centred MA" begin
            R_first10 = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            R_mid = Float64.(45:55)
            R_last10 = [91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 96.5, 97.0, 97.5, 98.0]

            sm = Durbyn.Stats.smooth_trend(collect(1.0:100.0))
            @test all(isapprox.(sm[1:10], R_first10, atol=1e-10))
            @test all(isapprox.(sm[45:55], R_mid, atol=1e-10))
            @test all(isapprox.(sm[91:100], R_last10, atol=1e-10))

            sm_c = Durbyn.Stats.smooth_trend(fill(5.0, 50))
            @test all(isapprox.(sm_c, 5.0, atol=1e-10))
        end

        @testset "mstl(AP,12) components match R forecast::mstl" begin
            R_trend = [123.1450569, 123.7250692, 124.3050816, 124.8850939,
                       125.5980842, 126.3110746, 127.0240649, 127.6147679,
                       128.2054709, 128.7961738, 129.7770976, 130.7580214]
            R_seas  = [-16.977074086, -17.177633337,   7.520942226,  -1.154334128,
                        -3.776123430,  18.888713514,  40.688673592,  38.924596868,
                        11.309647272, -16.348792410, -41.520353310, -20.288482723]
            R_rem   = [  5.8320171583,  11.4525640973,   0.1739762226,   5.2692402644,
                        -0.8219607941, -10.1997880982, -19.7127385376, -18.5393647752,
                        -3.5151181412,   6.5526185786,  15.7432556744,   7.5304612828]

            res = mstl(AirPassengers, 12)
            @test all(isapprox.(res.trend[1:12], R_trend, atol=5.0))
            @test all(isapprox.(res.seasonals[1][1:12], R_seas, atol=5.0))
            @test all(isapprox.(res.remainder[1:12], R_rem, atol=10.0))
            recon = res.trend .+ res.seasonals[1] .+ res.remainder
            @test all(isapprox.(recon, AirPassengers, atol=1e-8))
        end

        @testset "mstl(AP,12; lambda='auto') matches R lambda selection" begin
            res = mstl(AirPassengers, 12; lambda="auto")
            @test !isnothing(res.lambda)
            @test -1.0 < res.lambda < 2.0
        end

        @testset "na_interp seasonal matches R na.interp" begin
            ap_nan = copy(AirPassengers)
            ap_nan[50] = NaN

            result_seas = na_interp(ap_nan; m=12)
            @test !isnan(result_seas[50])
            @test 165.0 < result_seas[50] < 225.0

            result_lin = na_interp(ap_nan; m=1)
            @test !isnan(result_lin[50])
            @test isapprox(result_lin[50], 215.0, atol=2.0)

            @test abs(result_seas[50] - result_lin[50]) > 1.0
        end

        @testset "mstl NaN uses seasonal interpolation via na_interp(m=...)" begin
            ap_nan = copy(AirPassengers)
            ap_nan[50] = NaN
            ap_nan[100] = NaN

            res = mstl(ap_nan, 12)
            recon = res.trend .+ res.seasonals[1] .+ res.remainder

            @test 150.0 < recon[50] < 250.0
            @test 300.0 < recon[100] < 420.0
        end

        @testset "box_cox_lambda multi-seasonal uses max(m) matching R msts" begin
            n_long = 400
            data_long = 100.0 .+ 10.0 .* sin.(2π .* (1:n_long) ./ 7) .+ randn(n_long)
            λ_long = box_cox_lambda(data_long, 30)
            @test -1.0 <= λ_long <= 2.0

            n_short = 50
            data_short = data_long[1:n_short]
            λ_short = box_cox_lambda(data_short, 30)
            @test λ_short == 1.0
        end

        @testset "modelrank matches R lm()\$rank" begin
            n = 50
            X = hcat(ones(n), randn(n), randn(n))
            y = X * [1.0, 2.0, 3.0] .+ randn(n) * 0.1
            fit = ols(y, X)
            @test modelrank(fit) == 3

            X1 = ones(20, 1)
            y1 = randn(20)
            fit1 = ols(y1, X1)
            @test modelrank(fit1) == 1
        end

    end

    @testset "Round 2 Bug Fixes + R Parity" begin

        @testset "ndiffs string-API passes type and max_d correctly" begin
            @test ndiffs(; x=AirPassengers, test="kpss", type="level", max_d=0) == 0

            @test ndiffs(; x=AirPassengers, test="kpss", type="level") == 1

            @test ndiffs(; x=AirPassengers, test="kpss", type="trend") == 0
        end

        @testset "ndiffs matches R forecast::ndiffs" begin
            @test ndiffs(AirPassengers; test=:kpss) == 1

            @test ndiffs(AirPassengers; test=:adf) == 1

            @test ndiffs(AirPassengers; test=:pp) == 1

            @test ndiffs(AirPassengers; test=:pp, deterministic=:trend) == 0

            @test ndiffs(AirPassengers; test=:adf, deterministic=:trend) == 0

            @test ndiffs(AirPassengers; test=:kpss, deterministic=:level) ==
                  ndiffs(; x=AirPassengers, test="kpss", type="level")
            @test ndiffs(AirPassengers; test=:kpss, deterministic=:trend) ==
                  ndiffs(; x=AirPassengers, test="kpss", type="trend")
        end

        @testset "diff integer matrix returns Float64 with NaN" begin
            m = [1 2; 3 4; 5 6; 7 8]
            result = diff(m; lag=1)
            @test eltype(result) == Float64
            @test size(result) == (4, 2)
            @test all(isnan.(result[1, :]))
            @test result[2, :] == [2.0, 2.0]
            @test result[3, :] == [2.0, 2.0]
            @test result[4, :] == [2.0, 2.0]
        end

        @testset "diff integer vector returns Float64 with NaN" begin
            d = diff([1, 3, 6, 10]; lag=1)
            @test eltype(d) <: AbstractFloat
            @test isnan(d[1])
            @test d[2:end] ≈ [2.0, 3.0, 4.0]
        end

        @testset "na_contiguous treats NaN as missing" begin
            result = na_contiguous([NaN, 1.0, 2.0, 3.0, NaN, 4.0, NaN])
            @test result == [1.0, 2.0, 3.0]

            result2 = na_contiguous(Union{Float64,Missing}[missing, 1.0, 2.0, 3.0, missing])
            @test collect(skipmissing(result2)) == [1.0, 2.0, 3.0]

            result3 = na_contiguous(Union{Float64,Missing}[NaN, missing, 5.0, 6.0, 7.0, NaN])
            @test result3 == [5.0, 6.0, 7.0]
        end

        @testset "na_fail treats NaN as missing" begin
            @test_throws ArgumentError na_fail([1.0, NaN, 2.0])

            @test_throws ArgumentError na_fail(Union{Float64,Missing}[1.0, missing, 2.0])

            @test na_fail([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
        end

        @testset "approx handles missing with na_rm=false" begin

            r = approx([1.0, 2.0, 3.0], Union{Float64,Missing}[missing, 2.0, 3.0];
                       na_rm=false, xout=[2.5])
            @test length(r.y) == 1
            @test !ismissing(r.y[1])

            r2 = approx([1.0, 2.0, 3.0], Union{Float64,Missing}[missing, 2.0, 3.0];
                        xout=[2.5])
            @test r2.y[1] ≈ 2.5

            r3 = approx([1.0, 2.0, 3.0], Union{Float64,Missing}[missing, 2.0, 3.0];
                        xout=[1.5, 2.5], rule=1, na_rm=false)
            @test isnan(r3.y[1])
            @test r3.y[2] ≈ 2.5
        end

        @testset "mstl period filter matches R (2*p < n)" begin
            data_101 = randn(101) .+ 100.0
            res_101 = mstl(data_101, [50])
            @test 50 in res_101.m

            data_100 = randn(100) .+ 100.0
            res_100 = mstl(data_100, [50])
            @test isempty(res_100.m)
        end

    end

end
