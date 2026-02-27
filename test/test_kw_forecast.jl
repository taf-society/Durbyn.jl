using Test
using Random: Xoshiro
using Durbyn
using Durbyn.KolmogorovWiener: arma_autocovariance, build_toeplitz_gamma_hat,
    _effective_nlags, _build_prediction_rhs, _wiener_forecast_stationary,
    _wiener_forecast_integrated, _cumulate_forecasts, _cumulate_variance,
    _extract_differencing, _apply_differencing

# ─── Helper: simulate AR(1) ──────────────────────────────────────────

function simulate_ar1(phi, sigma2, n; seed=42)
    rng = Xoshiro(seed)
    y = zeros(n)
    y[1] = randn(rng) * sqrt(sigma2 / (1 - phi^2))
    for t in 2:n
        y[t] = phi * y[t-1] + randn(rng) * sqrt(sigma2)
    end
    return y
end

# ─── Validation ───────────────────────────────────────────────────────

@testset "Input validation" begin
    y = air_passengers()
    r = kolmogorov_wiener(y, :hp; m=12, maxcoef=50)

    @test_throws ArgumentError forecast(r; h=0)
    @test_throws ArgumentError forecast(r; h=-1)
    @test_throws ArgumentError forecast(r; h=5, ridge=-0.1)
    @test_throws ArgumentError forecast(r; h=5, level=[0, 95])
    @test_throws ArgumentError forecast(r; h=5, level=[80, 100])
end

# ─── _extract_differencing ────────────────────────────────────────────

@testset "_extract_differencing" begin
    y = air_passengers()
    r = kolmogorov_wiener(y, :hp; m=12, maxcoef=50)

    d_ns, D_s, s = _extract_differencing(r)
    @test d_ns >= 0
    @test D_s >= 0
    @test s >= 1
    @test d_ns + D_s == r.d
end

# ─── _apply_differencing ─────────────────────────────────────────────

@testset "_apply_differencing" begin
    # Pure nonseasonal d=1
    y = [1.0, 3.0, 6.0, 10.0]
    z = _apply_differencing(y, 1, 0, 1)
    @test z ≈ [2.0, 3.0, 4.0]

    # Pure seasonal D=1, s=2
    y2 = [1.0, 2.0, 4.0, 6.0, 9.0, 12.0]
    z2 = _apply_differencing(y2, 0, 1, 2)
    @test z2 ≈ [3.0, 4.0, 5.0, 6.0]

    # Mixed d=1, D=1, s=2: seasonal first, then nonseasonal
    z3 = _apply_differencing(y2, 1, 1, 2)
    # Seasonal diff: [3, 4, 5, 6], then nonseasonal diff: [1, 1, 1]
    @test z3 ≈ [1.0, 1.0, 1.0]

    # Pure nonseasonal d=2
    y4 = [1.0, 3.0, 6.0, 10.0, 15.0]
    z4 = _apply_differencing(y4, 2, 0, 1)
    @test z4 ≈ [1.0, 1.0, 1.0]

    # D=1, s=3
    y5 = Float64.(1:12)
    z5 = _apply_differencing(y5, 0, 1, 3)
    @test length(z5) == 9
    @test all(z5 .≈ 3.0)
end

# ─── _effective_nlags ─────────────────────────────────────────────────

@testset "_effective_nlags" begin
    # Stationary: T_eff = T
    @test _effective_nlags(100, 0, 0, 1, nothing) == 100
    @test _effective_nlags(100, 0, 0, 1, 50) == 50
    @test _effective_nlags(100, 0, 0, 1, 200) == 100  # clamped

    # Pure nonseasonal d=1: T_eff = T - 1
    @test _effective_nlags(100, 1, 0, 1, nothing) == 99
    @test _effective_nlags(100, 1, 0, 1, 50) == 50

    # Pure nonseasonal d=2: T_eff = T - 2
    @test _effective_nlags(100, 2, 0, 1, nothing) == 98

    # Seasonal D=1, s=12: T_eff = T - 12
    @test _effective_nlags(100, 0, 1, 12, nothing) == 88

    # Mixed d=1, D=1, s=12: T_eff = T - 12 - 1 = 87
    @test _effective_nlags(100, 1, 1, 12, nothing) == 87

    # Too short
    @test_throws ArgumentError _effective_nlags(12, 1, 1, 12, nothing)
    @test_throws ArgumentError _effective_nlags(2, 2, 0, 1, nothing)

    # Invalid nlags
    @test_throws ArgumentError _effective_nlags(100, 0, 0, 1, 0)
end

# ─── _build_prediction_rhs ───────────────────────────────────────────

@testset "_build_prediction_rhs" begin
    gamma = [10.0, 5.0, 3.0, 2.0, 1.0, 0.5]

    g = _build_prediction_rhs(gamma, 1, 3)
    @test g ≈ [5.0, 3.0, 2.0]

    g = _build_prediction_rhs(gamma, 2, 3)
    @test g ≈ [3.0, 2.0, 1.0]

    # Beyond maxlag: zero-padded
    g = _build_prediction_rhs(gamma, 4, 4)
    @test g ≈ [1.0, 0.5, 0.0, 0.0]
end

# ─── White noise forecast ────────────────────────────────────────────

@testset "White noise forecast" begin
    sigma2 = 2.0
    n = 100
    gamma = arma_autocovariance(Float64[], Float64[], sigma2, n + 10)

    y = randn(Xoshiro(123), n) .* sqrt(sigma2)
    p = n

    fc_mean, fc_var = _wiener_forecast_stationary(y, gamma, 5, p, 0.0)

    for j in 1:5
        @test abs(fc_mean[j]) < 2.0
    end

    for j in 1:5
        @test fc_var[j] ≈ sigma2 atol=0.5
    end
end

# ─── AR(1) stationary forecast ───────────────────────────────────────

@testset "AR(1) stationary forecast" begin
    phi = 0.8
    sigma2 = 1.0
    n = 200

    gamma = arma_autocovariance([phi], Float64[], sigma2, n + 20)
    y = simulate_ar1(phi, sigma2, n)
    p = n

    fc_mean, fc_var = _wiener_forecast_stationary(y, gamma, 10, p, 0.0)

    y_T = y[end]
    for j in 1:10
        expected = phi^j * y_T
        @test fc_mean[j] ≈ expected atol=0.5
    end

    for j in 1:10
        expected_mse = sigma2 * (1 - phi^(2j)) / (1 - phi^2)
        @test fc_var[j] ≈ expected_mse atol=0.3
    end
end

# ─── Cumulation: d=1 random walk (nonseasonal) ───────────────────────

@testset "Random walk (d=1) forecast" begin
    sigma2 = 1.0
    n = 100
    rng = Xoshiro(42)
    y = cumsum(randn(rng, n) .* sqrt(sigma2))

    gamma = arma_autocovariance(Float64[], Float64[], sigma2, n + 20)
    p = n - 1

    # d_ns=1, D_s=0, s=1
    fc_mean, fc_var = _wiener_forecast_integrated(y, gamma, 1, 0, 1, 10, p, 0.0)

    for j in 1:10
        @test fc_mean[j] ≈ y[end] atol=2.0
    end

    for j in 1:10
        @test fc_var[j] ≈ j * sigma2 atol=1.0
    end
end

# ─── _cumulate_forecasts (nonseasonal) ───────────────────────────────

@testset "_cumulate_forecasts nonseasonal" begin
    # d_ns=1: fc[j] = y[T] + cumsum(diff_fc)[j]
    y = [10.0, 12.0, 15.0, 14.0, 16.0]
    diff_fc = [1.0, 2.0, -0.5]

    fc = _cumulate_forecasts(y, diff_fc, 1, 0, 1)
    @test fc[1] ≈ 16.0 + 1.0
    @test fc[2] ≈ 16.0 + 1.0 + 2.0
    @test fc[3] ≈ 16.0 + 1.0 + 2.0 - 0.5

    # d_ns=2: double cumulation
    y2 = [1.0, 3.0, 6.0, 10.0, 15.0]
    diff2_fc = [1.0, 1.0]

    fc2 = _cumulate_forecasts(y2, diff2_fc, 2, 0, 1)
    # First diff of y2 = [2, 3, 4, 5]
    # Cumulate once: anchor = 5, step1 = [6, 7]
    # Cumulate twice: anchor = 15, fc2 = [21, 28]
    @test fc2[1] ≈ 21.0
    @test fc2[2] ≈ 28.0
end

# ─── _cumulate_forecasts (seasonal) ──────────────────────────────────

@testset "_cumulate_forecasts seasonal" begin
    # Linear trend y = 1:24, d_ns=1, D_s=1, s=12
    y = Float64.(1:24)
    diff_fc = [0.0, 0.0, 0.0]

    fc = _cumulate_forecasts(y, diff_fc, 1, 1, 12)
    # Seasonal diff: y[13:24] - y[1:12] = all 12s → w = [12, 12, ..., 12] (len 12)
    # Nonseasonal diff of w: all 0s → z = [0, ..., 0] (len 11)
    # Undo nonseasonal: anchor = w[end] = 12, fc_w = cumsum([0,0,0]) + 12 = [12, 12, 12]
    # Undo seasonal: fc_y[j] = fc_w[j] + y[24 + j - 12] = 12 + y[12+j]
    # fc_y[1] = 12 + y[13] = 12 + 13 = 25
    # fc_y[2] = 12 + y[14] = 12 + 14 = 26
    # fc_y[3] = 12 + y[15] = 12 + 15 = 27
    @test fc ≈ [25.0, 26.0, 27.0]

    # Pure seasonal D=1, s=4, no nonseasonal diff
    y3 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    # Seasonal diff: [5-1, 6-2, 7-3, 8-4] = [4, 4, 4, 4]
    # Forecast of seasonal diffs: [0, 0] (predicting "stays at 4")
    # Actually, for white noise diffs, forecast ≈ 0 not 4
    # Let's use a concrete diff_fc:
    diff_fc3 = [4.0, 4.0]
    fc3 = _cumulate_forecasts(y3, diff_fc3, 0, 1, 4)
    # Undo seasonal: fc_y[j] = diff_fc3[j] + y[8 + j - 4] = diff_fc3[j] + y[4+j]
    # fc_y[1] = 4 + y[5] = 4 + 5 = 9
    # fc_y[2] = 4 + y[6] = 4 + 6 = 10
    @test fc3 ≈ [9.0, 10.0]
end

# ─── _cumulate_forecasts (seasonal, h > s) ───────────────────────────

@testset "_cumulate_forecasts seasonal h > s" begin
    # Test case where forecast horizon exceeds seasonal period
    # y = [10, 20, 30, 40] with s=2, D=1
    y = [10.0, 20.0, 30.0, 40.0]
    # Seasonal diff: [30-10, 40-20] = [20, 20]
    # Forecast diffs for h=3: [20, 20, 20]
    diff_fc = [20.0, 20.0, 20.0]
    fc = _cumulate_forecasts(y, diff_fc, 0, 1, 2)
    # j=1: back_idx = 4+1-2 = 3, y[3]=30, fc[1] = 20+30 = 50
    # j=2: back_idx = 4+2-2 = 4, y[4]=40, fc[2] = 20+40 = 60
    # j=3: back_idx = 4+3-2 = 5 > 4, so fc[3] = 20+fc[5-4] = 20+fc[1] = 20+50 = 70
    @test fc ≈ [50.0, 60.0, 70.0]
end

# ─── End-to-end: kolmogorov_wiener → forecast ────────────────────────

@testset "End-to-end: kolmogorov_wiener then forecast" begin
    y = air_passengers()
    r = kolmogorov_wiener(y, :hp; m=12, maxcoef=50)
    fc = forecast(r; h=12)

    @test fc.method == "Kolmogorov-Wiener"
    @test length(fc.mean) == 12
    @test fc.level == [80, 95]
    @test size(fc.upper) == (12, 2)
    @test size(fc.lower) == (12, 2)
    @test length(fc.x) == length(y)
    @test length(fc.fitted) == length(y)
    @test length(fc.residuals) == length(y)

    for j in 1:12
        for col in 1:2
            @test fc.lower[j, col] < fc.mean[j] < fc.upper[j, col]
        end
    end

    for j in 1:12
        @test fc.upper[j, 2] - fc.lower[j, 2] > fc.upper[j, 1] - fc.lower[j, 1]
    end

    @test all(isfinite, fc.mean)
    @test all(isfinite, fc.upper)
    @test all(isfinite, fc.lower)
end

@testset "Forecast with trend output" begin
    y = air_passengers()
    r = kolmogorov_wiener(y, :hp; m=12, maxcoef=50, output=:trend)
    fc = forecast(r; h=6)

    @test length(fc.mean) == 6
    @test all(isfinite, fc.mean)
end

# ─── Prediction intervals dimensions ─────────────────────────────────

@testset "Prediction interval dimensions" begin
    y = air_passengers()
    r = kolmogorov_wiener(y, :hp; m=12, maxcoef=50)

    fc1 = forecast(r; h=5, level=[90])
    @test size(fc1.upper) == (5, 1)
    @test size(fc1.lower) == (5, 1)

    fc3 = forecast(r; h=5, level=[50, 80, 95])
    @test size(fc3.upper) == (5, 3)
    @test size(fc3.lower) == (5, 3)
end

# ─── nlags parameter ─────────────────────────────────────────────────

@testset "nlags parameter" begin
    y = air_passengers()
    r = kolmogorov_wiener(y, :hp; m=12, maxcoef=50)

    fc_short = forecast(r; h=6, nlags=24)
    @test length(fc_short.mean) == 6
    @test all(isfinite, fc_short.mean)

    fc_min = forecast(r; h=3, nlags=5)
    @test length(fc_min.mean) == 3
    @test all(isfinite, fc_min.mean)
end

# ─── Ridge regularization ────────────────────────────────────────────

@testset "Ridge regularization" begin
    y = air_passengers()
    r = kolmogorov_wiener(y, :hp; m=12, maxcoef=50)

    fc_ridge = forecast(r; h=6, ridge=1e-4)
    @test all(isfinite, fc_ridge.mean)
    @test all(isfinite, fc_ridge.upper)

    fc_big_ridge = forecast(r; h=6, ridge=10.0)
    @test all(isfinite, fc_big_ridge.mean)
end

# ─── Bandpass filter forecast ─────────────────────────────────────────

@testset "Bandpass filter forecast" begin
    y = air_passengers()
    r = kolmogorov_wiener(y, :bandpass; low=6, high=32, m=12, maxcoef=50)
    fc = forecast(r; h=8)

    @test length(fc.mean) == 8
    @test all(isfinite, fc.mean)
    @test size(fc.upper) == (8, 2)
end
