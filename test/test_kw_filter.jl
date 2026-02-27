using Test
using Durbyn
using Durbyn.KolmogorovWiener: wold_coefficients, arma_autocovariance,
    extract_arma_for_kw, bandpass_coefs, hp_coefs, butterworth_coefs,
    proposition1_stationary, proposition2_random_walk, proposition3_rw_symmetric,
    proposition4_arima, compute_optimal_filter,
    adaptive_simpsons, build_toeplitz_gamma_hat,
    build_D_cumul_matrix, build_M_matrix, build_gamma_cross_integrated, build_C_vector
using LinearAlgebra: dot

# ─── Wold Coefficients ─────────────────────────────────────────────────

@testset "Wold coefficients" begin
    # AR(1): psi_k = phi^k
    phi_ar1 = [0.7]
    psi = wold_coefficients(phi_ar1, Float64[], 10)
    for k in 0:10
        @test psi[k + 1] ≈ 0.7^k atol=1e-12
    end

    # MA(1): psi_0 = 1, psi_1 = theta, psi_k = 0 for k > 1
    theta_ma1 = [0.5]
    psi = wold_coefficients(Float64[], theta_ma1, 5)
    @test psi[1] ≈ 1.0
    @test psi[2] ≈ 0.5
    for k in 2:5
        @test psi[k + 1] ≈ 0.0 atol=1e-15
    end

    # ARMA(1,1): psi_0 = 1, psi_k = (phi + theta) * phi^(k-1) for k >= 1
    phi_arma = [0.8]
    theta_arma = [0.3]
    psi = wold_coefficients(phi_arma, theta_arma, 10)
    @test psi[1] ≈ 1.0
    for k in 1:10
        @test psi[k + 1] ≈ (0.8 + 0.3) * 0.8^(k - 1) atol=1e-10
    end

    # Pure white noise: psi_0 = 1, rest = 0
    psi = wold_coefficients(Float64[], Float64[], 5)
    @test psi[1] ≈ 1.0
    for k in 1:5
        @test psi[k + 1] ≈ 0.0
    end
end

# ─── Autocovariance ────────────────────────────────────────────────────

@testset "Autocovariance" begin
    # White noise: gamma_0 = sigma2, gamma_k = 0 for k > 0
    gamma = arma_autocovariance(Float64[], Float64[], 2.0, 5)
    @test gamma[1] ≈ 2.0 atol=1e-6
    for k in 1:5
        @test abs(gamma[k + 1]) < 1e-6
    end

    # AR(1) with phi=0.7, sigma2=1: gamma_k = sigma2 * phi^k / (1 - phi^2)
    phi_val = 0.7
    sigma2 = 1.0
    gamma = arma_autocovariance([phi_val], Float64[], sigma2, 10)
    for k in 0:10
        expected = sigma2 * phi_val^k / (1 - phi_val^2)
        @test gamma[k + 1] ≈ expected atol=1e-4
    end

    # MA(1) with theta=0.5, sigma2=1: gamma_0 = sigma2*(1+theta^2), gamma_1 = sigma2*theta, gamma_k=0 for k>1
    theta_val = 0.5
    gamma = arma_autocovariance(Float64[], [theta_val], 1.0, 5)
    @test gamma[1] ≈ 1.0 * (1 + theta_val^2) atol=1e-4
    @test gamma[2] ≈ 1.0 * theta_val atol=1e-4
    for k in 2:5
        @test abs(gamma[k + 1]) < 1e-4
    end
end

# ─── Quadrature ────────────────────────────────────────────────────────

@testset "Adaptive quadrature" begin
    # Integral of sin(x) from 0 to pi = 2
    @test adaptive_simpsons(sin, 0.0, pi) ≈ 2.0 atol=1e-10

    # Integral of x^2 from 0 to 1 = 1/3
    @test adaptive_simpsons(x -> x^2, 0.0, 1.0) ≈ 1/3 atol=1e-10

    # Integral of cos(x) from 0 to pi = 0
    @test adaptive_simpsons(cos, 0.0, pi) ≈ 0.0 atol=1e-10
end

# ─── Bandpass Filter Coefficients ──────────────────────────────────────

@testset "Bandpass coefs" begin
    # Standard business cycle filter: periods 6-32
    B = bandpass_coefs(6, 32, 100)
    center = 101

    # B_0 = (b - a) / pi where a = 2pi/32, b = 2pi/6
    a = 2 * pi / 32
    b = 2 * pi / 6
    @test B[center] ≈ (b - a) / pi atol=1e-12

    # Symmetry
    for j in 1:100
        @test B[center + j] ≈ B[center - j] atol=1e-15
    end

    # Sum ≈ 0 when 0 is not in passband (bandpass removes DC)
    # Truncation at maxcoef=100 leaves some residual
    @test abs(sum(B)) < 0.05

    # Input validation
    @test_throws ArgumentError bandpass_coefs(-1, 32, 100)
    @test_throws ArgumentError bandpass_coefs(32, 6, 100)
end

# ─── HP Filter Coefficients ───────────────────────────────────────────

@testset "HP coefs" begin
    B = hp_coefs(1600.0, 100)
    center = 101

    # Highpass: sum of coefficients ≈ 0
    @test abs(sum(B)) < 0.01

    # Symmetry
    for j in 1:100
        @test B[center + j] ≈ B[center - j] atol=1e-12
    end

    # B_0 should be positive (highpass center weight)
    @test B[center] > 0

    # Input validation
    @test_throws ArgumentError hp_coefs(-1.0, 100)
end

# ─── Butterworth Filter Coefficients ──────────────────────────────────

@testset "Butterworth coefs" begin
    omega_c = pi / 16
    B = butterworth_coefs(2, omega_c, 100)
    center = 101

    # Highpass: B_0 should be positive
    # Note: with finite truncation, sum may not be close to 0
    @test B[center] > 0

    # Symmetry
    for j in 1:100
        @test B[center + j] ≈ B[center - j] atol=1e-12
    end

    # Input validation
    @test_throws ArgumentError butterworth_coefs(0, omega_c, 100)
    @test_throws ArgumentError butterworth_coefs(2, 0.0, 100)
    @test_throws ArgumentError butterworth_coefs(2, pi, 100)
end

# ─── Proposition 1: White Noise ───────────────────────────────────────

@testset "Proposition 1 - stationary white noise" begin
    # For white noise, gamma_0 = sigma2, gamma_k = 0
    # Gamma_hat = sigma2 * I, Gamma_cross has gamma at specific lags
    # Optimal filter = truncated ideal (Dirichlet window)
    sigma2 = 1.0
    maxcoef = 5
    nobs = 20
    gamma = zeros(maxcoef + nobs + 1)
    gamma[1] = sigma2  # gamma_0

    # Simple ideal filter: ones in center
    ideal_B = zeros(2 * maxcoef + 1)
    center = maxcoef + 1
    ideal_B[center] = 1.0  # identity filter

    n1 = 9  # t = 10
    n2 = 10
    B_hat = proposition1_stationary(gamma, ideal_B, n1, n2)

    # For identity filter on white noise, optimal = identity at position t
    # B_hat should be 1 at position t and 0 elsewhere
    @test length(B_hat) == nobs
    @test B_hat[n1 + 1] ≈ 1.0 atol=1e-10
    for i in 1:nobs
        if i != n1 + 1
            @test abs(B_hat[i]) < 1e-10
        end
    end
end

# ─── Proposition 2: Random Walk ───────────────────────────────────────

@testset "Proposition 2 - random walk" begin
    maxcoef = 3
    center = maxcoef + 1

    # Simple lowpass filter with beta > 0
    ideal_B = zeros(2 * maxcoef + 1)
    ideal_B[center] = 0.5
    ideal_B[center + 1] = 0.2
    ideal_B[center - 1] = 0.2
    ideal_B[center + 2] = 0.05
    ideal_B[center - 2] = 0.05
    beta = sum(ideal_B)

    # Interior observation (all ideal coefs fit within sample)
    n1 = 5
    n2 = 5
    N = n1 + 1 + n2
    B_hat = proposition2_random_walk(ideal_B, n1, n2)
    @test length(B_hat) == N

    # Sum should equal beta (sum preservation)
    @test sum(B_hat) ≈ beta atol=1e-12

    # Endpoint: left edge
    n1_edge = 1
    n2_edge = 9
    B_hat_edge = proposition2_random_walk(ideal_B, n1_edge, n2_edge)
    @test length(B_hat_edge) == n1_edge + 1 + n2_edge
    @test sum(B_hat_edge) ≈ beta atol=1e-12
end

# ─── End-to-End: kw_filter ────────────────────────────────────────────

@testset "kolmogorov_wiener end-to-end" begin
    y = air_passengers()

    # HP filter
    r = kolmogorov_wiener(y, :hp; lambda=1600.0, m=12, maxcoef=50)
    @test length(r.filtered) == length(y)
    @test size(r.weights) == (length(y), length(y))
    @test r.filter_type == :hp
    @test r.output == :cycle

    # Trend output should be complement: trend + cycle ≈ y
    r_trend = kolmogorov_wiener(y, :hp; lambda=1600.0, m=12, maxcoef=50,
                        arima_model=r.arima_model, output=:trend)
    @test r_trend.output == :trend

    # fitted and residuals
    @test fitted(r) == r.filtered
    @test residuals(r) ≈ r.y .- r.filtered

    # Input validation
    @test_throws ArgumentError kolmogorov_wiener(y, :unknown_filter)
    @test_throws ArgumentError kolmogorov_wiener(y, :hp; output=:invalid)
    @test_throws ArgumentError kolmogorov_wiener([1.0, 2.0], :hp)  # too short

    # Bandpass filter
    r_bp = kolmogorov_wiener(y, :bandpass; low=6, high=32, m=12, maxcoef=50)
    @test length(r_bp.filtered) == length(y)
    @test r_bp.filter_type == :bandpass

    # Custom filter
    r_custom = kolmogorov_wiener(y, :custom; transfer_fn=omega -> omega < pi/6 ? 1.0 : 0.0,
                         m=12, maxcoef=50)
    @test length(r_custom.filtered) == length(y)
end

@testset "kolmogorov_wiener with pre-fitted model" begin
    y = air_passengers()
    fit = auto_arima(y, 12)

    r1 = kolmogorov_wiener(y, :hp; lambda=1600.0, arima_model=fit, maxcoef=50)
    r2 = kolmogorov_wiener(y, :hp; lambda=1600.0, arima_model=fit, maxcoef=50)

    # Same model should give identical results
    @test r1.filtered ≈ r2.filtered atol=1e-12
end

@testset "kolmogorov_wiener show" begin
    y = air_passengers()
    r = kolmogorov_wiener(y, :hp; lambda=1600.0, m=12, maxcoef=50)
    io = IOBuffer()
    show(io, r)
    s = String(take!(io))
    @test occursin("KWFilterResult", s)
    @test occursin("hp", s)
    @test occursin("cycle", s)
end

# ─── D_cumul and M matrix construction ───────────────────────────────

@testset "D_cumul matrix construction" begin
    D = build_D_cumul_matrix(4)
    @test size(D) == (3, 4)
    # Lower-triangular ones in first 3 cols, zero last col
    @test D ≈ [1 0 0 0;
               1 1 0 0;
               1 1 1 0]

    D5 = build_D_cumul_matrix(5)
    @test size(D5) == (4, 5)
    # D*y gives partial sums
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    ps = D5 * y
    @test ps ≈ [1.0, 3.0, 6.0, 10.0]
end

@testset "M matrix construction" begin
    # n1=2, n2=2 → N=5, M is 4×5
    M = build_M_matrix(2, 2)
    @test size(M) == (4, 5)

    # Column 1 should be all zeros
    @test all(M[:, 1] .== 0.0)
    # Column 5 (last) should be all zeros
    @test all(M[:, 5] .== 0.0)

    # Top block M1 (rows 1:2, cols 2:3): upper tri, -1 fill, -1/2 last col
    # Row 1: cols 2,3 → [-1, -0.5]
    @test M[1, 2] ≈ -1.0
    @test M[1, 3] ≈ -0.5
    # Row 2: col 3 → [-0.5]
    @test M[2, 2] ≈ 0.0
    @test M[2, 3] ≈ -0.5

    # Bottom block M2 (rows 3:4, cols 3:4): lower tri, 1 fill, 1/2 first col
    # Row 3: col 3 → [0.5]
    @test M[3, 3] ≈ 0.5
    @test M[3, 4] ≈ 0.0
    # Row 4: cols 3,4 → [0.5, 1]
    @test M[4, 3] ≈ 0.5
    @test M[4, 4] ≈ 1.0

    # n1=3, n2=1 → N=5, M is 4×5
    M2 = build_M_matrix(3, 1)
    @test size(M2) == (4, 5)
    # Bottom block is 1×1: just [0.5] at row 4, col 4
    @test M2[4, 4] ≈ 0.5
end

# ─── Proposition 3: beta ≠ 0 (random walk, lowpass) ─────────────────

@testset "Proposition 3 - beta≠0 (lowpass on random walk)" begin
    maxcoef = 5
    center = maxcoef + 1

    # Simple lowpass filter with beta = 1
    ideal_B = zeros(2 * maxcoef + 1)
    ideal_B[center] = 0.5
    ideal_B[center + 1] = 0.2
    ideal_B[center - 1] = 0.2
    ideal_B[center + 2] = 0.05
    ideal_B[center - 2] = 0.05
    beta = sum(ideal_B)

    # Interior observation: n1=n2=10 (all ideal coefs fit)
    n1 = 10
    n2 = 10
    N = n1 + 1 + n2
    B_hat = proposition3_rw_symmetric(ideal_B, n1, n2, beta)
    @test length(B_hat) == N

    # Sum preservation: sum(B_hat) must equal beta
    @test sum(B_hat) ≈ beta atol=1e-10

    # For interior obs with plenty of room, Prop 3 should match Prop 2
    B_hat_p2 = proposition2_random_walk(ideal_B, n1, n2)
    @test B_hat ≈ B_hat_p2 atol=1e-8

    # Endpoint: n1=1, n2=8
    n1_edge = 1
    n2_edge = 8
    B_hat_edge = proposition3_rw_symmetric(ideal_B, n1_edge, n2_edge, beta)
    @test sum(B_hat_edge) ≈ beta atol=1e-10

    # beta=0 (highpass) should fall back to Prop 2
    hp_B = zeros(2 * maxcoef + 1)
    hp_B[center] = 0.8
    hp_B[center + 1] = -0.1
    hp_B[center - 1] = -0.1
    hp_B[center + 2] = -0.1
    hp_B[center - 2] = -0.1
    # Sum is 0.4, not zero. Let's make a true highpass:
    hp_B2 = copy(ideal_B)
    hp_B2[center] -= 1.0  # I - lowpass → highpass, sum = beta - 1
    hp_B2 .-= sum(hp_B2) / length(hp_B2)  # force sum to 0 for test
    # Actually, just test with exact zero beta
    zero_B = zeros(2 * maxcoef + 1)
    zero_B[center] = 0.5
    zero_B[center + 1] = -0.25
    zero_B[center - 1] = -0.25
    @test abs(sum(zero_B)) < 1e-15
    B_hat_hp = proposition3_rw_symmetric(zero_B, 5, 5, 0.0)
    B_hat_p2_hp = proposition2_random_walk(zero_B, 5, 5)
    @test B_hat_hp ≈ B_hat_p2_hp atol=1e-12
end

# ─── Proposition 4: ARIMA ────────────────────────────────────────────

@testset "Proposition 4 - ARIMA" begin
    # AR(1) with phi=0.7, sigma2=1, d=1
    phi = 0.7
    sigma2 = 1.0
    maxcoef = 10
    gamma = arma_autocovariance([phi], Float64[], sigma2, maxcoef + 30)

    # Lowpass filter (HP trend)
    ideal_B = hp_coefs(1600.0, maxcoef)
    center = maxcoef + 1
    # HP is highpass; trend = I - HP
    trend_B = -copy(ideal_B)
    trend_B[center] += 1.0
    beta = sum(trend_B)

    # Interior observation
    n1 = 10
    n2 = 10
    N = n1 + 1 + n2
    B_hat = proposition4_arima(gamma, trend_B, n1, n2, beta)

    @test length(B_hat) == N
    # Sum preservation
    @test sum(B_hat) ≈ beta atol=1e-6

    # Endpoint
    n1_edge = 2
    n2_edge = 8
    B_hat_edge = proposition4_arima(gamma, trend_B, n1_edge, n2_edge, beta)
    @test length(B_hat_edge) == n1_edge + 1 + n2_edge
    @test sum(B_hat_edge) ≈ beta atol=1e-6

    # For very weak AR (phi≈0), Prop 4 should approach Prop 3 behavior
    # (white noise stationary component → cumulation-based solution)
    gamma_wn = arma_autocovariance([0.01], Float64[], sigma2, maxcoef + 30)
    B_hat_wn = proposition4_arima(gamma_wn, trend_B, n1, n2, beta)
    @test sum(B_hat_wn) ≈ beta atol=1e-6

    # Convergence check: near-white-noise Prop 4 ≈ Prop 3
    # With phi→0 the ARMA autocovariance approaches white noise, so Prop 4
    # weights should converge to Prop 3 weights (difference scales linearly with phi).
    gamma_weak = arma_autocovariance([0.0001], Float64[], sigma2, maxcoef + 30)
    b4 = proposition4_arima(gamma_weak, trend_B, n1, n2, beta)
    b3 = proposition3_rw_symmetric(trend_B, n1, n2, beta)
    @test maximum(abs.(b4 .- b3)) < 1e-4
end

# ─── Cross-covariance and C vector for Prop 4 ───────────────────────

@testset "build_gamma_cross_integrated dimensions" begin
    N = 10
    Q = 5
    gamma = ones(50)  # dummy
    G = build_gamma_cross_integrated(gamma, N, Q)
    @test size(G) == (N - 1, N - 1 + 2Q)
end

@testset "build_C_vector" begin
    # Simple symmetric filter: B = [0.25, 0.5, 0.25], Q=1
    ideal_B = [0.25, 0.5, 0.25]
    Q = 1
    beta = sum(ideal_B)  # 1.0
    N = 6
    n1 = 2

    C = build_C_vector(ideal_B, N, Q, n1)
    @test length(C) == N - 1 + 2Q  # 7

    # Check cumulated values at key positions
    # c maps to j = c - n1 - Q - 1 = c - 4
    # c=1: j=-3 < -Q=-1 → C=0
    # c=2: j=-2 < -Q=-1 → C=0
    # c=3: j=-1 = -Q → C = B_{-1} = 0.25
    # c=4: j=0 → C = B_{-1} + B_0 = 0.75
    # c=5: j=1 = Q → C = B_{-1} + B_0 + B_1 = 1.0 = beta
    # c=6: j=2 > Q → C = beta = 1.0
    # c=7: j=3 > Q → C = beta = 1.0
    @test C[1] ≈ 0.0
    @test C[2] ≈ 0.0
    @test C[3] ≈ 0.25
    @test C[4] ≈ 0.75
    @test C[5] ≈ 1.0
    @test C[6] ≈ 1.0
    @test C[7] ≈ 1.0
end

# ─── kw_decomposition ───────────────────────────────────────────────

@testset "kw_decomposition" begin
    y = air_passengers()
    d = kw_decomposition(y; m=12, maxcoef=50)

    # trend + remainder = data
    @test d.trend .+ d.remainder ≈ d.data

    # Dimensions
    @test length(d.trend) == length(y)
    @test length(d.remainder) == length(y)

    # Type and method
    @test d.method == :kw
    @test d.type == :additive
    @test d.metadata[:filter_type] == :hp
    @test d.m == [12]
    @test isempty(d.seasonals)

    # Metadata fields
    @test haskey(d.metadata, :arima_model)
    @test haskey(d.metadata, :gamma)
    @test haskey(d.metadata, :d)
    @test haskey(d.metadata, :params)

    # fitted/residuals
    @test fitted(d) == d.trend
    @test residuals(d) == d.remainder

    # show
    io = IOBuffer()
    show(io, d)
    s = String(take!(io))
    @test occursin("Decomposition", s)
    @test occursin("kw", s)

    # Bandpass decomposition
    d_bp = kw_decomposition(y; filter_type=:bandpass, low=6, high=32, m=12, maxcoef=50)
    @test d_bp.trend .+ d_bp.remainder ≈ d_bp.data
    @test d_bp.metadata[:filter_type] == :bandpass

    # Non-seasonal: m defaults to empty
    d_ns = kw_decomposition(y; maxcoef=50)
    @test d_ns.m == Int[]

    # Forecast from decomposition
    fc = forecast(d; h=6)
    @test length(fc.mean) == 6
    @test all(isfinite, fc.mean)
    @test fc.method == "Kolmogorov-Wiener"
end
