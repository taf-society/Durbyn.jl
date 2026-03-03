using Durbyn.Arima: _solve_stationary_covariance
using LinearAlgebra

@testset "AR(1): P[1,1] = 1/(1 - φ²)" begin
    for φ in [0.5, -0.3, 0.9, 0.999]
        P = _solve_stationary_covariance([φ], Float64[])
        @test size(P) == (1, 1)
        @test P[1, 1] ≈ 1.0 / (1.0 - φ^2)
    end
end

@testset "White noise: P = [1]" begin
    P = _solve_stationary_covariance(Float64[], Float64[])
    @test P ≈ fill(1.0, 1, 1)
end

@testset "MA(1): Lyapunov identity P = T P T' + Q" begin
    theta = [0.6]
    P = _solve_stationary_covariance(Float64[], theta)
    r = 2
    T = zeros(r, r); T[2, 1] = 1.0
    R_vec = [1.0, theta[1]]
    Q = R_vec * R_vec'
    @test P ≈ T * P * T' + Q atol = 1e-12
end

@testset "ARMA(1,1): Lyapunov identity" begin
    phi, theta = [0.7], [0.4]
    P = _solve_stationary_covariance(phi, theta)
    r = 2
    T = zeros(r, r); T[1, 1] = phi[1]; T[2, 1] = 1.0
    R_vec = [1.0, theta[1]]
    Q = R_vec * R_vec'
    @test P ≈ T * P * T' + Q atol = 1e-12
end

@testset "AR(2): Lyapunov identity" begin
    phi = [0.5, -0.3]
    P = _solve_stationary_covariance(phi, Float64[])
    r = 2
    T = zeros(r, r); T[1, 1] = phi[1]; T[1, 2] = phi[2]; T[2, 1] = 1.0
    R_vec = [1.0, 0.0]
    Q = R_vec * R_vec'
    @test P ≈ T * P * T' + Q atol = 1e-12
end

@testset "ARMA(2,1): Lyapunov identity and symmetry" begin
    phi, theta = [0.3, 0.2], [0.5]
    P = _solve_stationary_covariance(phi, theta)
    r = 2
    T = zeros(r, r); T[1, 1] = phi[1]; T[1, 2] = phi[2]; T[2, 1] = 1.0
    R_vec = [1.0, theta[1]]
    Q = R_vec * R_vec'
    @test P ≈ T * P * T' + Q atol = 1e-12
    @test issymmetric(P)
end

@testset "ARMA(2,3): r = 4, positive semi-definite" begin
    phi, theta = [0.4, -0.2], [0.3, -0.1, 0.2]
    P = _solve_stationary_covariance(phi, theta)
    @test size(P) == (4, 4)
    @test issymmetric(P)
    eigenvalues = eigvals(P)
    @test all(λ -> λ ≥ -1e-12, eigenvalues)
end

@testset "Large state dimension r = 14 (typical SARIMA)" begin
    phi = zeros(14); phi[1] = 0.3; phi[12] = 0.2; phi[13] = -0.06
    theta = zeros(13); theta[1] = -0.4; theta[12] = -0.5; theta[13] = 0.2
    P = _solve_stationary_covariance(phi, theta)
    @test size(P) == (14, 14)
    @test issymmetric(P)
    eigenvalues = eigvals(P)
    @test all(λ -> λ ≥ -1e-10, eigenvalues)
end

@testset "Near-unit-root AR(1): φ = 0.999" begin
    P = _solve_stationary_covariance([0.999], Float64[])
    @test P[1, 1] ≈ 1.0 / (1.0 - 0.999^2) rtol = 1e-8
end
