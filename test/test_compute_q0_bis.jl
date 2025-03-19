@testset "compute_q0_bis translation tests" begin

    phi = Float64[]
    theta = [0.3, 0.7]
    Q0 = compute_q0_bis(phi, theta)
    @test size(Q0) == (length(theta)+1, length(theta)+1)
    @test isapprox(Q0, Q0', atol=1e-10)  # Must be symmetric

    phi = [0.5, -0.3, 0.1]
    theta = Float64[] # q = 0, so r = max(3, 0+1) = 3
    Q0 = compute_q0_bis(phi, theta)
    @test size(Q0) == (max(length(phi), length(theta)+1), max(length(phi), length(theta)+1))
    @test isapprox(Q0, Q0', atol=1e-10)

    phi = [0.5]
    theta = [0.2]
    Q0 = compute_q0_bis(phi, theta)
    @test size(Q0) == (2, 2)
    @test isapprox(Q0, Q0', atol=1e-10)
    
    phi = randn(3)                       # p = 3
    theta = randn(2)                     # q = 2, so r = max(3, 2+1) = 3
    Q0a = compute_q0_bis(phi, theta)
    Q0b = compute_q0_bis(phi, theta)
    r = max(length(phi), length(theta)+1)
    @test size(Q0a) == (r, r)
    @test isapprox(Q0a, Q0b, atol=1e-10)
    @test isapprox(Q0a, Q0a', atol=1e-10)

end
