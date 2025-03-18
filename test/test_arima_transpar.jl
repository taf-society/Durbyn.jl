@testset "arima_transpar tests" begin
    
    @testset "User example (trans = true)" begin
        par = [1.26377432, 0.82436223, -0.51341576]
        arma = [2, 1, 0, 0, 12, 1, 1]
        expected_phi   = [0.2748562, 0.6774372]
        expected_theta = [-0.5134158]
        
        phi, theta = arima_transpar(par, arma, true)
        
        @test length(phi)   == length(expected_phi)
        @test length(theta) == length(expected_theta)
        
        @test all(isapprox.(phi,   expected_phi;   rtol=1e-7, atol=1e-7))
        @test all(isapprox.(theta, expected_theta; rtol=1e-7, atol=1e-7))
    end

    @testset "User example (trans = false)" begin
        par = [1.26377432, 0.82436223, -0.51341576]
        arma = [2, 1, 0, 0, 12, 1, 1]
        expected_phi=[1.26377432, 0.82436223]
        expected_theta=[-0.51341576]

        phi, theta = arima_transpar(par, arma, false)
        
        @test all(isapprox.(phi,   expected_phi;   rtol=1e-7, atol=1e-7))
        @test all(isapprox.(theta, expected_theta; rtol=1e-7, atol=1e-7))
    end

    @testset "AR(2) no transform" begin
        arma = [2, 0, 0, 0, 0]
        par = [0.5, -0.2] 
        expected_phi = [0.5, -0.2]
        expected_theta = Float64[]
        phi, theta = arima_transpar(par, arma, false)
        @test all(isapprox.(phi,   expected_phi;   rtol=1e-7, atol=1e-7))
        @test all(isapprox.(theta, expected_theta; rtol=1e-7, atol=1e-7))
    end

  
    @testset "MA(1) with transform" begin
        
        arma = [0, 1, 0, 0, 0]
        par = [1.0] 
        expected_phi = Float64[]
        expected_theta = [1.0]
        phi, theta = arima_transpar(par, arma, true)

        @test length(phi) == 0 
        @test length(theta) == 1
        @test all(isapprox.(phi,   expected_phi;   rtol=1e-7, atol=1e-7))
        @test all(isapprox.(theta, expected_theta; rtol=1e-7, atol=1e-7))
    end


    @testset "Seasonal AR(1) with s=12" begin
        arma = [1, 0, 1, 0, 12]
        par = [0.4, 0.2]
        expected_phi = [0.37994896,0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.19737532, -0.07499255]
        expected_theta = Float64[]
        phi, theta = arima_transpar(par, arma, true)

        @test length(phi) == 13
        @test length(theta) == 0
        @test all(isapprox.(phi,   expected_phi;   rtol=1e-2, atol=1e-2)) #TODO check
        @test all(isapprox.(theta, expected_theta; rtol=1e-7, atol=1e-7))
    end
end
