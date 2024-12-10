using Test

@testset "Decomposition Tests" begin

    # Test 1: Additive decomposition
    ap = air_passengers()
    result_add = decompose(x = ap, m = 12, type = "additive")
    
    # Check if result has correct components
    @test fieldnames(typeof(result_add)) == (:x, :seasonal, :trend, :random, :figure, :type, :m)
    
    # Check if trend, seasonal, and remainder are of correct length
    @test length(result_add.trend) == length(ap)
    @test length(result_add.seasonal) == length(ap)
    @test length(result_add.random) == length(ap)
    
    # Test 2: Multiplicative decomposition
    result_mult = decompose(x = ap, m = 12, type = "multiplicative")
    
    # Check if result has correct components
    @test fieldnames(typeof(result_mult)) == (:x, :seasonal, :trend, :random, :figure, :type, :m)
    
    # Check if trend, seasonal, and remainder are of correct length
    @test length(result_mult.trend) == length(ap)
    @test length(result_mult.seasonal) == length(ap)
    @test length(result_mult.random) == length(ap)
end
