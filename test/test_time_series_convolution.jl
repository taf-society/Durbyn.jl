@testset "time_series_convolution correctness" begin
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = time_series_convolution(x, x)
    expected = [1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 264, 296, 315, 320, 310, 284, 241, 180, 100]
    @test isapprox(result, expected; atol = 1e-7)
end