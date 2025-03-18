@testset "arima_inverse_transform correctness" begin
    x = [0.1, 0.01, 2.0, 2.1]
    arma = [1, 0, 1]
    out = arima_inverse_transform(x, arma)
    expected = [0.10033534773107558, 0.010000333353334763, 2.0, 2.1]

    @test isapprox(out, expected; atol = 1e-7)
end
