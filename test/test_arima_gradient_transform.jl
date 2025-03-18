@testset "arima_gradient_transform correctness" begin
    x = [0.5, 0.3, 2.0, 2.1]
    arma = [1, 0, 1]
    
    m = arima_gradient_transform(x, arma)
    expected = [0.7860842079382979, 0.9148701436583195, 1.0, 1.0]
    
    @test isapprox(diag(m), expected; atol = 1e-7)
end


@testset "arima_undo_params correctness" begin
    x = [0.5, 0.3, 2.0, 2.1]
    arma = [1, 0, 1]
    
    m = arima_undo_params(x, arma)
    expected = [0.46211715726000974, 0.2913126124515909, 2.0, 2.1]
    
    @test isapprox(diag(m), expected; atol = 1e-7)
end