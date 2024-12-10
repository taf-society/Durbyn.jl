@testset "calculate_opt_sse function tests" begin
    
    select = [1,1,1]
    p = [0.016666667, 0.001666667, 0.049166667]
    x = air_passengers()
    lenx = 144
    alpha = nothing
    beta = nothing
    gamma = nothing
    seasonal = "additive"
    m = 12
    exponential = false
    phi = 1.0
    l_start =  [126.6667]
    b_start = [1.083333]
    s_start = [-14.6666666666667, -8.66666666666667, 5.33333333333333, 2.33333333333333, -5.66666666666667, 8.33333333333333, 
            21.3333333333333, 21.3333333333333, 9.33333333333333, -7.66666666666667, -22.6666666666667, -8.66666666666667]
    @test calculate_opt_sse(p, select, x, lenx, alpha, beta, gamma, seasonal, m, exponential, phi, l_start, b_start, s_start) == 489658.7470470203
    end