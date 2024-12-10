@testset "fourier function tests" begin
    # Test 1: Fourier terms for in-sample time series (no forecast horizon)
    y = rand(100)  # Example time series with 100 data points
    m = 12         # Seasonality (e.g., annual seasonality for monthly data)
    K = 6          # Number of Fourier terms

    # Generate Fourier terms for the original time series
    result = fourier(y, m = m, K = K)
    
    # Check the size of the resulting matrix (should be 100 rows and 12 columns)
    @test size(result) == (100, 2 * K)

    # Test 2: Fourier terms for a forecast horizon (out-of-sample)
    h = 12  # Forecast horizon (next 12 periods)
    resulth = fourier(y, m=m, K=K, h=h)
    
    # Check the size of the resulting matrix for forecast horizon
    @test size(resulth) == (h, 2 * K)
    
    # Test 3: Ensure that the values for Fourier terms are periodic (e.g., sin and cos values)
    t = 1:100
    @test all(abs.(sin.(2 * π * 1 * t / m) - result[:, 1]) .< 1e-6)  # Sine for first harmonic
    @test all(abs.(cos.(2 * π * 1 * t / m) - result[:, 2]) .< 1e-6)  # Cosine for first harmonic
    
    # Test 4: Check for smaller time series input (length 10)
    y_small = rand(10)
    result_small = fourier(y_small, m=m, K=K)
    
    # Ensure that the result matrix has 10 rows and 12 columns
    @test size(result_small) == (10, 2 * K)
    
    # Test 5: Check for forecast terms for small input series
    h_small = 5  # Short forecast horizon
    resulth_small = fourier(y_small, m=m, K=K, h=h_small)
    
    # Ensure that the forecast matrix has 5 rows and 12 columns
    @test size(resulth_small) == (h_small, 2 * K)

    # Test 6: Check if the output contains NaN or non-numeric values
    @test all(isfinite.(result))
    @test all(isfinite.(resulth))
end