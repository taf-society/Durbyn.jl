using Test

# Tolerance for approximate comparison
tol = 1e-4

# Function to compare OrderedDict with tolerance for numerical values
function compare_with_tolerance(result, expected, tol)
    for key in keys(expected)
        if isnothing(expected[key])
            @test isnothing(result[key])  # Check if both are nothing
        else
            @test isapprox(result[key], expected[key], rtol=tol)  # Approximate comparison with tolerance
        end
    end
end

function compare_with_tolerance2(result, expected, tol)
    for key in keys(expected)
        if isnan(expected[key])
            @test isnan(result[key])  # Check if both are nothing
        else
            @test isapprox(result[key], expected[key], rtol=tol)  # Approximate comparison with tolerance
        end
    end
end

@testset "Test initparam Function with Tolerance" begin

    alpha = 1.0
    beta = nothing
    gamma = nothing
    phi = nothing
    trendtype = "N"
    seasontype = "N"
    damped = false
    lower = [0.0001, 0.0001, 0.0001, 0.8]
    upper = [0.9999, 0.9999, 0.0, 0.98]
    m = 1
    bounds = "both"

    @test_throws ArgumentError initparam(alpha, beta, gamma, phi, trendtype, seasontype, damped, lower, upper, m, bounds, nothing_as_nan = true)
    

    # Test case 1: trendtype = "N", seasontype = "N"
    alpha = nothing
    beta = nothing
    gamma = nothing
    phi = nothing
    trendtype = "N"
    seasontype = "N"
    damped = false
    lower = [1e-04, 1e-04, 1e-04, 8e-01]
    upper = [0.9999, 0.9999, 0.9999, 0.9800]
    m = 1
    bounds = "both"
    result = initparam(alpha, beta, gamma, phi, trendtype, seasontype, damped, lower, upper, m, bounds)
    expected = OrderedDict("alpha" => 0.20006, "beta" => nothing, "gamma" => nothing, "phi" => nothing)
    compare_with_tolerance(result, expected, tol)

    # Test case 2: trendtype = "N", seasontype = "M"
    seasontype = "M"
    m = 12
    result = initparam(alpha, beta, gamma, phi, trendtype, seasontype, damped, lower, upper, m, bounds)
    expected = OrderedDict("alpha" => 0.0167633, "beta" => nothing, "gamma" => 0.0492568, "phi" => nothing)
    compare_with_tolerance(result, expected, tol)

    # Test case 3: trendtype = "A", seasontype = "N"
    trendtype = "A"
    seasontype = "N"
    m = 1
    result = initparam(alpha, beta, gamma, phi, trendtype, seasontype, damped, lower, upper, m, bounds)
    expected = OrderedDict("alpha" => 0.20006, "beta" => 0.020096, "gamma" => nothing, "phi" => nothing)
    compare_with_tolerance(result, expected, tol)

    # Test case 3: trendtype = "A", seasontype = "N"
    trendtype = "A"
    seasontype = "N"
    m = 1
    result = initparam(alpha, beta, gamma, phi, trendtype, seasontype, damped, lower, upper, m, bounds,  nothing_as_nan = true)
    expected = OrderedDict("alpha" => 0.20006, "beta" => 0.020096, "gamma" => NaN, "phi" => NaN)
    compare_with_tolerance2(result, expected, tol)

    # Test case 4: trendtype = "A", seasontype = "A", damped = true
    seasontype = "A"
    damped = true
    m = 12
    result = initparam(alpha, beta, gamma, phi, trendtype, seasontype, damped, lower, upper, m, bounds)
    expected = OrderedDict("alpha" => 0.0167633, "beta" => 0.00176633, "gamma" => 0.0492568, "phi" => 0.9782)
    compare_with_tolerance(result, expected, tol)

    # Test case 5: trendtype = "A", seasontype = "A", damped = false
    damped = false
    result = initparam(alpha, beta, gamma, phi, trendtype, seasontype, damped, lower, upper, m, bounds)
    expected = OrderedDict("alpha" => 0.0167633, "beta" => 0.00176633, "gamma" => 0.0492568, "phi" => nothing)
    compare_with_tolerance(result, expected, tol)

    damped = false
    result = initparam(alpha, beta, gamma, phi, trendtype, seasontype, damped, lower, upper, m, bounds, nothing_as_nan = true)
    expected = OrderedDict("alpha" => 0.0167633, "beta" => 0.00176633, "gamma" => 0.0492568, "phi" => NaN)
    compare_with_tolerance2(result, expected, tol)

end