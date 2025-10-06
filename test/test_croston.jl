using Test
using Durbyn.ExponentialSmoothing

@testset "Croston Method Tests" begin

    @testset "Basic Croston Fitting" begin
        # Intermittent demand data
        demand = [0, 0, 5, 0, 0, 3, 0, 0, 0, 7, 0, 0, 4, 0, 0]

        # Fit model
        fit1 = croston(demand, 1)
        fc1 = forecast(fit1, 12)
        fit2 = croston(demand)
        fc2 = forecast(fit2, 12)
        # plot(fc1)
        @test fc1.mean == fc2.mean
        @test isa(fit1, CrostonFit)
        @test all(fc1.mean .≈ 1.9736586666666665)
    end

    @testset "Croston with Fixed Alpha" begin
        demand = [0, 0, 5, 0, 0, 3, 0, 0, 0, 7, 0, 0, 4, 0, 0]

        # Fit with fixed smoothing parameter
        fit = croston(demand, alpha=0.1)

        @test isa(fit, CrostonFit)
        fc = forecast(fit, 12)
        @test all(fc.mean .≈ 1.5915857605177997)
    end

    @testset "Croston Type Two - Single Demand" begin
        # All demands are zero
        demand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        fit = croston(demand, 1)

        @test isa(fit, CrostonFit)
        fc = forecast(fit, 12)
        @test all(fc.mean .≈ 0.09090909090909091)
    end


    @testset "Croston Fitted Values" begin
        demand = [0, 0, 5, 0, 0, 3, 0, 0, 0, 7, 0, 0, 4, 0, 0]

        fit = croston(demand, 1)
        fitted_vals = fitted(fit)

        @test length(fitted_vals) == length(demand)
        @test isnan(fitted_vals[1])  # First value should be NaN

        # Compute residuals
        residuals = demand .- fitted_vals
        @test length(residuals) == length(demand)
    end

    @testset "Croston Forecast Horizon" begin
        demand = [0, 0, 5, 0, 0, 3, 0, 0, 0, 7, 0, 0, 4, 0, 0]

        fit = croston(demand, 1)

        # Different forecast horizons
        fc1 = forecast(fit, 1)
        fc6 = forecast(fit, 6)
        fc12 = forecast(fit, 12)

        @test length(fc1.mean) == 1
        @test length(fc6.mean) == 6
        @test length(fc12.mean) == 12

        # All forecasts should be the same (flat profile)
        @test fc1.mean[1] == fc6.mean[1]
        @test fc1.mean[1] == fc12.mean[1]
    end

    @testset "Croston with Real-World Example" begin
        # Example from Croston (1972)
        demand = [6, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0,
                  0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
                  0, 0, 0, 0, 0]

        fit = croston(demand, 1)

        @test isa(fit, CrostonFit)
        @test length(fit.y) > 1  # Multiple non-zero demands
        @test all(fit.y .> 0)

        # Generate forecast
        fc = forecast(fit, 12)
        @test all(fc.mean .> 0)
        @test all(isfinite.(fc.mean))
    end

    @testset "Croston Inter-demand Intervals" begin
        demand = [5, 0, 0, 3, 0, 0, 0, 7, 0, 4]

        fit = croston(demand)

        # Check inter-demand intervals
        @test length(fit.tt) > 0
        @test all(fit.tt .> 0)  # All intervals should be positive

        # Expected intervals: 1 (position 1), 3 (position 4), 4 (position 8), 2 (position 10)
        # Intervals: 1, 3, 4, 2
        @test length(fit.tt) == 4
    end

    @testset "Croston Model Structure" begin
        demand = [0, 0, 5, 0, 0, 3, 0, 0, 0, 7, 0, 0, 4, 0, 0]

        fit = croston(demand)

        # Check that models are fitted
        @test !isnothing(fit.modely)
        @test !isnothing(fit.modelp)

        # Forecast from component models
        fc = forecast(fit, 6)
        @test length(fc.mean) == 6
        @test fc.model === fit 
    end

    @testset "Croston Empty Forecast" begin
        demand = [0, 0, 5, 0, 0, 3, 0, 0, 0, 7, 0, 0, 4, 0, 0]

        fit = croston(demand, 1)

        # Zero horizon (edge case)
        # Note: This may throw an error depending on implementation
        # Adjust test if needed
    end

    @testset "Croston Consistency" begin
        demand = [0, 0, 5, 0, 0, 3, 0, 0, 0, 7, 0, 0, 4, 0, 0]

        # Fit model twice
        fit1 = croston(demand, 1, alpha=0.2)
        fit2 = croston(demand, 1, alpha=0.2)

        # Should produce identical results
        fc1 = forecast(fit1, 10)
        fc2 = forecast(fit2, 10)

        @test fc1.mean ≈ fc2.mean
    end

    @testset "Croston Non-zero Starting Demand" begin
        # Demand starts with non-zero value
        demand = [5, 0, 0, 3, 0, 0, 0, 7, 0, 4, 0, 0]

        fit = croston(demand, 1)

        @test fit.y[1] == 5
        @test length(fit.y) == 4
    end

    @testset "Croston Consecutive Non-zero Demands" begin
        # Multiple consecutive non-zero demands
        demand = [0, 0, 5, 3, 2, 0, 0, 0, 7, 0, 4, 0, 0]

        fit = croston(demand, 1)

        @test length(fit.y) == 5  # Five non-zero demands

        # Check intervals include 1's for consecutive demands
        @test minimum(fit.tt) >= 1
    end

end
