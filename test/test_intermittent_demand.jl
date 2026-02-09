using Test
using Durbyn.IntermittentDemand: croston_classic, croston_sba, croston_sbj,
    IntermittentDemandCrostonFit, IntermittentDemandForecast
import Durbyn.Generics: forecast, fitted, residuals

@testset "Intermittent Demand Module" begin

    demand = [6, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0]

    @testset "Basic fitting and forecasting" begin
        for (name, fn) in [("classic", croston_classic), ("sba", croston_sba), ("sbj", croston_sbj)]
            fit = fn(demand)
            @test isa(fit, IntermittentDemandCrostonFit)
            @test length(fit.x) == length(demand)

            fc = forecast(fit, h=12)
            @test isa(fc, IntermittentDemandForecast)
            @test length(fc.mean) == 12
            @test all(isfinite.(fc.mean))
            @test all(fc.mean .> 0)
        end
    end

    @testset "rm_missing=true produces correct fitted/residuals lengths" begin
        demand_with_missing = [6, missing, 0, 1, 0, missing, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0]
        fit = croston_classic(demand_with_missing; rm_missing=true)

        fitted_vals = fitted(fit)
        resid_vals = residuals(fit)

        # After removing 2 missings, data has 18 elements
        @test length(fit.x) == 18
        @test length(fitted_vals) == length(fit.x)
        @test length(resid_vals) == length(fit.x)
        @test fit.na_rm == true  # reflects user's rm_missing choice
    end

    @testset "rm_missing=false with missings throws ArgumentError" begin
        demand_with_missing = [6, missing, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0]
        @test_throws ArgumentError croston_classic(demand_with_missing; rm_missing=false)
        @test_throws ArgumentError croston_sba(demand_with_missing; rm_missing=false)
        @test_throws ArgumentError croston_sbj(demand_with_missing; rm_missing=false)
    end

    @testset "number_of_params=1 returns flat weight vector" begin
        fit = croston_classic(demand; number_of_params=1)
        @test isa(fit.weights, AbstractVector)
        @test length(fit.weights) == 1
        @test eltype(fit.weights) <: Number

        # Fitted/residuals should still work
        fitted_vals = fitted(fit)
        @test length(fitted_vals) == length(demand)
    end

    @testset "optimize_init=true works correctly" begin
        fit = croston_classic(demand; optimize_init=true)
        @test isa(fit, IntermittentDemandCrostonFit)
        fc = forecast(fit, h=5)
        @test length(fc.mean) == 5
        @test all(isfinite.(fc.mean))
    end

    @testset "optimize_init=false works correctly" begin
        fit = croston_classic(demand; optimize_init=false)
        @test isa(fit, IntermittentDemandCrostonFit)
        fc = forecast(fit, h=5)
        @test length(fc.mean) == 5
        @test all(isfinite.(fc.mean))
    end

    @testset "Forecast horizon validation" begin
        fit = croston_classic(demand)

        fc1 = forecast(fit, h=1)
        @test length(fc1.mean) == 1

        fc12 = forecast(fit, h=12)
        @test length(fc12.mean) == 12

        # Flat forecast: all values should be identical
        @test all(fc12.mean .== fc12.mean[1])
    end

    @testset "Fitted/residuals consistency" begin
        fit = croston_sba(demand)
        fitted_vals = fitted(fit)
        resid_vals = residuals(fit)

        @test length(fitted_vals) == length(demand)
        @test length(resid_vals) == length(demand)
        @test resid_vals ≈ fit.x .- fitted_vals
    end

    @testset "AbstractVector input" begin
        v = view(demand, 1:length(demand))
        fit = croston_classic(v)
        @test isa(fit, IntermittentDemandCrostonFit)
        fc = forecast(fit, h=5)
        @test length(fc.mean) == 5
    end

    @testset "Too few non-zero values throws ArgumentError" begin
        bad_demand = [0, 0, 0, 1, 0, 0, 0]  # only one non-zero
        @test_throws ArgumentError croston_classic(bad_demand)
    end

    @testset "show methods" begin
        fit = croston_classic(demand)
        fc = forecast(fit, h=5)

        io = IOBuffer()
        show(io, fit)
        output = String(take!(io))
        @test occursin("IntermittentDemandCrostonFit", output)

        io = IOBuffer()
        show(io, fc)
        output = String(take!(io))
        @test occursin("IntermittentDemandForecast", output)
        @test occursin("Mean (first 5 values)", output)
    end

    @testset "forecast h=0 returns nothing mean" begin
        fit = croston_classic(demand)
        fc = forecast(fit, h=0)
        @test isnothing(fc.mean)

        # show should handle nothing mean gracefully
        io = IOBuffer()
        show(io, fc)
        output = String(take!(io))
        @test occursin("not yet forecasted", output)
    end

    @testset "na_rm=false without missings works normally" begin
        fit = croston_classic(demand; rm_missing=false)
        @test fit.na_rm == false
        @test length(fit.x) == length(demand)

        fitted_vals = fitted(fit)
        @test length(fitted_vals) == length(demand)
    end

    @testset "manually constructed model with missings errors on forecast/fitted" begin
        # Simulate a user manually constructing a model with dirty data
        bad_model = IntermittentDemandCrostonFit(
            [0.1, 0.1], [2.0, 3.0], "croston", true, [6, missing, 0, 1, 0, 0, 2]
        )
        @test_throws ArgumentError forecast(bad_model, h=5)
        @test_throws ArgumentError fitted(bad_model)
    end

    @testset "SBA applies bias correction" begin
        fit_classic = croston_classic(demand)
        fit_sba = croston_sba(demand)
        fc_classic = forecast(fit_classic, h=1)
        fc_sba = forecast(fit_sba, h=1)
        # SBA should generally produce lower forecasts than classic due to bias correction
        @test fc_sba.mean[1] <= fc_classic.mean[1]
    end

    @testset "SBJ applies bias correction" begin
        fit_classic = croston_classic(demand)
        fit_sbj = croston_sbj(demand)
        fc_classic = forecast(fit_classic, h=1)
        fc_sbj = forecast(fit_sbj, h=1)
        # SBJ should generally produce lower forecasts than classic
        @test fc_sbj.mean[1] <= fc_classic.mean[1]
    end
end
