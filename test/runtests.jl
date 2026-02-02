using Test

@testset "Durbyn.jl" begin
    @testset "Utils Module" begin
        include("test_utils.jl")
    end

    @testset "TableOps Module" begin
        include("test_tableops.jl")
    end

    @testset "Stats Module" begin
        include("test_stats.jl")
    end

    @testset "Optimize Module" begin
        include("test_optimize.jl")
    end

    @testset "Grammar Module" begin
        include("test_grammar.jl")
    end

    @testset "ARAR Model" begin
        include("test_arar.jl")
    end

    @testset "ARARMA Model" begin
        include("test_ararma.jl")
    end

    @testset "Exponential Smoothing Models" begin
        include("test_exponential_smoothing.jl")
    end

    @testset "ARIMA Model" begin
        include("test_arima_forecast.jl")
    end

    @testset "Auto ARIMA Model" begin
        include("test_auto_arima.jl")
    end

    @testset "BATS Model" begin
        include("test_bats.jl")
    end

    @testset "TBATS Model" begin
        include("test_tbats_formula.jl")
    end

    @testset "Croston Model" begin
        include("test_croston.jl")
    end

    @testset "Theta Model" begin
        include("test_theta.jl")
    end

    @testset "Diffusion Models" begin
        include("test_diffusion.jl")
    end

    @testset "Naive Model" begin
        include("test_naive.jl")
    end

    @testset "Reference Tests" begin
        include("test_reference.jl")
    end

    @testset "Accuracy Module" begin
        include("test_accuracy.jl")
    end
end
