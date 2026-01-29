using Test
using Durbyn

@testset "Diffusion Models" begin

    @testset "Type Definitions" begin
        @test Bass isa DiffusionModelType
        @test Gompertz isa DiffusionModelType
        @test GSGompertz isa DiffusionModelType
        @test Weibull isa DiffusionModelType
    end

    @testset "Bass Curve Generation" begin
        # Test basic curve generation
        curve = Durbyn.Diffusion.bass_curve(10, 1000.0, 0.03, 0.38)

        @test length(curve.cumulative) == 10
        @test length(curve.adoption) == 10
        @test length(curve.innovators) == 10
        @test length(curve.imitators) == 10

        # Cumulative should be monotonically increasing
        for i in 2:10
            @test curve.cumulative[i] >= curve.cumulative[i-1]
        end

        # Cumulative should approach but not exceed market potential
        @test curve.cumulative[end] < 1000.0
        @test curve.cumulative[end] > 0

        # Adoption should be non-negative
        @test all(curve.adoption .>= 0)

        # Innovators + imitators should approximately equal adoption
        for i in 1:10
            @test curve.innovators[i] + curve.imitators[i] ≈ curve.adoption[i] atol=1e-10
        end
    end

    @testset "Gompertz Curve Generation" begin
        curve = Durbyn.Diffusion.gompertz_curve(10, 1000.0, 5.0, 0.5)

        @test length(curve.cumulative) == 10
        @test length(curve.adoption) == 10

        # Cumulative should be monotonically increasing
        for i in 2:10
            @test curve.cumulative[i] >= curve.cumulative[i-1]
        end

        # Should approach market potential
        @test curve.cumulative[end] < 1000.0
        @test curve.cumulative[end] > 0
    end

    @testset "GSGompertz Curve Generation" begin
        curve = Durbyn.Diffusion.gsgompertz_curve(10, 1000.0, 0.1, 0.5, 1.0)

        @test length(curve.cumulative) == 10
        @test length(curve.adoption) == 10

        # Cumulative should be monotonically increasing
        for i in 2:10
            @test curve.cumulative[i] >= curve.cumulative[i-1]
        end
    end

    @testset "Weibull Curve Generation" begin
        curve = Durbyn.Diffusion.weibull_curve(10, 1000.0, 5.0, 2.0)

        @test length(curve.cumulative) == 10
        @test length(curve.adoption) == 10

        # Cumulative should be monotonically increasing
        for i in 2:10
            @test curve.cumulative[i] >= curve.cumulative[i-1]
        end
    end

    @testset "Bass Initialization" begin
        # Create synthetic Bass-like data
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        init = Durbyn.Diffusion.bass_init(y)

        @test haskey(pairs(init) |> collect .|> first |> Set, :m)
        @test haskey(pairs(init) |> collect .|> first |> Set, :p)
        @test haskey(pairs(init) |> collect .|> first |> Set, :q)

        # Market potential should be at least as large as cumulative adoption
        @test init.m >= sum(y)

        # p and q should be positive
        @test init.p > 0
        @test init.q > 0
    end

    @testset "Gompertz Initialization" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        init = Durbyn.Diffusion.gompertz_init(y)

        @test haskey(pairs(init) |> collect .|> first |> Set, :m)
        @test haskey(pairs(init) |> collect .|> first |> Set, :a)
        @test haskey(pairs(init) |> collect .|> first |> Set, :b)

        @test init.m >= sum(y)
        @test init.a > 0
        @test init.b > 0
    end

    @testset "GSGompertz Initialization" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        init = Durbyn.Diffusion.gsgompertz_init(y)

        @test haskey(pairs(init) |> collect .|> first |> Set, :m)
        @test haskey(pairs(init) |> collect .|> first |> Set, :a)
        @test haskey(pairs(init) |> collect .|> first |> Set, :b)
        @test haskey(pairs(init) |> collect .|> first |> Set, :c)

        @test init.m >= sum(y)
        @test init.a > 0
        @test init.b > 0
        @test init.c > 0
    end

    @testset "Weibull Initialization" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        init = Durbyn.Diffusion.weibull_init(y)

        @test haskey(pairs(init) |> collect .|> first |> Set, :m)
        @test haskey(pairs(init) |> collect .|> first |> Set, :a)
        @test haskey(pairs(init) |> collect .|> first |> Set, :b)

        @test init.m >= sum(y) * 0.99  # Allow small tolerance
        @test init.a > 0
        @test init.b > 0
    end

    @testset "Bass Model Fitting" begin
        # Generate synthetic data from known Bass curve
        true_curve = Durbyn.Diffusion.bass_curve(15, 1000.0, 0.03, 0.38)
        y = true_curve.adoption .+ randn(15) * 2  # Add noise
        y = max.(y, 0.1)  # Ensure positive

        fit = fit_diffusion(y, model_type=Bass)

        @test fit isa DiffusionFit
        @test fit.model_type == Bass
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :m)
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :p)
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :q)

        @test length(fit.fitted) == 15
        @test length(fit.cumulative) == 15
        @test length(fit.residuals) == 15
        @test fit.mse >= 0
    end

    @testset "Gompertz Model Fitting" begin
        true_curve = Durbyn.Diffusion.gompertz_curve(15, 1000.0, 5.0, 0.5)
        y = true_curve.adoption .+ randn(15) * 2
        y = max.(y, 0.1)

        fit = fit_diffusion(y, model_type=Gompertz)

        @test fit isa DiffusionFit
        @test fit.model_type == Gompertz
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :m)
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :a)
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :b)

        @test length(fit.fitted) == 15
        @test fit.mse >= 0
    end

    @testset "GSGompertz Model Fitting" begin
        true_curve = Durbyn.Diffusion.gsgompertz_curve(15, 1000.0, 0.1, 0.5, 1.0)
        y = true_curve.adoption .+ randn(15) * 2
        y = max.(y, 0.1)

        fit = fit_diffusion(y, model_type=GSGompertz)

        @test fit isa DiffusionFit
        @test fit.model_type == GSGompertz
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :m)
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :a)
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :b)
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :c)

        @test length(fit.fitted) == 15
        @test fit.mse >= 0
    end

    @testset "Weibull Model Fitting" begin
        true_curve = Durbyn.Diffusion.weibull_curve(15, 1000.0, 5.0, 2.0)
        y = true_curve.adoption .+ randn(15) * 2
        y = max.(y, 0.1)

        fit = fit_diffusion(y, model_type=Weibull)

        @test fit isa DiffusionFit
        @test fit.model_type == Weibull
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :m)
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :a)
        @test haskey(pairs(fit.params) |> collect .|> first |> Set, :b)

        @test length(fit.fitted) == 15
        @test fit.mse >= 0
    end

    @testset "Fixed Parameter Fitting" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        # Fix market potential
        fit = fit_diffusion(y, model_type=Bass, w=(m=1000.0, p=nothing, q=nothing))

        @test fit.params.m ≈ 1000.0 atol=1e-3

        # Fix p coefficient
        fit2 = fit_diffusion(y, model_type=Bass, w=(m=nothing, p=0.03, q=nothing))

        @test fit2.params.p ≈ 0.03 atol=1e-6
    end

    @testset "Loss Function Options" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        # L2 loss (default)
        fit_l2 = fit_diffusion(y, model_type=Bass, loss=2)
        @test fit_l2.loss == 2

        # L1 loss
        fit_l1 = fit_diffusion(y, model_type=Bass, loss=1)
        @test fit_l1.loss == 1
    end

    @testset "Cumulative vs Adoption Optimization" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        # Optimize on cumulative (default)
        fit_cum = fit_diffusion(y, model_type=Bass, cumulative=true)
        @test fit_cum.optim_cumulative == true

        # Optimize on adoption
        fit_adp = fit_diffusion(y, model_type=Bass, cumulative=false)
        @test fit_adp.optim_cumulative == false
    end

    @testset "Forecasting" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        fit = fit_diffusion(y, model_type=Bass)
        fc = forecast(fit, h=5)

        @test length(fc.mean) == 5
        @test length(fc.upper) == 2  # 80% and 95%
        @test length(fc.lower) == 2
        @test length(fc.upper[1]) == 5
        @test length(fc.upper[2]) == 5
        @test length(fc.lower[1]) == 5
        @test length(fc.lower[2]) == 5

        # Forecasts should be non-negative
        @test all(fc.mean .>= 0)

        # Upper bounds should be >= mean >= lower bounds
        for i in 1:5
            @test fc.upper[1][i] >= fc.mean[i]
            @test fc.mean[i] >= fc.lower[1][i]
        end
    end

    @testset "Forecast with Custom Confidence Levels" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        fit = fit_diffusion(y, model_type=Bass)
        fc = forecast(fit, h=5, level=[90])

        @test length(fc.upper) == 1
        @test length(fc.lower) == 1
        @test fc.level == [90]
    end

    @testset "Model Display" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        fit = fit_diffusion(y, model_type=Bass)

        # Test that show method works
        io = IOBuffer()
        show(io, fit)
        output = String(take!(io))

        @test occursin("Diffusion Model", output)
        @test occursin("Bass", output)
        @test occursin("Observations", output)
    end

    @testset "Grammar Integration" begin
        # Test DiffusionTerm creation
        term = Durbyn.Grammar.diffusion()
        @test term isa Durbyn.Grammar.DiffusionTerm

        # Test with model type
        term = Durbyn.Grammar.diffusion(model=:Bass)
        @test term.model_type == :Bass

        # Test with parameters
        term = Durbyn.Grammar.diffusion(model=:Bass, m=1000.0)
        @test term.m == 1000.0

        # Test formula parsing
        formula = @formula(adoption = diffusion())
        @test formula isa Durbyn.Grammar.ModelFormula
        @test formula.target == :adoption
    end

    @testset "Edge Cases" begin
        # Very short series
        y_short = [5.0, 10.0, 15.0]
        fit_short = fit_diffusion(y_short, model_type=Bass)
        @test fit_short isa DiffusionFit

        # Series with leading zeros
        y_zeros = [0.0, 0.0, 5.0, 15.0, 35.0, 65.0, 95.0]
        fit_zeros = fit_diffusion(y_zeros, model_type=Bass)
        @test fit_zeros isa DiffusionFit

        # Large values
        y_large = [5000.0, 15000.0, 35000.0, 65000.0, 95000.0]
        fit_large = fit_diffusion(y_large, model_type=Bass)
        @test fit_large isa DiffusionFit
    end

    @testset "Parameter Recovery" begin
        # Test that we can approximately recover known parameters
        true_m, true_p, true_q = 1000.0, 0.03, 0.38
        true_curve = Durbyn.Diffusion.bass_curve(20, true_m, true_p, true_q)
        y = true_curve.adoption

        fit = fit_diffusion(y, model_type=Bass)

        # Parameters should be reasonably close (within 20%)
        @test abs(fit.params.m - true_m) / true_m < 0.2
        @test abs(fit.params.p - true_p) / true_p < 0.5  # p is harder to estimate
        @test abs(fit.params.q - true_q) / true_q < 0.2
    end

    @testset "Diffusion Convenience Function" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        # Test that diffusion() works the same as fit_diffusion()
        fit1 = diffusion(y, model_type=Bass)
        fit2 = fit_diffusion(y, model_type=Bass)

        @test fit1.model_type == fit2.model_type
        @test fit1.params.m ≈ fit2.params.m
        @test fit1.params.p ≈ fit2.params.p
        @test fit1.params.q ≈ fit2.params.q
    end

end
