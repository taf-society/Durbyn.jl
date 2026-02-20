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

        @test hasfield(typeof(init), :m)
        @test hasfield(typeof(init), :p)
        @test hasfield(typeof(init), :q)

        # All values should be finite (clamping to positive is caller's job)
        @test isfinite(init.m)
        @test isfinite(init.p)
        @test isfinite(init.q)

        # Market potential should be reasonable
        @test init.m >= sum(y)
    end

    @testset "Gompertz Initialization" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        init = Durbyn.Diffusion.gompertz_init(y)

        @test hasfield(typeof(init), :m)
        @test hasfield(typeof(init), :a)
        @test hasfield(typeof(init), :b)

        @test init.m >= sum(y)
        @test init.a > 0
        @test init.b > 0
    end

    @testset "GSGompertz Initialization" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        init = Durbyn.Diffusion.gsgompertz_init(y)

        @test hasfield(typeof(init), :m)
        @test hasfield(typeof(init), :a)
        @test hasfield(typeof(init), :b)
        @test hasfield(typeof(init), :c)

        @test init.m >= sum(y)
        @test init.a > 0
        @test init.b > 0
        @test init.c > 0
    end

    @testset "Weibull Initialization" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        init = Durbyn.Diffusion.weibull_init(y)

        @test hasfield(typeof(init), :m)
        @test hasfield(typeof(init), :a)
        @test hasfield(typeof(init), :b)

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
        @test hasfield(typeof(fit.params), :m)
        @test hasfield(typeof(fit.params), :p)
        @test hasfield(typeof(fit.params), :q)

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
        @test hasfield(typeof(fit.params), :m)
        @test hasfield(typeof(fit.params), :a)
        @test hasfield(typeof(fit.params), :b)

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
        @test hasfield(typeof(fit.params), :m)
        @test hasfield(typeof(fit.params), :a)
        @test hasfield(typeof(fit.params), :b)
        @test hasfield(typeof(fit.params), :c)

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
        @test hasfield(typeof(fit.params), :m)
        @test hasfield(typeof(fit.params), :a)
        @test hasfield(typeof(fit.params), :b)

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
        @test size(fc.upper, 2) == 2  # 80% and 95%
        @test size(fc.lower, 2) == 2
        @test size(fc.upper, 1) == 5
        @test size(fc.lower, 1) == 5

        # Forecasts should be non-negative
        @test all(fc.mean .>= 0)

        # Upper bounds should be >= mean >= lower bounds
        for i in 1:5
            @test fc.upper[i, 1] >= fc.mean[i]
            @test fc.mean[i] >= fc.lower[i, 1]
        end
    end

    @testset "Forecast with Custom Confidence Levels" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        fit = fit_diffusion(y, model_type=Bass)
        fc = forecast(fit, h=5, level=[90])

        @test size(fc.upper, 2) == 1
        @test size(fc.lower, 2) == 1
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

        # Test formula parsing (use qualified Grammar.diffusion to avoid ambiguity)
        formula = @formula(adoption = Durbyn.Grammar.diffusion())
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

    @testset "Leading Zeros (cleanlead parameter)" begin
        # Test data with leading zeros
        y_with_zeros = [0.0, 0.0, 5.0, 15.0, 35.0, 65.0, 95.0]
        y_no_zeros = [5.0, 15.0, 35.0, 65.0, 95.0]

        # Default behavior (cleanlead=true): fitted values for cleaned series only
        fit_clean = fit_diffusion(y_with_zeros, model_type=Bass, cleanlead=true)
        @test fit_clean.offset == 2
        @test length(fit_clean.y) == 5  # Cleaned data length
        @test length(fit_clean.y_original) == 7  # Original data length
        @test length(fit_clean.fitted) == 5  # Fitted values for cleaned series
        @test fit_clean.y == y_no_zeros

        # cleanlead=false: keep original length and fit on full series
        fit_full = fit_diffusion(y_with_zeros, model_type=Bass, cleanlead=false)
        @test fit_full.offset == 0
        @test length(fit_full.y) == 7  # Full data length
        @test length(fit_full.fitted) == 7  # Fitted values for full series

        # Note: Parameters may differ significantly because cleanlead=false fits
        # on the full series (including zeros) while cleanlead=true fits only on
        # the non-zero portion. We just verify both produce valid fits.
        @test fit_full.params.m > 0
        @test fit_full.params.p > 0
        @test fit_full.params.q > 0
    end

    @testset "Offset Field Persistence" begin
        y = [0.0, 0.0, 0.0, 5.0, 15.0, 35.0, 65.0, 95.0]

        fit = fit_diffusion(y, model_type=Bass, cleanlead=true)

        # Offset should be correctly stored
        @test fit.offset == 3

        # y_original should be the full original data
        @test fit.y_original == y

        # y should be the cleaned data
        @test length(fit.y) == length(y) - 3
        @test fit.y == y[4:end]
    end

    @testset "Bass Decomposition Convention" begin
        # Test that innovators + imitators = adoption
        curve = Durbyn.Diffusion.bass_curve(10, 1000.0, 0.03, 0.38)

        for t in 1:10
            @test curve.innovators[t] + curve.imitators[t] ≈ curve.adoption[t] atol=1e-10
        end

        # Test R convention: innovators uses current cumulative At
        # innovators[t] = p * (m - cumulative[t])
        m, p, q = 1000.0, 0.03, 0.38
        for t in 1:10
            expected_innovators = p * (m - curve.cumulative[t])
            @test curve.innovators[t] ≈ expected_innovators atol=1e-10
        end
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
        # Use qualified Durbyn.Diffusion.diffusion to avoid ambiguity with Grammar.diffusion
        fit1 = Durbyn.Diffusion.diffusion(y, model_type=Bass)
        fit2 = fit_diffusion(y, model_type=Bass)

        @test fit1.model_type == fit2.model_type
        @test fit1.params.m ≈ fit2.params.m
        @test fit1.params.p ≈ fit2.params.p
        @test fit1.params.q ≈ fit2.params.q
    end

    # =========================================================================
    # Additional tests for parameter recovery, forecasting, edge cases, etc.
    # =========================================================================

    @testset "Gompertz Parameter Recovery" begin
        true_m, true_a, true_b = 1000.0, 5.0, 0.5
        true_curve = Durbyn.Diffusion.gompertz_curve(20, true_m, true_a, true_b)
        y = true_curve.adoption

        fit = fit_diffusion(y, model_type=Gompertz)

        # Parameters should be reasonably close
        @test abs(fit.params.m - true_m) / true_m < 0.3
        @test fit.params.a > 0
        @test fit.params.b > 0
        @test fit.mse < 1.0  # Near-perfect data should have tiny MSE
    end

    @testset "GSGompertz Parameter Recovery" begin
        true_m, true_a, true_b, true_c = 1000.0, 0.1, 0.5, 1.0
        true_curve = Durbyn.Diffusion.gsgompertz_curve(20, true_m, true_a, true_b, true_c)
        y = true_curve.adoption

        fit = fit_diffusion(y, model_type=GSGompertz)

        @test fit.params.m > 0
        @test fit.params.a > 0
        @test fit.params.b > 0
        @test fit.params.c > 0
        # GSGompertz has 4 parameters and can be ill-conditioned; allow larger MSE
        @test fit.mse < 10.0
    end

    @testset "Weibull Parameter Recovery" begin
        true_m, true_a, true_b = 1000.0, 5.0, 2.0
        true_curve = Durbyn.Diffusion.weibull_curve(20, true_m, true_a, true_b)
        y = true_curve.adoption

        fit = fit_diffusion(y, model_type=Weibull)

        @test abs(fit.params.m - true_m) / true_m < 0.3
        @test fit.params.a > 0
        @test fit.params.b > 0
        @test fit.mse < 1.0
    end

    @testset "Forecast for Gompertz" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        fit = fit_diffusion(y, model_type=Gompertz)
        fc = forecast(fit, h=5)

        @test length(fc.mean) == 5
        @test size(fc.upper, 2) == 2
        @test size(fc.lower, 2) == 2
        for i in 1:5
            @test fc.upper[i, 1] >= fc.mean[i]
            @test fc.mean[i] >= fc.lower[i, 1]
        end
    end

    @testset "Forecast for GSGompertz" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        fit = fit_diffusion(y, model_type=GSGompertz)
        fc = forecast(fit, h=5)

        @test length(fc.mean) == 5
        @test size(fc.upper, 2) == 2
        @test size(fc.lower, 2) == 2
        for i in 1:5
            @test fc.upper[i, 1] >= fc.mean[i]
            @test fc.mean[i] >= fc.lower[i, 1]
        end
    end

    @testset "Forecast for Weibull" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        fit = fit_diffusion(y, model_type=Weibull)
        fc = forecast(fit, h=5)

        @test length(fc.mean) == 5
        @test size(fc.upper, 2) == 2
        @test size(fc.lower, 2) == 2
        for i in 1:5
            @test fc.upper[i, 1] >= fc.mean[i]
            @test fc.mean[i] >= fc.lower[i, 1]
        end
    end

    @testset "Forecast Lower Bounds Non-negative" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        fit = fit_diffusion(y, model_type=Bass)
        fc = forecast(fit, h=10)

        # Lower bounds should be clamped at 0
        for lv_idx in 1:size(fc.lower, 2)
            @test all(fc.lower[:, lv_idx] .>= 0.0)
        end
    end

    @testset "Predict Function" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        fit = fit_diffusion(y, model_type=Bass)

        # Predict for periods 1-20
        pred = Durbyn.Diffusion.predict(fit, 1:20)
        @test length(pred.adoption) == 20
        @test length(pred.cumulative) == 20

        # Cumulative should be monotonically increasing
        for i in 2:20
            @test pred.cumulative[i] >= pred.cumulative[i-1]
        end

        # Predict values at in-sample periods should match fitted values
        pred_insample = Durbyn.Diffusion.predict(fit, 1:10)
        for i in 1:10
            @test pred_insample.adoption[i] ≈ fit.fitted[i] atol=1e-10
        end
    end

    @testset "Predict Validation" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        fit = fit_diffusion(y, model_type=Bass)

        # Non-integer time points should error
        @test_throws ArgumentError Durbyn.Diffusion.predict(fit, [1.5, 2.5])

        # Non-positive time points should error
        @test_throws ArgumentError Durbyn.Diffusion.predict(fit, [0, 1, 2])
        @test_throws ArgumentError Durbyn.Diffusion.predict(fit, [-1, 1, 2])

        # Inf/NaN time points should throw ArgumentError (not InexactError)
        @test_throws ArgumentError Durbyn.Diffusion.predict(fit, [Inf])
        @test_throws ArgumentError Durbyn.Diffusion.predict(fit, [-Inf])
        @test_throws ArgumentError Durbyn.Diffusion.predict(fit, [NaN])
        @test_throws ArgumentError Durbyn.Diffusion.predict(fit, [1.0, Inf, 3.0])
    end

    @testset "Preset Initialization" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        for model_type in [Bass, Gompertz, GSGompertz, Weibull]
            fit = fit_diffusion(y, model_type=model_type, initpar="preset")
            @test fit isa DiffusionFit
            @test fit.model_type == model_type
            @test fit.params.m > 0
        end
    end

    @testset "Numeric Initpar" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        # Bass: 3 params
        fit_bass = fit_diffusion(y, model_type=Bass, initpar=[500.0, 0.03, 0.38])
        @test fit_bass isa DiffusionFit

        # Gompertz: 3 params
        fit_gomp = fit_diffusion(y, model_type=Gompertz, initpar=[500.0, 5.0, 0.5])
        @test fit_gomp isa DiffusionFit

        # GSGompertz: 4 params
        fit_gsg = fit_diffusion(y, model_type=GSGompertz, initpar=[500.0, 0.1, 0.5, 1.0])
        @test fit_gsg isa DiffusionFit

        # Weibull: 3 params
        fit_wb = fit_diffusion(y, model_type=Weibull, initpar=[500.0, 5.0, 2.0])
        @test fit_wb isa DiffusionFit

        # Wrong number of params should error
        @test_throws ArgumentError fit_diffusion(y, model_type=Bass, initpar=[500.0, 0.03])
        @test_throws ArgumentError fit_diffusion(y, model_type=GSGompertz, initpar=[500.0, 0.1, 0.5])
    end

    @testset "mscal=false" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        fit_scaled = fit_diffusion(y, model_type=Bass, mscal=true)
        fit_unscaled = fit_diffusion(y, model_type=Bass, mscal=false)

        # Both should produce valid fits
        @test fit_scaled isa DiffusionFit
        @test fit_unscaled isa DiffusionFit
        @test fit_scaled.params.m > 0
        @test fit_unscaled.params.m > 0
    end

    @testset "Linearise Spelling" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        # British spelling should be accepted
        fit = fit_diffusion(y, model_type=Bass, initpar="linearise")
        @test fit isa DiffusionFit
    end

    @testset "NaN/Inf Input Validation" begin
        # NaN in data
        y_nan = [5.0, NaN, 35.0, 65.0, 95.0]
        @test_throws ArgumentError fit_diffusion(y_nan, model_type=Bass)

        # Inf in data
        y_inf = [5.0, Inf, 35.0, 65.0, 95.0]
        @test_throws ArgumentError fit_diffusion(y_inf, model_type=Bass)

        # -Inf in data
        y_neginf = [5.0, -Inf, 35.0, 65.0, 95.0]
        @test_throws ArgumentError fit_diffusion(y_neginf, model_type=Bass)
    end

    @testset "Too Few Observations" begin
        # Less than 3 observations
        @test_throws ArgumentError fit_diffusion([5.0, 10.0], model_type=Bass)

        # 3 observations but only 2 non-zero after cleanlead
        @test_throws ArgumentError fit_diffusion([0.0, 0.0, 0.0, 5.0, 10.0], model_type=Bass, cleanlead=false)
    end

    @testset "Integer Input" begin
        # Integer vector should be accepted and converted
        y = [5, 10, 25, 45, 70, 85, 75, 50, 30, 15]
        fit = fit_diffusion(y, model_type=Bass)
        @test fit isa DiffusionFit
        @test fit.params.m > 0
    end

    @testset "Single Leading Zero" begin
        y = [0.0, 5.0, 15.0, 35.0, 65.0, 95.0]
        fit = fit_diffusion(y, model_type=Bass, cleanlead=true)
        @test fit.offset == 1
        @test length(fit.y) == 5
    end

    @testset "Very Small Values" begin
        y = [0.001, 0.002, 0.005, 0.01, 0.008, 0.005, 0.003]
        fit = fit_diffusion(y, model_type=Bass)
        @test fit isa DiffusionFit
        @test fit.params.m > 0
    end

    @testset "Monotonically Increasing (No Peak)" begin
        y = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        fit = fit_diffusion(y, model_type=Bass)
        @test fit isa DiffusionFit
        @test fit.params.m > 0
    end

    @testset "Fixed Parameters for Non-Bass Models" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        # Gompertz: fix m
        fit_gomp = fit_diffusion(y, model_type=Gompertz, w=(m=1000.0, a=nothing, b=nothing))
        @test fit_gomp.params.m ≈ 1000.0 atol=1e-3

        # GSGompertz: fix m
        fit_gsg = fit_diffusion(y, model_type=GSGompertz, w=(m=1000.0, a=nothing, b=nothing, c=nothing))
        @test fit_gsg.params.m ≈ 1000.0 atol=1e-3

        # Weibull: fix m
        fit_wb = fit_diffusion(y, model_type=Weibull, w=(m=1000.0, a=nothing, b=nothing))
        @test fit_wb.params.m ≈ 1000.0 atol=1e-3
    end

    @testset "Fully Fixed Parameters for All Models" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        # Bass: all fixed
        fit_bass = fit_diffusion(y, model_type=Bass, w=(m=500.0, p=0.03, q=0.38))
        @test fit_bass.params.m ≈ 500.0
        @test fit_bass.params.p ≈ 0.03
        @test fit_bass.params.q ≈ 0.38

        # Gompertz: all fixed
        fit_gomp = fit_diffusion(y, model_type=Gompertz, w=(m=500.0, a=5.0, b=0.5))
        @test fit_gomp.params.m ≈ 500.0
        @test fit_gomp.params.a ≈ 5.0
        @test fit_gomp.params.b ≈ 0.5

        # GSGompertz: all fixed
        fit_gsg = fit_diffusion(y, model_type=GSGompertz, w=(m=500.0, a=0.1, b=0.5, c=1.0))
        @test fit_gsg.params.m ≈ 500.0
        @test fit_gsg.params.a ≈ 0.1
        @test fit_gsg.params.b ≈ 0.5
        @test fit_gsg.params.c ≈ 1.0

        # Weibull: all fixed
        fit_wb = fit_diffusion(y, model_type=Weibull, w=(m=500.0, a=5.0, b=2.0))
        @test fit_wb.params.m ≈ 500.0
        @test fit_wb.params.a ≈ 5.0
        @test fit_wb.params.b ≈ 2.0
    end

    @testset "Show Method for All Model Types" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        for (model_type, expected_name) in [(Bass, "Bass"), (Gompertz, "Gompertz"),
                                             (GSGompertz, "GSGompertz"), (Weibull, "Weibull")]
            fit = fit_diffusion(y, model_type=model_type)
            io = IOBuffer()
            show(io, fit)
            output = String(take!(io))

            @test occursin("Diffusion Model", output)
            @test occursin(expected_name, output)
            @test occursin("Observations", output)
            @test occursin("MSE", output)
            @test occursin("Parameters", output)
        end
    end

    @testset "Show Method with Offset" begin
        y = [0.0, 0.0, 5.0, 15.0, 35.0, 65.0, 95.0]
        fit = fit_diffusion(y, model_type=Bass, cleanlead=true)
        io = IOBuffer()
        show(io, fit)
        output = String(take!(io))

        @test occursin("leading zeros removed", output)
    end

    @testset "Cumulative Adoption Consistency" begin
        # For all model types: cumulative[end] should be close to sum of adoption
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        for model_type in [Bass, Gompertz, GSGompertz, Weibull]
            fit = fit_diffusion(y, model_type=model_type)

            # fitted cumulative should equal cumsum of fitted adoption
            expected_cum = cumsum(fit.fitted)
            for i in 1:length(fit.fitted)
                @test fit.cumulative[i] ≈ expected_cum[i] atol=1e-8
            end
        end
    end

    @testset "Residuals Consistency" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        for model_type in [Bass, Gompertz, GSGompertz, Weibull]
            fit = fit_diffusion(y, model_type=model_type)

            # residuals = actual - fitted
            for i in 1:length(y)
                @test fit.residuals[i] ≈ (fit.y[i] - fit.fitted[i]) atol=1e-10
            end

            # MSE should be mean of squared residuals
            @test fit.mse ≈ sum(fit.residuals .^ 2) / length(fit.residuals) atol=1e-10
        end
    end

    @testset "L3 Loss Function" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        fit = fit_diffusion(y, model_type=Bass, loss=3)
        @test fit isa DiffusionFit
        @test fit.loss == 3
    end

    @testset "Forecast Method String" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        for (model_type, name) in [(Bass, "Bass"), (Gompertz, "Gompertz"),
                                    (GSGompertz, "GSGompertz"), (Weibull, "Weibull")]
            fit = fit_diffusion(y, model_type=model_type)
            fc = forecast(fit, h=3)
            @test occursin(name, fc.method)
        end
    end

    @testset "Curve Generation Adoption Sums to Cumulative" begin
        # For all curve types: cumsum(adoption) == cumulative
        curve_bass = Durbyn.Diffusion.bass_curve(20, 1000.0, 0.03, 0.38)
        @test cumsum(curve_bass.adoption) ≈ curve_bass.cumulative atol=1e-8

        curve_gomp = Durbyn.Diffusion.gompertz_curve(20, 1000.0, 5.0, 0.5)
        @test cumsum(curve_gomp.adoption) ≈ curve_gomp.cumulative atol=1e-8

        curve_gsg = Durbyn.Diffusion.gsgompertz_curve(20, 1000.0, 0.1, 0.5, 1.0)
        @test cumsum(curve_gsg.adoption) ≈ curve_gsg.cumulative atol=1e-8

        curve_wb = Durbyn.Diffusion.weibull_curve(20, 1000.0, 5.0, 2.0)
        @test cumsum(curve_wb.adoption) ≈ curve_wb.cumulative atol=1e-8
    end

    @testset "get_curve Dispatch" begin
        params_bass = (m=1000.0, p=0.03, q=0.38)
        curve = Durbyn.Diffusion.get_curve(Bass, 10, params_bass)
        @test length(curve.cumulative) == 10
        @test length(curve.adoption) == 10

        params_gomp = (m=1000.0, a=5.0, b=0.5)
        curve = Durbyn.Diffusion.get_curve(Gompertz, 10, params_gomp)
        @test length(curve.cumulative) == 10

        params_gsg = (m=1000.0, a=0.1, b=0.5, c=1.0)
        curve = Durbyn.Diffusion.get_curve(GSGompertz, 10, params_gsg)
        @test length(curve.cumulative) == 10

        params_wb = (m=1000.0, a=5.0, b=2.0)
        curve = Durbyn.Diffusion.get_curve(Weibull, 10, params_wb)
        @test length(curve.cumulative) == 10
    end

    @testset "Preset Init Values Match R" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        y_sum = sum(y)

        # Bass preset: (0.5, 0.5, 0.5) with m scaled
        init = Durbyn.Diffusion.preset_init(Bass, y)
        @test init.m ≈ 0.5 * 10 * y_sum
        @test init.p ≈ 0.5
        @test init.q ≈ 0.5

        # Gompertz preset: (1, 1, 1) with m scaled
        init = Durbyn.Diffusion.preset_init(Gompertz, y)
        @test init.m ≈ 1.0 * 10 * y_sum
        @test init.a ≈ 1.0
        @test init.b ≈ 1.0

        # GSGompertz preset: (0.5, 0.5, 0.5, 0.5) with m scaled
        init = Durbyn.Diffusion.preset_init(GSGompertz, y)
        @test init.m ≈ 0.5 * 10 * y_sum
        @test init.a ≈ 0.5
        @test init.b ≈ 0.5
        @test init.c ≈ 0.5

        # Weibull preset: (0.5, 0.5, 0.5) with m scaled
        init = Durbyn.Diffusion.preset_init(Weibull, y)
        @test init.m ≈ 0.5 * 10 * y_sum
        @test init.a ≈ 0.5
        @test init.b ≈ 0.5
    end

    @testset "Preset Init Without mscal" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        init = Durbyn.Diffusion.preset_init(Bass, y; mscal=false)
        @test init.m ≈ 0.5
        @test init.p ≈ 0.5
        @test init.q ≈ 0.5
    end

    @testset "Cleanzero Utility" begin
        # Leading zeros
        cleaned, offset = Durbyn.Diffusion._cleanzero([0.0, 0.0, 5.0, 10.0])
        @test cleaned == [5.0, 10.0]
        @test offset == 2

        # No leading zeros
        cleaned, offset = Durbyn.Diffusion._cleanzero([5.0, 10.0, 15.0])
        @test cleaned == [5.0, 10.0, 15.0]
        @test offset == 0

        # All zeros
        cleaned, offset = Durbyn.Diffusion._cleanzero([0.0, 0.0, 0.0])
        @test offset == 0  # Returns original when all zeros

        # Trailing zeros
        cleaned, offset = Durbyn.Diffusion._cleanzero([5.0, 10.0, 0.0, 0.0]; lead=false)
        @test cleaned == [5.0, 10.0]
        @test offset == 2

        # Single leading zero
        cleaned, offset = Durbyn.Diffusion._cleanzero([0.0, 5.0, 10.0])
        @test cleaned == [5.0, 10.0]
        @test offset == 1
    end

    @testset "Bass Init Complex Roots" begin
        # Data that produces negative discriminant in bass_init regression
        # Complex roots use Re(-c1/(2*c2))
        y_complex = [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]
        init = Durbyn.Diffusion.bass_init(y_complex)

        # bass_init: no clamping, values may be negative (caller handles)
        @test isfinite(init.m)
        @test isfinite(init.p)
        @test isfinite(init.q)

        # Verify it uses -c1/(2*c2) for complex discriminant, NOT Y[end]*1.5 fallback
        Y = cumsum(Float64.(y_complex))
        cf = hcat(ones(10), Y, Y .^ 2) \ Float64.(y_complex)
        disc = cf[2]^2 - 4 * cf[3] * cf[1]
        if disc < 0 && abs(cf[3]) >= 1e-12
            expected_m = -cf[2] / (2 * cf[3])
            @test init.m ≈ expected_m rtol=1e-10
        end

        # p and q should be consistent with the regression coefficients
        if abs(init.m) > 1e-12
            @test init.p ≈ cf[1] / init.m rtol=1e-10
            @test init.q ≈ cf[2] + init.p rtol=1e-10
        end

        # Even with negative init, fit_diffusion should still work (caller clamps)
        fit = fit_diffusion(y_complex, model_type=Bass)
        @test fit isa DiffusionFit
        @test fit.params.m > 0
        @test fit.params.p > 0
        @test fit.params.q > 0
    end

    @testset "Bass Init Degenerate Quadratic (c2 ≈ 0)" begin
        # Nearly linear data makes c2 ≈ 0 in the regression
        y_linear = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        init = Durbyn.Diffusion.bass_init(y_linear)

        @test isfinite(init.m)
        @test isfinite(init.p)
        @test isfinite(init.q)
    end

    @testset "Loss = -1 Matches L1" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        fit_l1 = fit_diffusion(y, model_type=Bass, loss=1)
        fit_neg1 = fit_diffusion(y, model_type=Bass, loss=-1)

        # loss=-1 should behave identically to loss=1 (both are L1)
        @test fit_neg1.params.m ≈ fit_l1.params.m rtol=1e-6
        @test fit_neg1.params.p ≈ fit_l1.params.p rtol=1e-6
        @test fit_neg1.params.q ≈ fit_l1.params.q rtol=1e-6
    end

    @testset "Numeric Initpar Clamped to Bounds" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        # Negative init values should be clamped (not cause stall/failure)
        fit = fit_diffusion(y, model_type=Bass, initpar=[-10.0, -0.5, -0.5])
        @test fit isa DiffusionFit
        @test fit.params.m > 0
        @test fit.params.p > 0
        @test fit.params.q > 0

        # Zero init values should also be handled
        fit2 = fit_diffusion(y, model_type=Bass, initpar=[0.0, 0.0, 0.0])
        @test fit2 isa DiffusionFit
        @test fit2.params.m > 0
    end

    @testset "Forecast h Validation" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        fit = fit_diffusion(y, model_type=Bass)

        # h=0 should error
        @test_throws ArgumentError forecast(fit, h=0)

        # h<0 should error
        @test_throws ArgumentError forecast(fit, h=-1)
        @test_throws ArgumentError forecast(fit, h=-10)

        # h=1 should work
        fc = forecast(fit, h=1)
        @test length(fc.mean) == 1
    end

    @testset "Forecast Level Validation" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        fit = fit_diffusion(y, model_type=Bass)

        # level=0 should error
        @test_throws ArgumentError forecast(fit, h=5, level=[0])

        # level=100 should error
        @test_throws ArgumentError forecast(fit, h=5, level=[100])

        # level>100 should error
        @test_throws ArgumentError forecast(fit, h=5, level=[150])

        # level<0 should error
        @test_throws ArgumentError forecast(fit, h=5, level=[-10])

        # Valid levels should work
        fc = forecast(fit, h=5, level=[50, 80, 95, 99])
        @test size(fc.upper, 2) == 4
        @test size(fc.lower, 2) == 4
    end

    @testset "Bass Init No-Clamp Consistency" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]
        init = Durbyn.Diffusion.bass_init(y)

        # p and q should be consistent with m (derived from same regression)
        Y = cumsum(Float64.(y))
        X = hcat(ones(10), Y, Y .^ 2)
        cf = X \ Float64.(y)

        if abs(init.m) > 1e-12
            @test init.p ≈ cf[1] / init.m rtol=1e-10
            @test init.q ≈ cf[2] + init.p rtol=1e-10
        end
    end

    @testset "Large Values All Model Types" begin
        y_large = [5000.0, 15000.0, 35000.0, 65000.0, 95000.0, 105000.0, 95000.0]

        for model_type in [Bass, Gompertz, GSGompertz, Weibull]
            fit = fit_diffusion(y_large, model_type=model_type)
            @test fit isa DiffusionFit
            @test fit.params.m > 0
            @test isfinite(fit.mse)
        end
    end

    @testset "Linearization Fallback to Preset" begin
        y_tricky = [1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15]

        for model_type in [Gompertz, GSGompertz, Weibull]
            fit = fit_diffusion(y_tricky, model_type=model_type)
            @test fit isa DiffusionFit
            @test isfinite(fit.mse)
        end
    end

    @testset "Cumulative=false in Bass Init for Gompertz/GSGompertz" begin
        y = [5.0, 15.0, 35.0, 65.0, 95.0, 105.0, 95.0, 70.0, 45.0, 25.0]

        fit_gomp = fit_diffusion(y, model_type=Gompertz)
        @test fit_gomp isa DiffusionFit
        @test fit_gomp.params.m > 0
        @test isfinite(fit_gomp.mse)

        fit_gsg = fit_diffusion(y, model_type=GSGompertz)
        @test fit_gsg isa DiffusionFit
        @test fit_gsg.params.m > 0
        @test isfinite(fit_gsg.mse)
    end

end
