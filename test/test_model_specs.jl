using Test
using Durbyn
using Durbyn.ModelSpecs: DiffusionSpec, FittedDiffusion, ThetaSpec, FittedTheta,
    MeanfSpec, FittedMeanf, ArimaSpec, FittedArima,
    GroupedFittedModels, GroupedForecasts,
    fit, forecast, extract_metrics

# Bass-shaped adoption data for diffusion tests
const ADOPTION_DATA_MS = Float64[5, 10, 25, 45, 70, 85, 75, 50, 30, 15]

# AirPassengers subset for other tests
const AP_SHORT_MS = Float64[
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140
]

function make_grouped_data_ms()
    n = 24
    y = vcat(AP_SHORT_MS, AP_SHORT_MS .* 1.1)
    group = vcat(fill("A", n), fill("B", n))
    return (y = y, group = group)
end

# =================================================================
# Fix 1: DiffusionSpec fit/forecast integration
# =================================================================

@testset "DiffusionSpec fit and forecast" begin

    @testset "Basic fit" begin
        spec = DiffusionSpec(@formula(y = Durbyn.Grammar.diffusion()))
        data = (y = ADOPTION_DATA_MS,)
        fitted = fit(spec, data)

        @test fitted isa FittedDiffusion
        @test fitted.target_col == :y
        @test fitted.fit.model_type == Bass
        @test isfinite(fitted.fit.mse)
        @test length(fitted.fit.fitted) > 0
    end

    @testset "Specific model type" begin
        spec = DiffusionSpec(@formula(y = Durbyn.Grammar.diffusion(model=:Gompertz)))
        data = (y = ADOPTION_DATA_MS,)
        fitted = fit(spec, data)

        @test fitted isa FittedDiffusion
        @test fitted.fit.model_type == Gompertz
    end

    @testset "Fixed parameters" begin
        spec = DiffusionSpec(@formula(y = Durbyn.Grammar.diffusion(model=:Bass, m=500.0)))
        data = (y = ADOPTION_DATA_MS,)
        fitted = fit(spec, data)

        @test fitted isa FittedDiffusion
        @test fitted.fit.params.m == 500.0
    end

    @testset "Forecast" begin
        spec = DiffusionSpec(@formula(y = Durbyn.Grammar.diffusion()))
        data = (y = ADOPTION_DATA_MS,)
        fitted = fit(spec, data)
        fc = forecast(fitted, h=5)

        @test length(fc.mean) == 5
        @test all(isfinite, fc.mean)
        @test length(fc.lower) == 2  # default [80, 95]
        @test length(fc.upper) == 2
    end

    @testset "Forecast with custom levels" begin
        spec = DiffusionSpec(@formula(y = Durbyn.Grammar.diffusion()))
        data = (y = ADOPTION_DATA_MS,)
        fitted = fit(spec, data)
        fc = forecast(fitted, h=5, level=[90, 99])

        @test length(fc.lower) == 2
        @test length(fc.upper) == 2
    end

    @testset "extract_metrics" begin
        spec = DiffusionSpec(@formula(y = Durbyn.Grammar.diffusion()))
        data = (y = ADOPTION_DATA_MS,)
        fitted = fit(spec, data)
        metrics = extract_metrics(fitted)

        @test haskey(metrics, :mse)
        @test isfinite(metrics[:mse])
    end

    @testset "Grouped fit" begin
        n = 10
        y = vcat(ADOPTION_DATA_MS, ADOPTION_DATA_MS .* 1.2)
        group = vcat(fill("A", n), fill("B", n))
        data = (y = y, group = group)

        spec = DiffusionSpec(@formula(y = Durbyn.Grammar.diffusion()))
        fitted = fit(spec, data, groupby=:group)

        @test fitted isa GroupedFittedModels
        @test fitted.successful >= 1
    end

    @testset "Target validation" begin
        spec = DiffusionSpec(@formula(nonexistent = Durbyn.Grammar.diffusion()))
        data = (y = ADOPTION_DATA_MS,)
        @test_throws ArgumentError fit(spec, data)
    end

    @testset "Incompatible terms rejected" begin
        # Non-diffusion terms should throw, not silently fit Bass
        spec = DiffusionSpec(@formula(y = p(1) + q(1)))
        data = (y = ADOPTION_DATA_MS,)
        @test_throws ArgumentError fit(spec, data)
    end

    @testset "Grouped incompatible terms rejected" begin
        n = 10
        y = vcat(ADOPTION_DATA_MS, ADOPTION_DATA_MS .* 1.2)
        group = vcat(fill("A", n), fill("B", n))
        data = (y = y, group = group)

        spec = DiffusionSpec(@formula(y = p(1) + q(1)))
        @test_throws ArgumentError fit(spec, data, groupby=:group)
    end
end

# =================================================================
# Fix 2: Grouped forecast typed Dict newdata
# =================================================================

@testset "Grouped forecast typed Dict newdata" begin
    n = 36
    x_a = randn(n) .+ 5.0
    x_b = randn(n) .+ 3.0
    y_a = cumsum(randn(n)) .+ 100.0 .+ 0.5 .* x_a
    y_b = cumsum(randn(n)) .+ 200.0 .+ 0.3 .* x_b
    data = (
        y = vcat(y_a, y_b),
        x = vcat(x_a, x_b),
        group = vcat(fill("A", n), fill("B", n))
    )

    spec = ArimaSpec(@formula(y = p(1) + d(1) + q(0) + x))
    fitted = fit(spec, data, m=1, groupby=:group)

    @test fitted isa GroupedFittedModels

    # Test with a naturally typed Dict (not Dict{NamedTuple, Any})
    h = 5
    typed_newdata = Dict(
        (group="A",) => (x = randn(h) .+ 5.0,),
        (group="B",) => (x = randn(h) .+ 3.0,)
    )

    # This should NOT throw - it's a valid Dict with NamedTuple keys
    fc = forecast(fitted, h=h, newdata=typed_newdata)
    @test fc isa GroupedForecasts
    @test fc.successful >= 1

    # Dict newdata with wrong horizon should throw upfront
    bad_newdata = Dict(
        (group="A",) => (x = randn(3),),   # 3 != h=5
        (group="B",) => (x = randn(h),)
    )
    @test_throws ArgumentError forecast(fitted, h=h, newdata=bad_newdata)

    # Dict newdata with missing xreg column should throw upfront
    missing_col_newdata = Dict(
        (group="A",) => (z = randn(h),),    # :z not :x
        (group="B",) => (z = randn(h),)
    )
    @test_throws ArgumentError forecast(fitted, h=h, newdata=missing_col_newdata)
end

# =================================================================
# Fix 4: MeanfSpec biasadj propagation + forecast kwargs
# =================================================================

@testset "MeanfSpec biasadj propagation" begin

    @testset "Single series biasadj" begin
        spec = MeanfSpec(@formula(y = meanf_term()), biasadj=true)
        data = (y = AP_SHORT_MS,)
        fitted = fit(spec, data, m=12, lambda=0.5)

        @test fitted isa FittedMeanf
        @test fitted.fit.biasadj == true
    end

    @testset "biasadj kwarg override" begin
        spec = MeanfSpec(@formula(y = meanf_term()), biasadj=false)
        data = (y = AP_SHORT_MS,)
        fitted = fit(spec, data, m=12, lambda=0.5, biasadj=true)

        @test fitted.fit.biasadj == true
    end

    @testset "Grouped biasadj" begin
        gdata = make_grouped_data_ms()
        spec = MeanfSpec(@formula(y = meanf_term()), biasadj=true)
        fitted = fit(spec, gdata, m=12, groupby=:group, lambda=0.5)

        @test fitted isa GroupedFittedModels
        for key in fitted.groups
            model = fitted.models[key]
            if !(model isa Exception)
                @test model.fit.biasadj == true
                break
            end
        end
    end
end

@testset "MeanfSpec forecast kwargs forwarding" begin
    spec = MeanfSpec(@formula(y = meanf_term()))
    data = (y = AP_SHORT_MS,)
    fitted = fit(spec, data, m=12)

    # fan=true should be forwarded and produce many levels
    fc = forecast(fitted, h=12, fan=true)
    @test length(fc.mean) == 12
    @test length(fc.level) > 2

    # bootstrap should also work
    fc_boot = forecast(fitted, h=12, bootstrap=true, npaths=100)
    @test length(fc_boot.mean) == 12
end

# =================================================================
# Fix 5: Theta non-integer confidence levels
# =================================================================

@testset "Theta non-integer confidence levels" begin
    fit_result = Durbyn.Theta.auto_theta(AP_SHORT_MS, 12)

    @testset "Float64 levels work" begin
        fc = forecast(fit_result; h=6, level=[80.0, 95.0])
        @test length(fc.mean) == 6
        @test length(fc.lower) == 2
        @test length(fc.upper) == 2
    end

    @testset "Non-round levels round to Int for keys" begin
        fc = forecast(fit_result; h=6, level=[80.5])
        @test length(fc.mean) == 6
        @test length(fc.lower) == 1
        @test length(fc.upper) == 1
    end

    @testset "ModelSpecs Theta passes levels through" begin
        spec = ThetaSpec(@formula(y = theta()))
        data = (y = AP_SHORT_MS,)
        fitted = fit(spec, data, m=12)
        fc = forecast(fitted, h=6, level=[80.0, 95.0])

        @test length(fc.mean) == 6
        @test length(fc.lower) == 2
    end
end

# =================================================================
# Diffusion non-integer confidence levels
# =================================================================

@testset "Diffusion non-integer confidence levels" begin
    diff_fit = Durbyn.Diffusion.diffusion(ADOPTION_DATA_MS)

    @testset "Float64 levels work" begin
        fc = forecast(diff_fit; h=3, level=[80.0, 95.0])
        @test length(fc.mean) == 3
        @test length(fc.lower) == 2
    end

    @testset "Non-round levels" begin
        fc = forecast(diff_fit; h=3, level=[80.5])
        @test length(fc.mean) == 3
        @test length(fc.lower) == 1
    end
end
