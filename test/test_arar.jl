using Test
using Durbyn
import Durbyn.Generics: Forecast
using Durbyn.Ararma: ARAR, ArarmaModel, ararma, auto_ararma
# Import Grammar.arar for use in @formula
using Durbyn.Grammar: arar
using Durbyn.ModelSpecs: ArarSpec, FittedArar, GroupedFittedModels, GroupedForecasts

@testset "Durbyn.Ararma - ARAR tests" begin
    ap = air_passengers()

    @testset "Basic fit & forecast (equal depths/lags)" begin
        # Use fully qualified name for array-based arar
        fit = Durbyn.Ararma.arar(ap; max_ar_depth = 13, max_lag = 13)
        @test fit isa ARAR

        @testset "Forecast horizons & levels" begin
            fc1 = forecast(fit; h = 12)
            @test length(fc1.mean) == 12
            @test fc1 isa Forecast

            fc2 = forecast(fit; h = 12, level = [80, 90, 95])
            @test size(fc2.upper) == (12,3)

            fc3 = forecast(fit; h = 12, level = [80])
            @test size(fc3.upper) == (12,1)
        end
    end

    @testset "Handling of `nothing` parameters" begin
        fit1 = Durbyn.Ararma.arar(ap; max_ar_depth = nothing, max_lag = 29)
        @test fit1 isa ARAR
        fc1 = forecast(fit1; h = 12)
        @test length(fc1.mean) == 12

        fit2 = Durbyn.Ararma.arar(ap; max_ar_depth = 26, max_lag = nothing)
        @test fit2 isa ARAR
        fc2 = forecast(fit2; h = 12)
        @test length(fc2.mean) == 12
    end

    @testset "Parameter clamping" begin
        # When max_ar_depth > max_lag, max_ar_depth gets clamped (with warning)
        # This should NOT throw, but warn and clamp
        fit = Durbyn.Ararma.arar(ap; max_ar_depth = 13, max_lag = 12)
        @test fit isa ARAR

        # max_ar_depth < 4 should still throw
        @test_throws ArgumentError Durbyn.Ararma.arar(ap; max_ar_depth = 3, max_lag = 12)
    end

    @testset "Short series behavior" begin
        ap1 = ap[1:11]
        fit_s1 = Durbyn.Ararma.arar(ap1)
        @test fit_s1 isa ARAR
        fc_s1 = forecast(fit_s1; h = 12)
        @test length(fc_s1.mean) == 12

        ap2 = ap[1:9]

        @test_logs (:warn,) begin
            fit_s2 = Durbyn.Ararma.arar(ap2)
            fc_s2 = forecast(fit_s2; h = 12)
            @test length(fc_s2.mean) == 12
        end
    end

    @testset "Formula interface" begin
        data = (sales = ap,)

        @testset "Basic formula with defaults" begin
            formula = @formula(sales = arar())
            fit = Durbyn.Ararma.arar(formula, data)
            @test fit isa ARAR
            @test length(fit.y) == length(ap)

            fc = forecast(fit; h = 12)
            @test length(fc.mean) == 12
            @test fc isa Forecast
        end

        @testset "Formula with max_ar_depth" begin
            formula = @formula(sales = arar(max_ar_depth=15))
            fit = Durbyn.Ararma.arar(formula, data)
            @test fit isa ARAR

            fc = forecast(fit; h = 12)
            @test length(fc.mean) == 12
        end

        @testset "Formula with both parameters" begin
            formula = @formula(sales = arar(max_ar_depth=20, max_lag=20))
            fit = Durbyn.Ararma.arar(formula, data)
            @test fit isa ARAR

            fc = forecast(fit; h = 12)
            @test length(fc.mean) == 12
        end

        @testset "Formula with missing target" begin
            formula = @formula(price = arar())
            @test_throws ArgumentError Durbyn.Ararma.arar(formula, data)
        end

        @testset "Formula with invalid data type" begin
            formula = @formula(sales = arar())
            @test_throws ArgumentError Durbyn.Ararma.arar(formula, ap)  # Not a table
        end

        @testset "Formula validation" begin
            # Test max_ar_depth validation
            @test_throws ArgumentError @formula(y = arar(max_ar_depth=3))

            # Test max_lag validation
            @test_throws ArgumentError @formula(y = arar(max_lag=-1))
        end
    end

    @testset "ModelSpecs interface" begin
        data = (sales = ap,)
        spec = ArarSpec(@formula(sales = arar()))

        fitted_spec = fit(spec, data)
        @test fitted_spec isa FittedArar
        @test fitted_spec.fit isa ARAR

        fc = forecast(fitted_spec; h = 6)
        @test length(fc.mean) == 6

        grouped_data = (
            sales = vcat(ap, ap .* 1.05),
            region = vcat(fill("north", length(ap)), fill("south", length(ap)))
        )

        grouped_fit = fit(spec, grouped_data; groupby = :region, parallel = false)
        @test grouped_fit isa GroupedFittedModels
        successful = Durbyn.successful_models(grouped_fit)
        @test length(successful) == 2
        for model in values(successful)
            @test model isa FittedArar
        end

        grouped_fc = forecast(grouped_fit; h = 6, parallel = false)
        @test grouped_fc isa GroupedForecasts
        @test grouped_fc.successful == 2
        for fc_group in values(grouped_fc.forecasts)
            @test fc_group isa Forecast
            @test length(fc_group.mean) == 6
        end

        panel = PanelData(grouped_data; groupby = :region)
        models = model(
            ArarSpec(@formula(sales = arar())),
            SesSpec(@formula(sales = ses())),
            names = ["arar_model", "ses_model"]
        )
        fitted_collection = fit(models, panel)
        fc_collection = forecast(fitted_collection; h = 4)
        @test "arar_model" in fc_collection.names
        arar_fc = fc_collection["arar_model"]
        @test arar_fc isa GroupedForecasts
        @test arar_fc.successful == 2
    end
end
