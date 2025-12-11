using Test
using Durbyn
using Durbyn.Grammar: @formula, TbatsTerm
using Durbyn.Grammar: tbats as grammar_tbats
using Durbyn.ModelSpecs: TbatsSpec, FittedTbats, fit, forecast, GroupedFittedModels
using Durbyn.Tbats: TBATSModel
import Durbyn.Tbats
import Durbyn.Generics: Forecast

# Alias for direct tbats call with formula
const tbats_direct = Durbyn.Tbats.tbats

@testset "TBATS Formula Interface" begin

    @testset "TbatsTerm creation via grammar" begin
        # Default (auto-select everything)
        term = grammar_tbats()
        @test term isa TbatsTerm
        @test isnothing(term.seasonal_periods)
        @test isnothing(term.k)
        @test isnothing(term.use_box_cox)
        @test isnothing(term.use_trend)
        @test isnothing(term.use_damped_trend)
        @test isnothing(term.use_arma_errors)

        # Single non-integer seasonal period
        term = grammar_tbats(seasonal_periods=52.18)
        @test term.seasonal_periods == 52.18

        # Multiple seasonal periods (non-integer)
        term = grammar_tbats(seasonal_periods=[7, 365.25])
        @test term.seasonal_periods == [7, 365.25]

        # With explicit Fourier orders
        term = grammar_tbats(seasonal_periods=[7, 365.25], k=[3, 10])
        @test term.seasonal_periods == [7, 365.25]
        @test term.k == [3, 10]

        # All component options
        term = grammar_tbats(
            seasonal_periods=52.18,
            use_box_cox=true,
            use_trend=true,
            use_damped_trend=false,
            use_arma_errors=true
        )
        @test term.seasonal_periods == 52.18
        @test term.use_box_cox == true
        @test term.use_trend == true
        @test term.use_damped_trend == false
        @test term.use_arma_errors == true
    end

    @testset "TbatsTerm validation" begin
        # seasonal_periods must be positive
        @test_throws ArgumentError grammar_tbats(seasonal_periods=0)
        @test_throws ArgumentError grammar_tbats(seasonal_periods=-1)
        @test_throws ArgumentError grammar_tbats(seasonal_periods=[7, -1])

        # k must be >= 1
        @test_throws ArgumentError grammar_tbats(seasonal_periods=7, k=0)
        @test_throws ArgumentError grammar_tbats(seasonal_periods=[7, 365.25], k=[3, 0])

        # k length must match seasonal_periods length
        @test_throws ArgumentError grammar_tbats(seasonal_periods=[7, 365.25], k=[3])
        @test_throws ArgumentError grammar_tbats(seasonal_periods=7, k=[3, 10])
    end

    @testset "TbatsSpec creation" begin
        formula = @formula(sales = tbats())
        spec = TbatsSpec(formula)
        @test spec isa TbatsSpec
        @test spec.formula === formula
        @test isempty(spec.options)

        # With options
        spec = TbatsSpec(formula, bc_lower=0.0, bc_upper=1.5)
        @test spec.options[:bc_lower] == 0.0
        @test spec.options[:bc_upper] == 1.5
    end

    @testset "Single series fit with TbatsSpec" begin
        # Create test data with seasonal pattern
        n = 156  # 3 years of weekly data
        t = 1:n
        y = 100.0 .+ 10.0 .* sin.(2π .* t ./ 52) .+ 2.0 .* randn(n)

        data = (sales = y,)

        # Basic TBATS with defaults (may return BATS or TBATS depending on detected seasonality)
        spec = TbatsSpec(@formula(sales = tbats()))
        fitted_model = fit(spec, data)

        @test fitted_model isa FittedTbats
        @test fitted_model.spec === spec
        @test fitted_model.target_col == :sales
        # TBATS may fall back to BATS when no seasonality is specified or detected
        @test fitted_model.fit isa Union{TBATSModel, Durbyn.Bats.BATSModel}

        # Check fitted model properties
        @test length(fitted_model.fit.fitted_values) == n
        @test length(fitted_model.fit.errors) == n
        @test all(isfinite, fitted_model.fit.fitted_values)
        @test all(isfinite, fitted_model.fit.errors)
        @test isfinite(fitted_model.fit.AIC)

        # TBATS with seasonal period
        spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=52)))
        fitted_model = fit(spec, data)
        @test fitted_model isa FittedTbats
        @test fitted_model.fit.seasonal_periods == [52]

        # TBATS with forced components
        spec = TbatsSpec(@formula(sales = tbats(
            seasonal_periods=52,
            use_box_cox=false,
            use_trend=true,
            use_arma_errors=false
        )))
        fitted_model = fit(spec, data)
        @test fitted_model isa FittedTbats
        @test isnothing(fitted_model.fit.lambda)  # Box-Cox disabled
    end

    @testset "Forecasting with FittedTbats" begin
        n = 156
        t = 1:n
        y = 100.0 .+ 10.0 .* sin.(2π .* t ./ 52) .+ 2.0 .* randn(n)
        data = (sales = y,)

        spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=52)))
        fitted_model = fit(spec, data)

        # Basic forecast
        fc = forecast(fitted_model, h=12)
        @test fc isa Forecast
        @test length(fc.mean) == 12
        @test all(isfinite, fc.mean)

        # Forecast with prediction intervals
        fc = forecast(fitted_model, h=12, level=[80, 95])
        @test size(fc.lower) == (12, 2)
        @test size(fc.upper) == (12, 2)
        @test all(isfinite, vec(fc.lower))
        @test all(isfinite, vec(fc.upper))

        # Lower bounds should be less than point forecast, upper should be greater
        @test all(fc.lower[:, 1] .<= fc.mean)
        @test all(fc.mean .<= fc.upper[:, 1])
    end

    @testset "Panel data fitting with TbatsSpec" begin
        # Create panel data with 3 groups
        n_per_group = 104  # 2 years of weekly data per group
        groups = ["A", "B", "C"]

        product = repeat(groups, inner=n_per_group)
        t = repeat(1:n_per_group, length(groups))
        sales = vcat([
            100.0 .+ 10.0 .* sin.(2π .* (1:n_per_group) ./ 52) .+ 2.0 .* randn(n_per_group)
            for _ in groups
        ]...)

        panel_data = (product=product, sales=sales)

        spec = TbatsSpec(@formula(sales = tbats(seasonal_periods=52)))

        # Grouped fit
        fitted_models = fit(spec, panel_data, groupby=:product)
        @test fitted_models isa GroupedFittedModels
        @test length(fitted_models.models) == 3

        # Check each group has a fitted model
        for (key, model) in fitted_models.models
            @test model isa FittedTbats
            @test length(model.fit.fitted_values) == n_per_group
        end
    end

    @testset "Direct formula interface (tbats(formula, data))" begin
        n = 156
        t = 1:n
        y = 100.0 .+ 10.0 .* sin.(2π .* t ./ 52) .+ 2.0 .* randn(n)
        data = (sales = y,)

        formula = @formula(sales = tbats(seasonal_periods=52))

        # Direct call without TbatsSpec
        model = tbats_direct(formula, data)
        @test model isa TBATSModel
        @test length(model.fitted_values) == n
    end

    @testset "Error handling" begin
        data = (sales = randn(100),)

        # Wrong target variable
        spec = TbatsSpec(@formula(wrong_col = tbats()))
        @test_throws ArgumentError fit(spec, data)

        # Non-table input
        y = randn(100)
        formula = @formula(sales = tbats())
        @test_throws ArgumentError tbats_direct(formula, y)
    end

end
