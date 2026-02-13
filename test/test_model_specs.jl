using Test
using Durbyn
using Durbyn.ModelSpecs: DiffusionSpec, FittedDiffusion, ThetaSpec, FittedTheta,
    MeanfSpec, FittedMeanf, ArimaSpec, FittedArima,
    EtsSpec, FittedEts, SesSpec, FittedSes, HoltSpec, FittedHolt,
    HoltWintersSpec, FittedHoltWinters, CrostonSpec, FittedCroston,
    NaiveSpec, FittedNaive, SnaiveSpec, FittedSnaive, RwSpec, FittedRw,
    ArarSpec, FittedArar, ArarmaSpec, FittedArarma, BatsSpec, FittedBats,
    TbatsSpec, FittedTbats,
    ModelCollection, FittedModelCollection, ForecastModelCollection,
    GroupedFittedModels, GroupedForecasts,
    fit, forecast, extract_metrics, as_table, model,
    successful_models, failed_groups, errors
using Durbyn.Generics: Forecast

# ─── Test Data ──────────────────────────────────────────────────────

# Bass-shaped adoption data for diffusion tests
const ADOPTION_DATA_MS = Float64[5, 10, 25, 45, 70, 85, 75, 50, 30, 15]

# AirPassengers subset (24 values, 2 full years)
const AP_SHORT_MS = Float64[
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140
]

# AirPassengers extended subset (48 values, 4 full years — needed for HW/BATS/TBATS/ETS seasonal)
const AP_LONG_MS = Float64[
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194
]

# Trending non-seasonal data (40 values — for Naive/RW/ARAR/ARARMA)
const NONSEASONAL_MS = Float64[
    10.0, 12.1, 11.5, 13.2, 14.0, 15.3, 14.8, 16.1, 17.5, 18.0,
    19.2, 18.7, 20.3, 21.1, 22.5, 23.0, 22.8, 24.1, 25.5, 26.0,
    27.2, 26.8, 28.3, 29.1, 30.5, 31.0, 30.8, 32.1, 33.5, 34.0,
    35.2, 34.7, 36.3, 37.1, 38.5, 39.0, 38.8, 40.1, 41.5, 42.0
]

# Sparse intermittent demand (30 values — for Croston)
const INTERMITTENT_MS = Float64[
    0, 0, 3, 0, 0, 0, 2, 0, 0, 5,
    0, 0, 0, 0, 1, 0, 0, 4, 0, 0,
    0, 3, 0, 0, 0, 0, 2, 0, 0, 1
]

function make_grouped_data_ms()
    n = 24
    y = vcat(AP_SHORT_MS, AP_SHORT_MS .* 1.1)
    group = vcat(fill("A", n), fill("B", n))
    return (y = y, group = group)
end

function make_grouped_long_data_ms()
    n = 48
    y = vcat(AP_LONG_MS, AP_LONG_MS .* 1.1)
    group = vcat(fill("A", n), fill("B", n))
    return (y = y, group = group)
end

function make_grouped_nonseasonal_ms()
    n = 40
    y = vcat(NONSEASONAL_MS, NONSEASONAL_MS .+ 5.0)
    group = vcat(fill("A", n), fill("B", n))
    return (y = y, group = group)
end

function make_grouped_intermittent_ms()
    n = 30
    y = vcat(INTERMITTENT_MS, INTERMITTENT_MS .* 2.0)
    group = vcat(fill("A", n), fill("B", n))
    return (y = y, group = group)
end

# ═════════════════════════════════════════════════════════════════════
# NEW TESTS: Individual Spec Types
# ═════════════════════════════════════════════════════════════════════

# -----------------------------------------------------------------
# EtsSpec
# -----------------------------------------------------------------
@testset "EtsSpec" begin
    @testset "Basic fit (non-seasonal)" begin
        spec = EtsSpec(@formula(y = e("A") + t("N") + s("N")))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)

        @test fitted_model isa FittedEts
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = EtsSpec(@formula(y = e("A") + t("N") + s("N")))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)
        fc = forecast(fitted_model, h=6)

        @test length(fc.mean) == 6
        @test all(isfinite, fc.mean)
        @test length(fc.level) == 2  # default [80, 95]
    end

    @testset "Extract metrics" begin
        spec = EtsSpec(@formula(y = e("A") + t("N") + s("N")))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)
        metrics = extract_metrics(fitted_model)

        for k in [:aic, :aicc, :bic, :mse, :sigma2]
            @test haskey(metrics, k)
            @test isfinite(metrics[k])
        end
    end

    @testset "Seasonal ETS" begin
        spec = EtsSpec(@formula(y = e("A") + t("A") + s("A")))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data, m=12)

        @test fitted_model isa FittedEts
        fc = forecast(fitted_model, h=12)
        @test length(fc.mean) == 12
        @test all(isfinite, fc.mean)
    end

    @testset "Damped trend via drift()" begin
        spec = EtsSpec(@formula(y = e("A") + t("A") + s("N") + drift()))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)

        @test fitted_model isa FittedEts
    end

    @testset "Grouped fit" begin
        gdata = make_grouped_long_data_ms()
        spec = EtsSpec(@formula(y = e("A") + t("N") + s("N")))
        fitted_model = fit(spec, gdata, m=1, groupby=:group)

        @test fitted_model isa GroupedFittedModels
        @test fitted_model.successful >= 1
    end

    @testset "Target validation" begin
        spec = EtsSpec(@formula(nonexistent = e("A") + t("N") + s("N")))
        data = (y = AP_SHORT_MS,)
        @test_throws ArgumentError fit(spec, data, m=1)
    end
end

# -----------------------------------------------------------------
# SesSpec
# -----------------------------------------------------------------
@testset "SesSpec" begin
    @testset "Basic fit" begin
        spec = SesSpec(@formula(y = ses()))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)

        @test fitted_model isa FittedSes
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = SesSpec(@formula(y = ses()))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)
        fc = forecast(fitted_model, h=6)

        @test length(fc.mean) == 6
        @test all(isfinite, fc.mean)
        @test length(fc.level) == 2  # default [80, 95]
    end

    @testset "Extract metrics" begin
        spec = SesSpec(@formula(y = ses()))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)
        metrics = extract_metrics(fitted_model)

        for k in [:aic, :aicc, :bic, :mse, :sigma2]
            @test haskey(metrics, k)
            @test isfinite(metrics[k])
        end
    end

    @testset "Grouped fit" begin
        gdata = make_grouped_data_ms()
        spec = SesSpec(@formula(y = ses()))
        fitted_model = fit(spec, gdata, m=1, groupby=:group)

        @test fitted_model isa GroupedFittedModels
        @test fitted_model.successful >= 1
    end

    @testset "Target validation" begin
        spec = SesSpec(@formula(nonexistent = ses()))
        data = (y = AP_SHORT_MS,)
        @test_throws ArgumentError fit(spec, data, m=1)
    end
end

# -----------------------------------------------------------------
# HoltSpec
# -----------------------------------------------------------------
@testset "HoltSpec" begin
    @testset "Basic fit" begin
        spec = HoltSpec(@formula(y = holt()))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)

        @test fitted_model isa FittedHolt
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = HoltSpec(@formula(y = holt()))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)
        fc = forecast(fitted_model, h=6)

        @test length(fc.mean) == 6
        @test all(isfinite, fc.mean)
        @test length(fc.level) == 2  # default [80, 95]
    end

    @testset "Extract metrics" begin
        spec = HoltSpec(@formula(y = holt()))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)
        metrics = extract_metrics(fitted_model)

        for k in [:aic, :aicc, :bic, :mse, :sigma2]
            @test haskey(metrics, k)
            @test isfinite(metrics[k])
        end
    end

    @testset "Damped Holt" begin
        spec = HoltSpec(@formula(y = holt(damped=true)))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)

        @test fitted_model isa FittedHolt
    end

    @testset "Exponential Holt" begin
        spec = HoltSpec(@formula(y = holt(exponential=true)))
        data = (y = AP_SHORT_MS,)
        fitted_model = fit(spec, data, m=1)

        @test fitted_model isa FittedHolt
    end

    @testset "Grouped fit" begin
        gdata = make_grouped_data_ms()
        spec = HoltSpec(@formula(y = holt()))
        fitted_model = fit(spec, gdata, m=1, groupby=:group)

        @test fitted_model isa GroupedFittedModels
        @test fitted_model.successful >= 1
    end

    @testset "Target validation" begin
        spec = HoltSpec(@formula(nonexistent = holt()))
        data = (y = AP_SHORT_MS,)
        @test_throws ArgumentError fit(spec, data, m=1)
    end
end

# -----------------------------------------------------------------
# HoltWintersSpec
# -----------------------------------------------------------------
@testset "HoltWintersSpec" begin
    @testset "Basic fit (additive)" begin
        spec = HoltWintersSpec(@formula(y = hw(seasonal="additive")))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data, m=12)

        @test fitted_model isa FittedHoltWinters
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = HoltWintersSpec(@formula(y = hw(seasonal="additive")))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data, m=12)
        fc = forecast(fitted_model, h=12)

        @test length(fc.mean) == 12
        @test all(isfinite, fc.mean)
        @test length(fc.level) == 2  # default [80, 95]
    end

    @testset "Extract metrics" begin
        spec = HoltWintersSpec(@formula(y = hw(seasonal="additive")))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data, m=12)
        metrics = extract_metrics(fitted_model)

        for k in [:aic, :aicc, :bic, :mse, :sigma2]
            @test haskey(metrics, k)
            @test isfinite(metrics[k])
        end
    end

    @testset "Multiplicative seasonal" begin
        spec = HoltWintersSpec(@formula(y = hw(seasonal="multiplicative")))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data, m=12)

        @test fitted_model isa FittedHoltWinters
    end

    @testset "Damped HoltWinters" begin
        spec = HoltWintersSpec(@formula(y = hw(seasonal="additive", damped=true)))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data, m=12)

        @test fitted_model isa FittedHoltWinters
    end

    @testset "Grouped fit" begin
        gdata = make_grouped_long_data_ms()
        spec = HoltWintersSpec(@formula(y = hw(seasonal="additive")))
        fitted_model = fit(spec, gdata, m=12, groupby=:group)

        @test fitted_model isa GroupedFittedModels
        @test fitted_model.successful >= 1
    end

    @testset "Target validation" begin
        spec = HoltWintersSpec(@formula(nonexistent = hw(seasonal="additive")))
        data = (y = AP_LONG_MS,)
        @test_throws ArgumentError fit(spec, data, m=12)
    end
end

# -----------------------------------------------------------------
# CrostonSpec
# -----------------------------------------------------------------
@testset "CrostonSpec" begin
    @testset "Basic fit (hyndman)" begin
        spec = CrostonSpec(@formula(y = croston()))
        data = (y = INTERMITTENT_MS,)
        fitted_model = fit(spec, data, m=1)

        @test fitted_model isa FittedCroston
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = CrostonSpec(@formula(y = croston()))
        data = (y = INTERMITTENT_MS,)
        fitted_model = fit(spec, data, m=1)
        fc = forecast(fitted_model, h=6)

        @test length(fc.mean) == 6
        @test all(isfinite, fc.mean)
    end

    @testset "Extract metrics (empty)" begin
        spec = CrostonSpec(@formula(y = croston()))
        data = (y = INTERMITTENT_MS,)
        fitted_model = fit(spec, data, m=1)
        metrics = extract_metrics(fitted_model)

        @test metrics isa Dict
    end

    @testset "Method classic" begin
        spec = CrostonSpec(@formula(y = croston(method="classic")))
        data = (y = INTERMITTENT_MS,)
        fitted_model = fit(spec, data, m=1)
        @test fitted_model isa FittedCroston
    end

    @testset "Method sba" begin
        spec = CrostonSpec(@formula(y = croston(method="sba")))
        data = (y = INTERMITTENT_MS,)
        fitted_model = fit(spec, data, m=1)
        @test fitted_model isa FittedCroston
    end

    @testset "Method sbj" begin
        spec = CrostonSpec(@formula(y = croston(method="sbj")))
        data = (y = INTERMITTENT_MS,)
        fitted_model = fit(spec, data, m=1)
        @test fitted_model isa FittedCroston
    end

    @testset "Grouped fit" begin
        gdata = make_grouped_intermittent_ms()
        spec = CrostonSpec(@formula(y = croston()))
        fitted_model = fit(spec, gdata, m=1, groupby=:group)

        @test fitted_model isa GroupedFittedModels
        @test fitted_model.successful >= 1
    end

    @testset "Target validation" begin
        spec = CrostonSpec(@formula(nonexistent = croston()))
        data = (y = INTERMITTENT_MS,)
        @test_throws ArgumentError fit(spec, data, m=1)
    end
end

# -----------------------------------------------------------------
# NaiveSpec
# -----------------------------------------------------------------
@testset "NaiveSpec" begin
    @testset "Basic fit" begin
        spec = NaiveSpec(@formula(y = naive_term()))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data, m=1)

        @test fitted_model isa FittedNaive
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = NaiveSpec(@formula(y = naive_term()))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data, m=1)
        fc = forecast(fitted_model, h=6)

        @test length(fc.mean) == 6
        @test all(isfinite, fc.mean)
        @test length(fc.lower) == 2
        @test length(fc.upper) == 2
    end

    @testset "Extract metrics (empty)" begin
        spec = NaiveSpec(@formula(y = naive_term()))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data, m=1)
        metrics = extract_metrics(fitted_model)

        @test metrics isa Dict
    end

    @testset "Grouped fit" begin
        gdata = make_grouped_nonseasonal_ms()
        spec = NaiveSpec(@formula(y = naive_term()))
        fitted_model = fit(spec, gdata, m=1, groupby=:group)

        @test fitted_model isa GroupedFittedModels
        @test fitted_model.successful >= 1
    end

    @testset "Target validation" begin
        spec = NaiveSpec(@formula(nonexistent = naive_term()))
        data = (y = NONSEASONAL_MS,)
        @test_throws ArgumentError fit(spec, data, m=1)
    end
end

# -----------------------------------------------------------------
# SnaiveSpec
# -----------------------------------------------------------------
@testset "SnaiveSpec" begin
    @testset "Basic fit" begin
        spec = SnaiveSpec(@formula(y = snaive_term()))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data, m=12)

        @test fitted_model isa FittedSnaive
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = SnaiveSpec(@formula(y = snaive_term()))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data, m=12)
        fc = forecast(fitted_model, h=12)

        @test length(fc.mean) == 12
        @test all(isfinite, fc.mean)
        @test length(fc.lower) == 2
        @test length(fc.upper) == 2
    end

    @testset "Extract metrics (empty)" begin
        spec = SnaiveSpec(@formula(y = snaive_term()))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data, m=12)
        metrics = extract_metrics(fitted_model)

        @test metrics isa Dict
    end

    @testset "Grouped fit" begin
        gdata = make_grouped_long_data_ms()
        spec = SnaiveSpec(@formula(y = snaive_term()))
        fitted_model = fit(spec, gdata, m=12, groupby=:group)

        @test fitted_model isa GroupedFittedModels
        @test fitted_model.successful >= 1
    end

    @testset "Target validation" begin
        spec = SnaiveSpec(@formula(nonexistent = snaive_term()))
        data = (y = AP_LONG_MS,)
        @test_throws ArgumentError fit(spec, data, m=12)
    end
end

# -----------------------------------------------------------------
# RwSpec
# -----------------------------------------------------------------
@testset "RwSpec" begin
    @testset "Basic fit" begin
        spec = RwSpec(@formula(y = rw_term()))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data, m=1)

        @test fitted_model isa FittedRw
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = RwSpec(@formula(y = rw_term()))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data, m=1)
        fc = forecast(fitted_model, h=6)

        @test length(fc.mean) == 6
        @test all(isfinite, fc.mean)
        @test length(fc.lower) == 2
        @test length(fc.upper) == 2
    end

    @testset "Extract metrics (empty)" begin
        spec = RwSpec(@formula(y = rw_term()))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data, m=1)
        metrics = extract_metrics(fitted_model)

        @test metrics isa Dict
    end

    @testset "Random walk with drift" begin
        spec = RwSpec(@formula(y = rw_term(drift=true)))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data, m=1)

        @test fitted_model isa FittedRw
        fc = forecast(fitted_model, h=6)
        @test length(fc.mean) == 6
        @test all(isfinite, fc.mean)
    end

    @testset "Grouped fit" begin
        gdata = make_grouped_nonseasonal_ms()
        spec = RwSpec(@formula(y = rw_term()))
        fitted_model = fit(spec, gdata, m=1, groupby=:group)

        @test fitted_model isa GroupedFittedModels
        @test fitted_model.successful >= 1
    end

    @testset "Target validation" begin
        spec = RwSpec(@formula(nonexistent = rw_term()))
        data = (y = NONSEASONAL_MS,)
        @test_throws ArgumentError fit(spec, data, m=1)
    end
end

# -----------------------------------------------------------------
# ArarSpec
# -----------------------------------------------------------------
@testset "ArarSpec" begin
    @testset "Basic fit" begin
        spec = ArarSpec(@formula(y = arar()))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data)

        @test fitted_model isa FittedArar
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = ArarSpec(@formula(y = arar()))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data)
        fc = forecast(fitted_model, h=6)

        @test length(fc.mean) == 6
        @test all(isfinite, fc.mean)
    end

    @testset "Extract metrics" begin
        spec = ArarSpec(@formula(y = arar()))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data)
        metrics = extract_metrics(fitted_model)

        @test haskey(metrics, :sigma2)
        @test isfinite(metrics[:sigma2])
    end

    @testset "Grouped fit" begin
        gdata = make_grouped_nonseasonal_ms()
        spec = ArarSpec(@formula(y = arar()))
        fitted_model = fit(spec, gdata, groupby=:group)

        @test fitted_model isa GroupedFittedModels
        @test fitted_model.successful >= 1
    end

    @testset "Target validation" begin
        spec = ArarSpec(@formula(nonexistent = arar()))
        data = (y = NONSEASONAL_MS,)
        @test_throws ArgumentError fit(spec, data)
    end
end

# -----------------------------------------------------------------
# ArarmaSpec
# -----------------------------------------------------------------
@testset "ArarmaSpec" begin
    @testset "Basic fit (fixed orders)" begin
        spec = ArarmaSpec(@formula(y = p(1) + q(1)))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data)

        @test fitted_model isa FittedArarma
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = ArarmaSpec(@formula(y = p(1) + q(1)))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data)
        fc = forecast(fitted_model, h=6)

        @test length(fc.mean) == 6
        @test all(isfinite, fc.mean)
    end

    @testset "Extract metrics" begin
        spec = ArarmaSpec(@formula(y = p(1) + q(1)))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data)
        metrics = extract_metrics(fitted_model)

        for k in [:aic, :bic, :sigma2, :loglik]
            @test haskey(metrics, k)
            @test isfinite(metrics[k])
        end
    end

    @testset "Grouped fit" begin
        gdata = make_grouped_nonseasonal_ms()
        spec = ArarmaSpec(@formula(y = p(1) + q(1)))
        fitted_model = fit(spec, gdata, groupby=:group)

        @test fitted_model isa GroupedFittedModels
        @test fitted_model.successful >= 1
    end

    @testset "Target validation" begin
        spec = ArarmaSpec(@formula(nonexistent = p(1) + q(1)))
        data = (y = NONSEASONAL_MS,)
        @test_throws ArgumentError fit(spec, data)
    end
end

# -----------------------------------------------------------------
# BatsSpec (basic only — slow)
# -----------------------------------------------------------------
@testset "BatsSpec" begin
    @testset "Basic fit" begin
        spec = BatsSpec(@formula(y = bats(seasonal_periods=12)))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data)

        @test fitted_model isa FittedBats
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = BatsSpec(@formula(y = bats(seasonal_periods=12)))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data)
        fc = forecast(fitted_model, h=12)

        @test length(fc.mean) == 12
        @test all(isfinite, fc.mean)
    end

    @testset "Extract metrics" begin
        spec = BatsSpec(@formula(y = bats(seasonal_periods=12)))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data)
        metrics = extract_metrics(fitted_model)

        for k in [:aic, :loglik, :sigma2]
            @test haskey(metrics, k)
            @test isfinite(metrics[k])
        end
    end

    @testset "Target validation" begin
        spec = BatsSpec(@formula(nonexistent = bats(seasonal_periods=12)))
        data = (y = AP_LONG_MS,)
        @test_throws ArgumentError fit(spec, data)
    end
end

# -----------------------------------------------------------------
# TbatsSpec (basic only — slow)
# -----------------------------------------------------------------
@testset "TbatsSpec" begin
    @testset "Basic fit" begin
        spec = TbatsSpec(@formula(y = tbats(seasonal_periods=12)))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data)

        @test fitted_model isa FittedTbats
        @test fitted_model.target_col == :y
    end

    @testset "Forecast" begin
        spec = TbatsSpec(@formula(y = tbats(seasonal_periods=12)))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data)
        fc = forecast(fitted_model, h=12)

        @test length(fc.mean) == 12
        @test all(isfinite, fc.mean)
    end

    @testset "Extract metrics" begin
        spec = TbatsSpec(@formula(y = tbats(seasonal_periods=12)))
        data = (y = AP_LONG_MS,)
        fitted_model = fit(spec, data)
        metrics = extract_metrics(fitted_model)

        for k in [:aic, :loglik, :sigma2]
            @test haskey(metrics, k)
            @test isfinite(metrics[k])
        end
    end

    @testset "Target validation" begin
        spec = TbatsSpec(@formula(nonexistent = tbats(seasonal_periods=12)))
        data = (y = AP_LONG_MS,)
        @test_throws ArgumentError fit(spec, data)
    end
end

# ═════════════════════════════════════════════════════════════════════
# NEW TESTS: model() function
# ═════════════════════════════════════════════════════════════════════

@testset "model() function" begin
    @testset "Single spec passthrough" begin
        spec = NaiveSpec(@formula(y = naive_term()))
        result = model(spec)
        @test result === spec
        @test result isa NaiveSpec
    end

    @testset "Multiple specs → ModelCollection" begin
        s1 = NaiveSpec(@formula(y = naive_term()))
        s2 = MeanfSpec(@formula(y = meanf_term()))
        result = model(s1, s2)

        @test result isa ModelCollection
        @test length(result) == 2
    end

    @testset "Custom names" begin
        s1 = NaiveSpec(@formula(y = naive_term()))
        s2 = MeanfSpec(@formula(y = meanf_term()))
        result = model(s1, s2, names=["naive", "meanf"])

        @test result isa ModelCollection
        @test result.names == ["naive", "meanf"]
    end

    @testset "Name count mismatch error" begin
        s1 = NaiveSpec(@formula(y = naive_term()))
        s2 = MeanfSpec(@formula(y = meanf_term()))
        @test_throws Exception model(s1, s2, names=["one"])
    end

    @testset "Single spec + names warns" begin
        spec = NaiveSpec(@formula(y = naive_term()))
        result = @test_logs (:warn, r"Single spec provided") model(spec, names=["a", "b"])
        @test result === spec
    end
end

# ═════════════════════════════════════════════════════════════════════
# NEW TESTS: ModelCollection
# ═════════════════════════════════════════════════════════════════════

@testset "ModelCollection" begin
    s1 = NaiveSpec(@formula(y = naive_term()))
    s2 = MeanfSpec(@formula(y = meanf_term()))
    coll = model(s1, s2, names=["naive", "meanf"])

    @testset "length" begin
        @test length(coll) == 2
    end

    @testset "getindex" begin
        @test coll[1] isa NaiveSpec
        @test coll[2] isa MeanfSpec
    end

    @testset "iterate" begin
        specs = collect(coll)
        @test length(specs) == 2
        @test specs[1] isa NaiveSpec
        @test specs[2] isa MeanfSpec
    end

    @testset "show" begin
        buf = IOBuffer()
        show(buf, coll)
        s = String(take!(buf))
        @test occursin("ModelCollection", s)
        @test occursin("2", s)
    end
end

# ═════════════════════════════════════════════════════════════════════
# NEW TESTS: FittedModelCollection
# ═════════════════════════════════════════════════════════════════════

@testset "FittedModelCollection" begin
    s1 = NaiveSpec(@formula(y = naive_term()))
    s2 = MeanfSpec(@formula(y = meanf_term()))
    coll = model(s1, s2, names=["naive", "meanf"])
    data = (y = NONSEASONAL_MS,)
    fitted_coll = fit(coll, data, m=1)

    @testset "Type and length" begin
        @test fitted_coll isa FittedModelCollection
        @test length(fitted_coll) == 2
    end

    @testset "getindex" begin
        @test fitted_coll[1] isa FittedNaive
        @test fitted_coll[2] isa FittedMeanf
    end

    @testset "Metrics dict" begin
        @test haskey(fitted_coll.metrics, "naive")
        @test haskey(fitted_coll.metrics, "meanf")
    end

    @testset "iterate" begin
        items = collect(fitted_coll)
        @test length(items) == 2
        name1, model1 = items[1]
        @test name1 == "naive"
        @test model1 isa FittedNaive
    end

    @testset "show" begin
        buf = IOBuffer()
        show(buf, fitted_coll)
        s = String(take!(buf))
        @test occursin("FittedModelCollection", s)
        @test occursin("2", s)
    end
end

# ═════════════════════════════════════════════════════════════════════
# NEW TESTS: ForecastModelCollection
# ═════════════════════════════════════════════════════════════════════

@testset "ForecastModelCollection" begin
    s1 = NaiveSpec(@formula(y = naive_term()))
    s2 = MeanfSpec(@formula(y = meanf_term()))
    coll = model(s1, s2, names=["naive", "meanf"])
    data = (y = NONSEASONAL_MS,)
    fitted_coll = fit(coll, data, m=1)
    fc_coll = forecast(fitted_coll, h=6)

    @testset "Type and length" begin
        @test fc_coll isa ForecastModelCollection
        @test length(fc_coll) == 2
    end

    @testset "getindex by index" begin
        fc1 = fc_coll[1]
        @test fc1 isa Forecast
        @test length(fc1.mean) == 6
    end

    @testset "getindex by name" begin
        fc = fc_coll["meanf"]
        @test fc isa Forecast
        @test length(fc.mean) == 6
    end

    @testset "keys" begin
        @test keys(fc_coll) == ["naive", "meanf"]
    end

    @testset "iterate" begin
        items = collect(fc_coll)
        @test length(items) == 2
        name1, fc1 = items[1]
        @test name1 == "naive"
        @test fc1 isa Forecast
    end

    @testset "show" begin
        buf = IOBuffer()
        show(buf, fc_coll)
        s = String(take!(buf))
        @test occursin("ForecastModelCollection", s)
    end

    @testset "show text/plain" begin
        buf = IOBuffer()
        show(buf, MIME("text/plain"), fc_coll)
        s = String(take!(buf))
        @test occursin("ForecastModelCollection", s)
        @test occursin("naive", s)
        @test occursin("meanf", s)
    end
end

# ═════════════════════════════════════════════════════════════════════
# NEW TESTS: GroupedFittedModels helpers
# ═════════════════════════════════════════════════════════════════════

@testset "GroupedFittedModels helpers" begin
    gdata = make_grouped_nonseasonal_ms()
    spec = NaiveSpec(@formula(y = naive_term()))
    gfitted = fit(spec, gdata, m=1, groupby=:group)

    @testset "Type" begin
        @test gfitted isa GroupedFittedModels
    end

    @testset "successful_models" begin
        sm = successful_models(gfitted)
        @test sm isa Dict
        @test length(sm) == gfitted.successful
    end

    @testset "failed_groups" begin
        fg = failed_groups(gfitted)
        @test fg isa Vector
        @test length(fg) == gfitted.failed
    end

    @testset "errors" begin
        errs = errors(gfitted)
        @test errs isa Dict
        @test length(errs) == gfitted.failed
    end

    @testset "length" begin
        @test length(gfitted) == 2
    end

    @testset "iterate" begin
        items = collect(gfitted)
        @test length(items) == 2
        key, mdl = items[1]
        @test key isa NamedTuple
        @test mdl isa FittedNaive
    end

    @testset "getindex" begin
        key = gfitted.groups[1]
        mdl = gfitted[key]
        @test mdl isa FittedNaive
    end

    @testset "show" begin
        buf = IOBuffer()
        show(buf, gfitted)
        s = String(take!(buf))
        @test occursin("GroupedFittedModels", s)
    end
end

# ═════════════════════════════════════════════════════════════════════
# NEW TESTS: GroupedForecasts helpers
# ═════════════════════════════════════════════════════════════════════

@testset "GroupedForecasts helpers" begin
    gdata = make_grouped_nonseasonal_ms()
    spec = NaiveSpec(@formula(y = naive_term()))
    gfitted = fit(spec, gdata, m=1, groupby=:group)
    gfc = forecast(gfitted, h=6)

    @testset "Type" begin
        @test gfc isa GroupedForecasts
    end

    @testset "successful_forecasts" begin
        sf = Durbyn.ModelSpecs.successful_forecasts(gfc)
        @test sf isa Dict
        @test length(sf) == gfc.successful
    end

    @testset "failed_groups" begin
        fg = failed_groups(gfc)
        @test fg isa Vector
        @test length(fg) == gfc.failed
    end

    @testset "errors" begin
        errs = errors(gfc)
        @test errs isa Dict
        @test length(errs) == gfc.failed
    end

    @testset "length" begin
        @test length(gfc) == 2
    end

    @testset "iterate" begin
        items = collect(gfc)
        @test length(items) == 2
        key, fc = items[1]
        @test key isa NamedTuple
        @test fc isa Forecast
    end

    @testset "getindex" begin
        key = gfc.groups[1]
        fc = gfc[key]
        @test fc isa Forecast
        @test length(fc.mean) == 6
    end

    @testset "show" begin
        buf = IOBuffer()
        show(buf, gfc)
        s = String(take!(buf))
        @test occursin("GroupedForecasts", s)
    end
end

# ═════════════════════════════════════════════════════════════════════
# NEW TESTS: as_table
# ═════════════════════════════════════════════════════════════════════

@testset "as_table" begin
    @testset "Forecast → NamedTuple" begin
        spec = NaiveSpec(@formula(y = naive_term()))
        data = (y = NONSEASONAL_MS,)
        fitted_model = fit(spec, data, m=1)
        fc = forecast(fitted_model, h=6, level=[80, 95])
        tbl = as_table(fc)

        @test tbl isa NamedTuple
        @test haskey(tbl, :step)
        @test haskey(tbl, :mean)
        @test haskey(tbl, :model)
        @test haskey(tbl, :lower_80)
        @test haskey(tbl, :upper_80)
        @test haskey(tbl, :lower_95)
        @test haskey(tbl, :upper_95)
        @test length(tbl.step) == 6
        @test tbl.step == [1, 2, 3, 4, 5, 6]
    end

    @testset "GroupedForecasts → stacked with group col" begin
        gdata = make_grouped_nonseasonal_ms()
        spec = NaiveSpec(@formula(y = naive_term()))
        gfitted = fit(spec, gdata, m=1, groupby=:group)
        gfc = forecast(gfitted, h=6, level=[80, 95])
        tbl = as_table(gfc)

        @test tbl isa NamedTuple
        @test haskey(tbl, :group)
        @test haskey(tbl, :step)
        @test haskey(tbl, :mean)
        # 2 groups × 6 steps = 12 rows
        @test length(tbl.step) == 12
    end

    @testset "ForecastModelCollection → stacked with model_name col" begin
        s1 = NaiveSpec(@formula(y = naive_term()))
        s2 = MeanfSpec(@formula(y = meanf_term()))
        coll = model(s1, s2, names=["naive", "meanf"])
        data = (y = NONSEASONAL_MS,)
        fitted_coll = fit(coll, data, m=1)
        fc_coll = forecast(fitted_coll, h=6)
        tbl = as_table(fc_coll)

        @test tbl isa NamedTuple
        @test haskey(tbl, :model_name)
        @test haskey(tbl, :step)
        @test haskey(tbl, :mean)
        # 2 models × 6 steps = 12 rows
        @test length(tbl.step) == 12
    end
end

# ═════════════════════════════════════════════════════════════════════
# PRESERVED EXISTING TESTS (verbatim)
# ═════════════════════════════════════════════════════════════════════

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
