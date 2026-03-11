using Test
using Durbyn
using Durbyn.Grammar: KwFilterTerm, kw_filter, _extract_single_term
using Durbyn.ModelSpecs: KwFilterSpec, FittedKwFilter

@testset "KolmogorovWiener Grammar Interface" begin

    @testset "KwFilterTerm construction" begin
        @testset "kw_filter() defaults" begin
            term = kw_filter()
            @test term isa KwFilterTerm
            @test term.filter_type == :hp
            @test isnothing(term.lambda)
            @test isnothing(term.low)
            @test isnothing(term.high)
            @test isnothing(term.order)
            @test isnothing(term.omega_c)
            @test isnothing(term.output)
            @test isnothing(term.maxcoef)
        end

        @testset "HP filter with options" begin
            term = kw_filter(filter=:hp, lambda=1600.0, output=:trend)
            @test term.filter_type == :hp
            @test term.lambda == 1600.0
            @test term.output == :trend
        end

        @testset "Bandpass filter" begin
            term = kw_filter(filter=:bandpass, low=6, high=32)
            @test term.filter_type == :bandpass
            @test term.low == 6.0
            @test term.high == 32.0
        end

        @testset "Butterworth filter" begin
            term = kw_filter(filter=:butterworth, order=2, omega_c=0.2)
            @test term.filter_type == :butterworth
            @test term.order == 2
            @test term.omega_c ≈ 0.2
        end

        @testset "Validation errors" begin
            @test_throws ArgumentError kw_filter(filter=:invalid)
            @test_throws ArgumentError kw_filter(lambda=-1.0)
            @test_throws ArgumentError kw_filter(low=-1.0)
            @test_throws ArgumentError kw_filter(filter=:bandpass, low=10, high=5)
            @test_throws ArgumentError kw_filter(filter=:bandpass, low=6)
            @test_throws ArgumentError kw_filter(filter=:butterworth)
            @test_throws ArgumentError kw_filter(output=:invalid)
            @test_throws ArgumentError kw_filter(order=0)
            @test_throws ArgumentError kw_filter(maxcoef=0)
        end
    end

    @testset "@formula with kw_filter" begin
        f = @formula(gdp = kw_filter())
        @test f isa Durbyn.Grammar.ModelFormula
        @test f.target == :gdp

        f2 = @formula(y = kw_filter(filter=:hp, lambda=1600))
        @test f2.target == :y
        term = _extract_single_term(f2, KwFilterTerm)
        @test term.filter_type == :hp
        @test term.lambda == 1600.0

        f3 = @formula(y = kw_filter(filter=:bandpass, low=6, high=32))
        term3 = _extract_single_term(f3, KwFilterTerm)
        @test term3.filter_type == :bandpass
        @test term3.low == 6.0
        @test term3.high == 32.0
    end

    @testset "KwFilterSpec construction" begin
        spec = KwFilterSpec(@formula(y = kw_filter()))
        @test spec isa KwFilterSpec
        @test spec.formula.target == :y
        @test isnothing(spec.m)

        spec_m = KwFilterSpec(@formula(y = kw_filter(filter=:hp, lambda=1600)), m=12)
        @test spec_m.m == 12
    end

    # ── Single-series fit & forecast ──────────────────────────────────────
    y = air_passengers()

    @testset "fit(KwFilterSpec) - single series" begin
        spec = KwFilterSpec(@formula(value = kw_filter()))
        data = (value = y,)
        fitted_model = fit(spec, data, m=12)

        @test fitted_model isa FittedKwFilter
        @test fitted_model.target_col == :value
        @test fitted_model.m == 12
        @test fitted_model.fit isa Durbyn.KWFilterResult
        @test length(Durbyn.fitted(fitted_model.fit)) == length(y)
    end

    @testset "fit(KwFilterSpec) - HP trend" begin
        spec = KwFilterSpec(@formula(value = kw_filter(filter=:hp, lambda=14400, output=:trend)))
        data = (value = y,)
        fitted_model = fit(spec, data, m=12)

        @test fitted_model.fit.output == :trend
        @test fitted_model.fit.filter_type == :hp
    end

    @testset "fit(KwFilterSpec) - bandpass" begin
        spec = KwFilterSpec(@formula(value = kw_filter(filter=:bandpass, low=6, high=32)))
        data = (value = y,)
        fitted_model = fit(spec, data, m=12)

        @test fitted_model.fit.filter_type == :bandpass
    end

    @testset "forecast(FittedKwFilter)" begin
        spec = KwFilterSpec(@formula(value = kw_filter()))
        data = (value = y,)
        fitted_model = fit(spec, data, m=12)

        fc = forecast(fitted_model, h=12)
        @test length(fc.mean) == 12
        @test size(fc.upper, 1) == 12
        @test size(fc.lower, 1) == 12
    end

    @testset "m defaults to 1 when not specified" begin
        spec = KwFilterSpec(@formula(value = kw_filter()))
        data = (value = y,)
        fitted_model = fit(spec, data)
        @test fitted_model.m == 1
    end

    @testset "m from spec" begin
        spec = KwFilterSpec(@formula(value = kw_filter()), m=12)
        data = (value = y,)
        fitted_model = fit(spec, data)
        @test fitted_model.m == 12
    end

    # ── Tables.jl compatibility (NamedTuple = column table) ───────────────
    @testset "Tables.jl - NamedTuple" begin
        data = (value = y, dummy = ones(length(y)))
        spec = KwFilterSpec(@formula(value = kw_filter()))
        fitted_model = fit(spec, data, m=12)
        @test fitted_model isa FittedKwFilter

        fc = forecast(fitted_model, h=6)
        @test length(fc.mean) == 6
    end

    # ── Panel / grouped data ──────────────────────────────────────────────
    @testset "Panel data - groupby" begin
        n = length(y)
        data = (
            value = vcat(y, y .+ 50.0),
            group = vcat(fill(:A, n), fill(:B, n)),
        )
        spec = KwFilterSpec(@formula(value = kw_filter()))
        grouped = fit(spec, data, m=12, groupby=:group)

        @test grouped isa Durbyn.GroupedFittedModels
        sm = Durbyn.successful_models(grouped)
        @test length(sm) == 2
        for (key, model) in sm
            @test model isa FittedKwFilter
        end
    end

    @testset "Panel data - multiple groupby columns" begin
        n = length(y)
        data = (
            value = vcat(y, y .+ 50.0, y .* 1.1, y .* 0.9),
            region = vcat(fill(:East, 2n), fill(:West, 2n)),
            product = vcat(fill(:A, n), fill(:B, n), fill(:A, n), fill(:B, n)),
        )
        spec = KwFilterSpec(@formula(value = kw_filter()))
        grouped = fit(spec, data, m=12, groupby=[:region, :product])

        @test grouped isa Durbyn.GroupedFittedModels
        sm = Durbyn.successful_models(grouped)
        @test length(sm) == 4
    end

    @testset "PanelData wrapper" begin
        n = length(y)
        raw_data = (
            value = vcat(y, y .+ 50.0),
            group = vcat(fill(:A, n), fill(:B, n)),
        )
        panel = PanelData(raw_data, groupby=[:group], m=12)
        spec = KwFilterSpec(@formula(value = kw_filter()))
        grouped = fit(spec, panel)

        @test grouped isa Durbyn.GroupedFittedModels
        sm = Durbyn.successful_models(grouped)
        @test length(sm) == 2
    end

    # ── Error handling ────────────────────────────────────────────────────
    @testset "Error: missing target column" begin
        spec = KwFilterSpec(@formula(missing_col = kw_filter()))
        data = (value = y,)
        @test_throws ArgumentError fit(spec, data, m=12)
    end

    @testset "Error: non-numeric target" begin
        spec = KwFilterSpec(@formula(value = kw_filter()))
        data = (value = fill("abc", 100),)
        @test_throws ArgumentError fit(spec, data, m=12)
    end

    # ── show methods ──────────────────────────────────────────────────────
    @testset "show methods" begin
        spec = KwFilterSpec(@formula(value = kw_filter()))
        data = (value = y,)
        fitted_model = fit(spec, data, m=12)

        buf = IOBuffer()
        show(buf, spec)
        @test occursin("KwFilterSpec", String(take!(buf)))

        show(buf, fitted_model)
        @test occursin("FittedKwFilter", String(take!(buf)))

        show(buf, MIME("text/plain"), fitted_model)
        s = String(take!(buf))
        @test occursin("FittedKwFilter", s)
        @test occursin("Filter:", s)
        @test occursin("MSE:", s)
    end
end
