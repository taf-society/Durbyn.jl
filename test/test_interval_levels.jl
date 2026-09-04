using Test
using Durbyn

import Durbyn.Utils: _normalize_levels
import Durbyn.Generics: forecast

const AP = Float64.(air_passengers())

@testset "_normalize_levels" begin
    @test _normalize_levels([80, 95]) == [80.0, 95.0]
    @test _normalize_levels([0.8, 0.95]) == [80.0, 95.0]
    @test _normalize_levels([0.5]) == [50.0]
    @test _normalize_levels(Float64[]) == Float64[]
    @test _normalize_levels([80]; fan = true) == collect(51.0:3.0:99.0)

    # Ordering is preserved; callers that need a particular column order sort it.
    @test _normalize_levels([95, 80]) == [95.0, 80.0]

    # Mixing the two conventions used to be read silently as 0.8% and 95%.
    @test_throws ArgumentError _normalize_levels([0.8, 95])
    @test_throws ArgumentError _normalize_levels([95, 0.8])

    # Out-of-range levels.
    @test_throws ArgumentError _normalize_levels([0, 95])
    @test_throws ArgumentError _normalize_levels([-5, 95])
    @test_throws ArgumentError _normalize_levels([80, 100])
    @test_throws ArgumentError _normalize_levels([150])
end

@testset "level conventions agree across models" begin
    fits = Any[snaive(AP, 12), naive(AP), meanf(AP, 12), arar(AP), ararma(AP, p = 1, q = 1)]
    for f in fits
        a = forecast(f, h = 6, level = [80, 95])
        b = forecast(f, h = 6, level = [0.8, 0.95])
        @test a.upper ≈ b.upper
        @test a.lower ≈ b.lower
        @test all(a.upper[:, 2] .>= a.upper[:, 1])   # 95% band is the wider one
        @test_throws ArgumentError forecast(f, h = 6, level = [0.8, 95])
        @test_throws ArgumentError forecast(f, h = 6, level = [150])
    end
end

@testset "ARAR/ARARMA reject a non-positive horizon" begin
    @test_throws ArgumentError forecast(arar(AP), h = 0)
    @test_throws ArgumentError forecast(ararma(AP, p = 1, q = 1), h = 0)
end

@testset "Theta forwards n_samples and seed" begin
    t = theta(AP, 12)
    f1 = forecast(t, h = 6, level = [95], n_samples = 500, seed = 1)
    f2 = forecast(t, h = 6, level = [95], n_samples = 500, seed = 1)
    f3 = forecast(t, h = 6, level = [95], n_samples = 500, seed = 2)
    f4 = forecast(t, h = 6, level = [95], n_samples = 50, seed = 1)

    @test f1.upper == f2.upper          # same seed reproduces
    @test f1.upper != f3.upper          # seed is honoured
    @test f1.upper != f4.upper          # n_samples is honoured
    @test_throws ArgumentError forecast(t, h = 6, n_samples = 1)

    # and through the spec layer, which forwards kwargs to the model method
    data = (value = AP, idx = collect(1:length(AP)))
    ft = fit(ThetaSpec(@formula(value = theta()), m = 12), data)
    g1 = forecast(ft, h = 6, level = [95], n_samples = 500, seed = 1)
    g2 = forecast(ft, h = 6, level = [95], n_samples = 500, seed = 1)
    g3 = forecast(ft, h = 6, level = [95], n_samples = 500, seed = 7)
    @test g1.upper == g2.upper
    @test g1.upper != g3.upper
end
