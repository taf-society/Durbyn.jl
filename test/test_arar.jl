using Test
using Durbyn
import Durbyn.Generics: Forecast
using Durbyn.Ararma

@testset "Durbyn.Ararma - ARAR tests" begin
    ap = air_passengers()

    @testset "Basic fit & forecast (equal depths/lags)" begin
        fit = arar(ap; max_ar_depth = 13, max_lag = 13)
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
        fit1 = arar(ap; max_ar_depth = nothing, max_lag = 29)
        @test fit1 isa ARAR
        fc1 = forecast(fit1; h = 12)
        @test length(fc1.mean) == 12

        fit2 = arar(ap; max_ar_depth = 26, max_lag = nothing)
        @test fit2 isa ARAR
        fc2 = forecast(fit2; h = 12)
        @test length(fc2.mean) == 12
    end

    @testset "Invalid parameter ordering (should error)" begin
    
        @test_throws ArgumentError arar(ap; max_ar_depth = 13, max_lag = 12)
    end

    @testset "Short series behavior" begin
        ap1 = ap[1:11] 
        fit_s1 = arar(ap1)
        @test fit_s1 isa ARAR
        fc_s1 = forecast(fit_s1; h = 12)
        @test length(fc_s1.mean) == 12

        ap2 = ap[1:9]
        
        @test_logs (:warn,) begin
            fit_s2 = arar(ap2)
            fc_s2 = forecast(fit_s2; h = 12)
            @test length(fc_s2.mean) == 12
        end
    end
end
