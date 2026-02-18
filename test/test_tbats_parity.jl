using Test
using Durbyn
using Durbyn.Tbats: tbats, TBATSModel
using Durbyn.Bats: BATSModel
import Durbyn.Generics: Forecast, forecast

@testset "TBATS parity: no Box-Cox, no trend, no ARMA" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=false, use_trend=false, use_arma_errors=false)
    @test fit isa TBATSModel

    @test fit.alpha ≈ 1.206920724927324 rtol=1e-6
    @test fit.AIC ≈ 1552.191520930223533 rtol=1e-6
    @test fit.variance ≈ 274.472216315131220 rtol=1e-6
    @test isnothing(fit.beta)
    @test isnothing(fit.damping_parameter)
    @test fit.k_vector == [5]

    @test fit.gamma_one_values[1] ≈ 0.001021925151200 rtol=0.01
    @test fit.gamma_two_values[1] ≈ 0.004214386646937 rtol=0.01

    expected_fc = [443.854406864186046, 432.262225041044360, 468.337429865421996,
                   460.477764925529186, 463.859526365489501, 501.571213244694547,
                   539.055030073148941, 535.611845930701861, 486.519719669145331,
                   447.305315121398451, 412.193426635988999, 436.591175210012352,
                   443.854406864186046, 432.262225041044303, 468.337429865421939,
                   460.477764925529129, 463.859526365489501, 501.571213244694491,
                   539.055030073148828, 535.611845930701975, 486.519719669145388,
                   447.305315121398451, 412.193426635988999, 436.591175210012352]
    fc = forecast(fit; h=24)
    @test fc.mean ≈ expected_fc rtol=1e-6
end

@testset "TBATS parity: no Box-Cox, trend + damped, no ARMA" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=false, use_trend=true, use_damped_trend=true, use_arma_errors=false)
    @test fit isa TBATSModel

    @test fit.alpha ≈ 1.278287183275996 rtol=1e-6
    @test fit.beta ≈ -0.132712913219964 rtol=1e-4
    @test fit.damping_parameter ≈ 0.800496054382298 rtol=1e-4
    @test fit.AIC ≈ 1558.761799728193182 rtol=1e-6
    @test fit.variance ≈ 275.561355461636708 rtol=1e-6
    @test fit.k_vector == [5]

    @test fit.gamma_one_values[1] ≈ -0.001607431257241 rtol=0.01
    @test fit.gamma_two_values[1] ≈ 0.000796782500375 rtol=0.01

    expected_fc = [449.180912033141169, 438.679397416313805, 471.363184621078119,
                   464.628822803483388, 467.637734153927681, 503.310998611945763,
                   541.481107417677777, 537.130541021870840, 486.776543889382594,
                   446.831396205030956, 411.618796617465819, 436.574742548801225,
                   447.556180178116676, 437.378805976937542, 470.322066305493934,
                   463.795411699713270, 466.970591853681299, 502.776953832886818,
                   541.053606679177733, 536.788328367456074, 486.502604009763957,
                   446.612108412258181, 411.443257604576900, 436.434224261593499]
    fc = forecast(fit; h=24)
    @test fc.mean ≈ expected_fc rtol=1e-6
end

@testset "TBATS parity: Box-Cox, trend + damped, no ARMA" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=true, use_trend=true, use_damped_trend=true, use_arma_errors=false)
    @test fit isa TBATSModel

    @test !isnothing(fit.lambda)
    @test abs(fit.lambda) < 0.01
    @test fit.alpha ≈ 0.763619800802097 rtol=0.01
    @test fit.beta ≈ 0.035183419858474 rtol=0.02
    @test fit.damping_parameter ≈ 0.989857669723660 rtol=0.001
    @test fit.AIC ≈ 1400.700441721485959 rtol=0.001
    @test fit.k_vector == [5]

    expected_fc = [447.074104914059035, 437.884140051731833, 504.542810874353734,
                   489.151342161855382, 494.597479584298128, 558.208958308334900,
                   626.630177075764323, 620.565268251174757, 543.550241954130911,
                   472.893328735540194, 414.626334446593205, 464.347505159249238]
    fc = forecast(fit; h=12)
    @test fc.mean ≈ expected_fc rtol=0.01
end
