using Test
using Durbyn
using Durbyn.Tbats: tbats, TBATSModel
using Durbyn.Bats: BATSModel
import Durbyn.Generics: Forecast, forecast

@testset "TBATS parity: no Box-Cox, no trend, no ARMA" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=false, use_trend=false, use_arma_errors=false)
    @test fit isa TBATSModel

    @test fit.alpha ≈ 1.2072355130090069 rtol=1e-6
    @test fit.aic ≈ 1552.1915069392826 rtol=1e-6
    @test fit.variance ≈ 274.47218964760094 rtol=1e-6
    @test isnothing(fit.beta)
    @test isnothing(fit.damping_parameter)
    @test fit.k_vector == [5]

    @test fit.gamma_one_values[1] ≈ 0.0010186093074610859 rtol=0.01
    @test fit.gamma_two_values[1] ≈ 0.004218128175729754 rtol=0.01

    expected_fc = [443.8672655303537, 432.27273644427993, 468.3494915696984,
                   460.49136854602733, 463.8685039626076, 501.5849524361126,
                   539.0661672532093, 535.6226508554283, 486.5318760779763,
                   447.3174783437759, 412.2069976750582, 436.6011154432794,
                   443.86726553035373, 432.2727364442799, 468.34949156969833,
                   460.4913685460273, 463.86850396260763, 501.5849524361125,
                   539.0661672532093, 535.6226508554284, 486.53187607797634,
                   447.317478343776, 412.20699767505823, 436.60111544327935]
    fc = forecast(fit; h=24)
    @test fc.mean ≈ expected_fc rtol=1e-6
end

@testset "TBATS parity: no Box-Cox, trend + damped, no ARMA" begin
    ap = air_passengers()
    fit = tbats(ap, 12; use_box_cox=false, use_trend=true, use_damped_trend=true, use_arma_errors=false)
    @test fit isa TBATSModel

    @test fit.alpha ≈ 1.2780588787108378 rtol=1e-6
    @test fit.beta ≈ -0.13404046932109637 rtol=1e-4
    @test fit.damping_parameter ≈ 0.8000027910379705 rtol=1e-4
    @test fit.aic ≈ 1558.7681363306365 rtol=1e-6
    @test fit.variance ≈ 275.5734815809231 rtol=1e-6
    @test fit.k_vector == [5]

    @test fit.gamma_one_values[1] ≈ -0.0015002931066820124 rtol=0.01
    @test fit.gamma_two_values[1] ≈ 0.0006690430847301612 rtol=0.01

    expected_fc = [449.16062946500887, 438.63263476201865, 471.3869017259752,
                   464.5821907022743, 467.60527211637833, 503.2971545575578,
                   541.4304832200908, 537.1046702481553, 486.7466987181322,
                   446.8146942326065, 411.55230591905877, 436.5285152016035,
                   447.50025026169703, 437.30432676518785, 470.3242516211524,
                   463.7320676525194, 466.9251713038487, 502.7530720093468,
                   540.9952156629671, 536.7564549876082, 486.4681255378125,
                   446.5918349108424, 411.3740178396387, 436.38588424045867]
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
    @test fit.aic ≈ 1400.700441721485959 rtol=0.001
    @test fit.k_vector == [5]

    expected_fc = [447.074104914059035, 437.884140051731833, 504.542810874353734,
                   489.151342161855382, 494.597479584298128, 558.208958308334900,
                   626.630177075764323, 620.565268251174757, 543.550241954130911,
                   472.893328735540194, 414.626334446593205, 464.347505159249238]
    fc = forecast(fit; h=12)
    @test fc.mean ≈ expected_fc rtol=0.01
end
