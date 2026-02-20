using Test
using Durbyn
using Durbyn.Utils: NamedMatrix
import Durbyn.Generics: Forecast, forecast, fitted, residuals
import Durbyn.Arima: arima_rjh, arima, auto_arima, ArimaFit, PDQ

const AP = air_passengers()
const N = length(AP)

# Deterministic xreg shared across tests (same formula used in R)
const XREG_SINGLE = NamedMatrix(
    reshape(sin.((1:N) ./ 5.0), :, 1),
    ["temperature"],
)
const NEWXREG_SINGLE = NamedMatrix(
    reshape(sin.(((N+1):(N+12)) ./ 5.0), :, 1),
    ["temperature"],
)

const XREG_MULTI = NamedMatrix(
    hcat(sin.((1:N) ./ 5.0), cos.((1:N) ./ 7.0)),
    ["temperature", "marketing"],
)
const NEWXREG_MULTI = NamedMatrix(
    hcat(sin.(((N+1):(N+12)) ./ 5.0), cos.(((N+1):(N+12)) ./ 7.0)),
    ["temperature", "marketing"],
)

const XREG_DRIFT = NamedMatrix(
    reshape(sin.((1:N) ./ 5.0), :, 1),
    ["exog"],
)
const NEWXREG_DRIFT = NamedMatrix(
    reshape(sin.(((N+1):(N+12)) ./ 5.0), :, 1),
    ["exog"],
)

# Tolerances: ARMA coefficients may differ due to optimizer differences
# between optimizers; xreg coefficients and forecasts are tighter.
const COEF_RTOL = 0.15    # ARMA coefs can have moderate optimizer-driven differences
const STAT_RTOL = 0.02    # sigma2, loglik, AIC
const FC_RTOL   = 0.02    # forecast point predictions

# ── Test 1: ARIMAX(1,1,1)(0,1,1)[12] with single xreg ──────────────────────
@testset "ARIMAX parity: single xreg ARIMA(1,1,1)(0,1,1)[12]" begin
    fit = arima_rjh(AP, 12;
        order=PDQ(1,1,1), seasonal=PDQ(0,1,1),
        xreg=XREG_SINGLE, include_mean=false)
    @test fit isa ArimaFit

    coefs = vec(fit.coef.data)
    r_coefs = [-0.237662748113926, -0.065394580355314,
               -0.103825428092521, 1.221723517033303]
    @test coefs ≈ r_coefs rtol=COEF_RTOL

    @test fit.sigma2 ≈ 139.430974322084580 rtol=STAT_RTOL
    @test fit.loglik ≈ -507.371528364868254 rtol=STAT_RTOL

    fc = forecast(fit; xreg=NEWXREG_SINGLE)
    r_fc = [444.901119781189209, 419.469128753601865, 450.913687458237121,
            487.459471488457211, 499.771032931003447, 561.986082697811071,
            647.837644656773932, 634.949132466659989, 537.492537555404965,
            490.050852465595085, 422.128044161865887, 464.485854719559313]
    @test fc.mean ≈ r_fc rtol=FC_RTOL
end

# ── Test 2: ARIMAX(1,0,1) multiple xreg with intercept ──────────────────────
@testset "ARIMAX parity: multiple xreg ARIMA(1,0,1) + intercept" begin
    fit = arima_rjh(AP, 12;
        order=PDQ(1,0,1), seasonal=PDQ(0,0,0),
        xreg=XREG_MULTI, include_mean=true)
    @test fit isa ArimaFit

    coefs = vec(fit.coef.data)
    r_coefs = [0.937376662112497, 0.426000179172340,
               281.095810359454049, 0.736972182652545, 3.407768572839609]
    @test coefs ≈ r_coefs rtol=COEF_RTOL

    @test fit.sigma2 ≈ 1003.286586932635714 rtol=STAT_RTOL
    @test fit.loglik ≈ -700.868096607563984 rtol=STAT_RTOL

    fc = forecast(fit; xreg=NEWXREG_MULTI)
    r_fc = [453.394936216274459, 441.958452912661528, 431.256308216318530,
            421.257524068816963, 411.931963071153575, 403.249921396467471,
            395.181796541058247, 387.697833489426614, 380.767950603413851,
            374.361644248028028, 368.447968920848098, 362.995587538634254]
    @test fc.mean ≈ r_fc rtol=FC_RTOL
end

# ── Test 3: ARIMAX(1,1,0) with drift + xreg ─────────────────────────────────
@testset "ARIMAX parity: drift + xreg ARIMA(1,1,0)" begin
    fit = arima_rjh(AP, 12;
        order=PDQ(1,1,0), seasonal=PDQ(0,0,0),
        xreg=XREG_DRIFT, include_drift=true)
    @test fit isa ArimaFit

    coefs = vec(fit.coef.data)
    r_coefs = [0.303269730115492, 2.394386013418049, 5.179498175669268]
    @test coefs ≈ r_coefs rtol=COEF_RTOL

    @test fit.sigma2 ≈ 1048.299848405887587 rtol=STAT_RTOL
    @test fit.loglik ≈ -698.717620058293960 rtol=STAT_RTOL

    fc = forecast(fit; xreg=NEWXREG_DRIFT)
    r_fc = [445.852942371497534, 451.275344926553316, 454.264451848131102,
            456.651946067921699, 459.000786746240124, 461.483434779894310,
            464.148122948651178, 466.999769824425300, 470.024905576684318,
            473.199827681900786, 476.493856392436783, 479.871258553277983]
    @test fc.mean ≈ r_fc rtol=FC_RTOL
end

# ── Test 4: auto.arima with xreg ────────────────────────────────────────────
# auto.arima involves multiple model evaluations; small optimizer differences
# accumulate, so we use wider tolerances and focus on model selection + forecasts.
@testset "ARIMAX parity: auto.arima with xreg" begin
    fit = auto_arima(AP, 12;
        xreg=XREG_MULTI, stepwise=true, approximation=false)
    @test fit isa ArimaFit

    # Expected order: ARIMA(2,0,0)(0,1,0)[12]
    @test fit.arma[1] == 2  # p
    @test fit.arma[6] == 0  # d
    @test fit.arma[2] == 0  # q
    @test fit.arma[7] == 1  # D

    coefs = vec(fit.coef.data)
    r_coefs = [0.563291338457426, 0.196532395572795,
               2.615065458334324, 0.990922198570922, 6.733983866051725]
    @test coefs ≈ r_coefs rtol=0.20

    @test fit.sigma2 ≈ 127.978781582187267 rtol=0.05
    @test fit.loglik ≈ -505.351352505748423 rtol=STAT_RTOL

    fc = forecast(fit; xreg=NEWXREG_MULTI)
    r_fc = [441.728418926323855, 414.280489743738769, 441.128845869878205,
            482.384956764328990, 493.002705521609585, 555.970911940566907,
            643.265470920342977, 627.860488380107881, 530.725442406054754,
            484.826093727124089, 415.124777690207907, 458.581090143451604]
    @test fc.mean ≈ r_fc rtol=FC_RTOL
end

# ── Test 5: ARIMAX CSS method ───────────────────────────────────────────────
@testset "ARIMAX parity: CSS method ARIMA(1,0,0)(0,1,0)[12]" begin
    fit = arima_rjh(AP, 12;
        order=PDQ(1,0,0), seasonal=PDQ(0,1,0),
        xreg=XREG_SINGLE, include_mean=false, method="CSS")
    @test fit isa ArimaFit

    coefs = vec(fit.coef.data)
    r_coefs = [0.944882525926739, 0.904614287585627]
    @test coefs ≈ r_coefs rtol=COEF_RTOL

    @test fit.sigma2 ≈ 148.601534375849241 rtol=STAT_RTOL

    fc = forecast(fit; xreg=NEWXREG_SINGLE)
    r_fc = [442.256311386305754, 414.658452924502967, 441.206591197236662,
            481.898574246333681, 491.729824349348178, 553.693338142186349,
            639.779793496552884, 622.977758337304863, 524.273992551312858,
            476.653830457017818, 405.101628119440022, 446.601257235968944]
    @test fc.mean ≈ r_fc rtol=FC_RTOL
end
