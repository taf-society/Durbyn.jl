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
const PI_RTOL   = 0.02    # prediction interval bounds

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

    r_lower80 = [429.768444376423, 401.023836973762, 429.104561764765,
                 462.860983031618, 472.642856151399, 532.550461556872,
                 616.261497429227, 601.368908116512, 502.021220270114,
                 452.784296874080, 383.148841567325, 423.866152063116]
    r_upper80 = [460.033795185955, 437.914420533442, 472.722813151709,
                 512.057959945296, 526.899209710608, 591.421703838750,
                 679.413791884321, 668.529356816808, 572.963854840696,
                 527.317408057110, 461.107246756407, 505.105557376003]
    r_lower95 = [421.757689910326, 391.259489325276, 417.559508006747,
                 449.839329821321, 458.282066834141, 516.968184979205,
                 599.546094788293, 583.592611130668, 483.243839155337,
                 433.056574005805, 362.514498140466, 402.363380361282]
    r_upper95 = [468.044549652052, 447.678768181928, 484.267866909727,
                 525.079613155594, 541.259999027866, 607.003980416417,
                 696.129194525254, 686.305653802652, 591.741235955473,
                 547.045130925385, 481.741590183266, 526.608329077837]
    @test fc.lower[:, 1] ≈ r_lower80 rtol=PI_RTOL
    @test fc.upper[:, 1] ≈ r_upper80 rtol=PI_RTOL
    @test fc.lower[:, 2] ≈ r_lower95 rtol=PI_RTOL
    @test fc.upper[:, 2] ≈ r_upper95 rtol=PI_RTOL
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

    r_lower80 = [412.802175513889, 373.324311893915, 345.221923532748,
                 322.431074859085, 303.099452795862, 286.329997518325,
                 271.591698237229, 258.530804687136, 246.892226641439,
                 236.481167364422, 227.142263585848, 218.747334237863]
    r_upper80 = [493.987696918660, 510.592593931408, 517.290692899889,
                 520.083973278549, 520.764473346445, 520.169845274610,
                 518.771894844887, 516.864862291717, 514.643674565389,
                 512.242121131634, 509.753674255848, 507.243840839406]
    r_lower95 = [391.313666021018, 336.991592332673, 299.678071377908,
                 270.115512792244, 245.487001961836, 224.436328531835,
                 206.167052200020, 190.153909975936, 176.022699281285,
                 163.491651866383, 152.339541673181, 142.386921525122]
    r_upper95 = [515.476206411531, 546.925313492650, 562.834545054729,
                 572.399535345389, 578.376924180471, 582.063514261100,
                 584.196540882096, 585.241757002917, 585.513201925543,
                 585.231636629673, 584.556396168515, 583.604253552147]
    @test fc.lower[:, 1] ≈ r_lower80 rtol=PI_RTOL
    @test fc.upper[:, 1] ≈ r_upper80 rtol=PI_RTOL
    @test fc.lower[:, 2] ≈ r_lower95 rtol=PI_RTOL
    @test fc.upper[:, 2] ≈ r_upper95 rtol=PI_RTOL
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

    r_lower80 = [404.359559158903, 383.113549270197, 364.834775066694,
                 349.485489845160, 336.472393255966, 325.268815878559,
                 315.489182236641, 306.857006015186, 299.167413595642,
                 292.260700965242, 286.005838101233, 280.290667032496]
    r_upper80 = [487.346325584093, 519.437140582910, 543.694128629568,
                 563.818402290683, 581.529180236514, 597.698053681230,
                 612.807063660661, 627.142533633665, 640.882397557726,
                 654.138954398560, 666.981874683641, 679.451850074060]
    r_lower95 = [382.394288908072, 347.030874239753, 317.493563845518,
                 292.754994751615, 271.609779797225, 253.161152327675,
                 236.793892647666, 222.082547693428, 208.720920801497,
                 196.477313119580, 185.167572398152, 174.639083812262]
    r_upper95 = [509.311595834923, 555.519815613354, 591.035339850744,
                 620.548897384229, 646.391793695255, 669.805717232114,
                 691.502353249636, 711.916991955422, 731.328890351871,
                 749.922342244222, 767.820140386721, 785.103433294294]
    @test fc.lower[:, 1] ≈ r_lower80 rtol=PI_RTOL
    @test fc.upper[:, 1] ≈ r_upper80 rtol=PI_RTOL
    @test fc.lower[:, 2] ≈ r_lower95 rtol=PI_RTOL
    @test fc.upper[:, 2] ≈ r_upper95 rtol=PI_RTOL
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

    r_lower80 = [426.633912143203, 393.165284655649, 415.584426824910,
                 453.085724165377, 460.341182502980, 520.171842383934,
                 604.462747291092, 586.131497371383, 486.114188975783,
                 437.358374886349, 364.819302454651, 405.457847098947]
    r_upper80 = [457.878710629409, 436.151621193357, 466.828755569563,
                 510.711424327290, 523.118466195716, 587.214833900439,
                 675.096839702014, 659.824019303226, 562.433796126843,
                 515.949286027687, 445.383953784229, 487.744667372990]
    r_lower95 = [418.363913536848, 381.787488654945, 402.020872318742,
                 437.833122672694, 443.725039215038, 502.426634436692,
                 585.767032352896, 566.626265038258, 465.913609187902,
                 416.556617246658, 343.495127343992, 383.677841403338]
    r_upper95 = [466.148709235764, 447.529417194061, 480.392310075732,
                 525.964025819973, 539.734609483658, 604.960041847681,
                 693.792554640210, 679.329251636352, 582.634375914723,
                 536.751043667378, 466.708128894888, 509.524673068600]
    @test fc.lower[:, 1] ≈ r_lower80 rtol=PI_RTOL
    @test fc.upper[:, 1] ≈ r_upper80 rtol=PI_RTOL
    @test fc.lower[:, 2] ≈ r_lower95 rtol=PI_RTOL
    @test fc.upper[:, 2] ≈ r_upper95 rtol=PI_RTOL
end
