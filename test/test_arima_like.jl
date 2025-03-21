using Test
using LinearAlgebra


@testset "Test arima likelihood function" begin

    ## -------------------------------
    ## Test 0: Provided core test case
    ## -------------------------------
    y = collect(0:9)
    phi = [0.99551517]
    theta = Float64[]
    delta = [1.0]
    a = [0.0, 0.0]
    P = zeros(2, 2)
    Pn = [5.32878591e+02  0.0; 0.0  1.0e6]
    up = 0
    use_resid = true

    ssq, sumlog, nu, rsResid = arima_like(y, phi, theta, delta, copy(a), copy(P), copy(Pn), up, use_resid)
    @test isapprox(ssq, 0.002051882527695168; atol=1e-10)
    @test isapprox(sumlog, 6.270663806368853; atol=1e-10)
    @test nu == 9
    @test all(isapprox.(rsResid, [0.0, 0.04348532, 0.00448483, 0.00448483, 0.00448483,
                                  0.00448483, 0.00448483, 0.00448483, 0.00448483, 0.00448483]; atol=1e-6))


    ## -------------------------------
    ## Example 1: Pure AR(1)
    ## -------------------------------
    y = collect(0:9)
    phi = [0.9]
    theta = Float64[]
    delta = Float64[]
    a = zeros(1)
    P = zeros(1, 1)
    Pn = [100.0]

    res1 = arima_like(y, phi, theta, delta, copy(a), copy(P), copy(Pn), up, use_resid)
    @test isapprox(res1[1], 18.24; atol=1e-2)
    @test isapprox(res1[2], log(100.0); atol=1e-10)
    @test res1[3] == 10
    @test all(isapprox.(res1[4], [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]; atol=1e-6))


    ## -------------------------------
    ## Example 2: ARMA(1,1)
    ## -------------------------------
    y = sin.(range(0, stop=3π, length=10))
    phi = [0.7]
    theta = [0.4]
    delta = Float64[]
    a = zeros(1)
    P = zeros(1, 1)
    Pn = [50.0]

    res2 = arima_like(y, phi, theta, delta, copy(a), copy(P), copy(Pn), up, use_resid)
    expected2 = [0.0, 0.8660254, 0.25980762, -0.60621778, -0.8660254, -0.25980762,
                 0.60621778, 0.8660254, 0.25980762, -0.60621778]
    @test isapprox(res2[1], 3.555; atol=1e-3)
    @test isapprox(res2[2], log(50.0); atol=1e-10)
    @test res2[3] == 10
    @test all(isapprox.(res2[4], expected2; atol=1e-6))


    ## -------------------------------
    ## Example 3: I(1) process
    ## -------------------------------
    y = [0.09647446, -0.60512967, -0.21111742, 3.03379793, 3.07952418, 3.81194573,
         4.25539357, 3.09052676, 3.05444808, 3.33943849]
    phi = Float64[]
    theta = Float64[]
    delta = [1.0]
    a = zeros(2)
    P = zeros(2, 2)
    Pn = Diagonal([100.0, 1e6]) |> Matrix

    res3 = arima_like(y, phi, theta, delta, copy(a), copy(P), copy(Pn), up, use_resid)
    expected3 = [9.64696351e-05, -0.70160413, 0.39401225, 3.24491534, 0.04572625,
                 0.73242155, 0.44344784, -1.16486680, -0.03607869, 0.28499041]
    @test isapprox(res3[1], 13.351583668356358; atol=1e-6)
    @test isapprox(res3[2], -1.8118839762046702e-11; atol=1e-8)
    @test res3[3] == 9
    @test all(isapprox.(res3[4], expected3; atol=1e-6))


    ## -------------------------------
    ## Example 4: ARIMA(1,1,1)
    ## -------------------------------
    y = [-1.66366425, -3.11626561, -2.50684729, -4.02478614, -4.16438822,
         -5.90046219, -6.2663671, -7.31741344, -6.59264671, -7.89426038]
    phi = [0.6]
    theta = [0.3]
    delta = [1.0]
    a = zeros(2)
    P = zeros(2, 2)
    Pn = Diagonal([10.0, 1e4]) |> Matrix

    res4 = arima_like(y, phi, theta, delta, copy(a), copy(P), copy(Pn), up, use_resid)
    expected4 = [-0.01662833, -0.63134689, 1.08275006, -1.29980383, 0.41324933,
                 -1.21379072, 0.35173627, -0.59091605, 0.96079203, -1.20271627]
    @test isapprox(res4[1], 7.747047423479254; atol=1e-6)
    @test isapprox(res4[2], 5.876176363102521; atol=1e-6)
    @test res4[3] == 9
    @test all(isapprox.(res4[4], expected4; atol=1e-6))


    ## -------------------------------
    ## Example 5: ARIMA(2,1,2)
    ## -------------------------------
    y = [-1.49622813, 0.41808685, -1.15936328, -1.34360461, 0.37170564,
         0.72984745, -0.11488035, 0.1306029, -0.58978569, -0.79444588,
         -0.61892474, -2.63854473, -2.52773286, -0.09869691, -1.68666057,
         -1.9517678, -4.38809894, -4.53240583, -3.59747028, -4.67863048]
    phi = [0.5, -0.2]
    theta = [0.3, 0.1]
    delta = [1.0]
    a = zeros(3)
    P = zeros(3, 3)
    Pn = Diagonal([20.0, 1e3, 1e3]) |> Matrix

    res5 = arima_like(y, phi, theta, delta, copy(a), copy(P), copy(Pn), up, use_resid)
    expected5 = [-0.04684872, 0.06081426, -1.77036579, 1.19983075, 1.1088675,
                 -0.76121543, -0.46600466, 0.79200086, -1.09159627, 0.41971801,
                 0.04606903, -1.97111292, 1.49780267, 1.48607056, -2.89253905,
                 1.55281377, -2.68338447, 1.50966098, 0.17825019, -1.50204102]
    @test isapprox(res5[1], 39.48565811467588; atol=1e-6)
    @test isapprox(res5[2], 17.822395581507877; atol=1e-6)
    @test res5[3] == 20
    @test all(isapprox.(res5[4], expected5; atol=1e-6))

end
