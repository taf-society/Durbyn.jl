using Test
using Durbyn.Optimize
using LinearAlgebra: dot

const EPS_OPT = 1e-4
const EPS_GRAD = 1e-5

rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function rosenbrock_grad(x)
    dx1 = -2.0 * (1.0 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2)
    dx2 = 200.0 * (x[2] - x[1]^2)
    return [dx1, dx2]
end

sphere(x) = sum(x .^ 2)

sphere_grad(x) = 2.0 .* x


@testset "Durbyn.Optimize Module Tests" begin

    @testset "BFGS via optimize - Rosenbrock with analytic gradient" begin
        x0 = [-1.2, 1.0]
        result = optimize(x0, rosenbrock; gr=rosenbrock_grad, method="BFGS")
        @test abs(result.par[1] - 1.0) <= 0.01
        @test abs(result.par[2] - 1.0) <= 0.01
        @test result.value < 1e-8
        @test result.convergence == 0
    end

    @testset "BFGS via optimize - Sphere with gradient" begin
        x0 = [5.0, -3.0, 2.0]
        result = optimize(x0, sphere; gr=sphere_grad, method="BFGS")
        @test all(abs.(result.par) .<= 0.01)
        @test result.value < 1e-6
    end

    @testset "BFGS via optimize - without gradient (numerical)" begin
        x0 = [5.0, -3.0]
        result = optimize(x0, sphere; method="BFGS")
        @test all(abs.(result.par) .<= 0.01)
    end

    @testset "bfgs - Sphere with analytic gradient" begin
        f_bfgs(n, x, ex) = sum(x .^ 2)
        function g_bfgs(n, x, grad, ex)
            grad .= 2.0 .* x
            return nothing
        end
        x0 = [5.0, -3.0]
        result = bfgs(f_bfgs, g_bfgs, x0)
        @test all(abs.(result.x_opt) .<= 0.01)
        @test result.f_opt < 1e-6
        @test result.fail == 0
    end

    @testset "bfgs - with mask (partial optimization)" begin
        f_bfgs(n, x, ex) = (x[1] - 3.0)^2 + (x[2] - 5.0)^2
        function g_bfgs(n, x, grad, ex)
            grad[1] = 2.0 * (x[1] - 3.0)
            grad[2] = 2.0 * (x[2] - 5.0)
            return nothing
        end
        x0 = [0.0, 0.0]
        mask = BitVector([true, false])
        result = bfgs(f_bfgs, g_bfgs, x0; mask=mask)
        @test abs(result.x_opt[1] - 3.0) <= 0.01
        @test result.x_opt[2] == 0.0
    end

    @testset "L-BFGS-B via optimize - Rosenbrock with bounds" begin
        x0 = [-1.2, 1.0]
        result = optimize(x0, rosenbrock; method="L-BFGS-B",
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
        @test abs(result.par[1] - 1.0) <= 0.1
        @test abs(result.par[2] - 1.0) <= 0.1
    end

    @testset "L-BFGS-B via optimize - Sphere with active lower bounds" begin
        x0 = [2.0, 2.0]
        result = optimize(x0, sphere; method="L-BFGS-B",
                       lower=[1.0, 1.0], upper=[10.0, 10.0])
        @test abs(result.par[1] - 1.0) <= 0.1
        @test abs(result.par[2] - 1.0) <= 0.1
    end

    @testset "L-BFGS-B via optimize - with gradient" begin
        x0 = [5.0, -3.0]
        result = optimize(x0, sphere; gr=sphere_grad, method="L-BFGS-B",
                       lower=[-10.0, -10.0], upper=[10.0, 10.0])
        @test all(abs.(result.par) .<= 0.01)
    end

    @testset "Brent via optimize - simple quadratic" begin
        f1d(x) = (x[1] - 2.0)^2
        result = optimize([3.0], f1d; method="Brent", lower=-10.0, upper=10.0)
        @test abs(result.par[1] - 2.0) <= EPS_OPT
        @test result.value < 1e-8
    end

    @testset "Brent via optimize - minimum at non-zero" begin
        f1d(x) = (x[1] + 3.0)^2
        result = optimize([0.0], f1d; method="Brent", lower=-10.0, upper=10.0)
        @test abs(result.par[1] - (-3.0)) <= EPS_OPT
    end

    @testset "Brent via optimize - minimum at boundary" begin
        f1d(x) = x[1]^2
        result = optimize([5.0], f1d; method="Brent", lower=2.0, upper=10.0)
        @test abs(result.par[1] - 2.0) <= 0.01
    end

    @testset "numgrad - Sphere gradient" begin
        f_ng(n, x, ex) = sum(x .^ 2)
        x = [1.0, 2.0, 3.0]
        grad = numgrad(f_ng, 3, x, nothing)
        @test isapprox(grad, 2.0 .* x, atol=1e-4)
    end

    @testset "numgrad - Quadratic gradient" begin
        A = [2.0 1.0; 1.0 3.0]
        f_quad(n, x, ex) = dot(x, ex * x)
        x = [1.0, 2.0]
        grad = numgrad(f_quad, 2, x, A)
        analytic = 2.0 .* (A * x)
        @test isapprox(grad, analytic, atol=1e-3)
    end

    @testset "numgrad - with explicit ndeps" begin
        f_ng(n, x, ex) = sum(x .^ 2)
        x = [1.0, -2.0]
        ndeps = fill(1e-5, 2)
        grad = numgrad(f_ng, 2, x, nothing, ndeps)
        @test isapprox(grad, 2.0 .* x, atol=1e-4)
    end

    @testset "numerical_hessian - Sphere at origin" begin
        x = [0.0, 0.0]
        H = numerical_hessian(sphere, x)
        @test isapprox(H, 2.0 * [1.0 0.0; 0.0 1.0], atol=0.1)
    end

    @testset "numerical_hessian - Rosenbrock at (1,1)" begin
        x = [1.0, 1.0]
        H = numerical_hessian(rosenbrock, x)
        @test isapprox(H[1,1], 802.0, rtol=0.05)
        @test isapprox(H[1,2], -400.0, rtol=0.05)
        @test isapprox(H[2,1], -400.0, rtol=0.05)
        @test isapprox(H[2,2], 200.0, rtol=0.05)
    end

    @testset "L-BFGS-B unbounded convergence (bug fix)" begin
        result = optimize([5.0, 3.0], sphere; method="L-BFGS-B")
        @test result.convergence == 0
        @test all(abs.(result.par) .< 0.1)
        @test result.value < 0.01
    end

    @testset "L-BFGS-B one-sided bounds convergence (bug fix)" begin
        result = optimize([5.0, 5.0], sphere; method="L-BFGS-B",
                       lower=[0.0, 0.0], upper=[Inf, Inf])
        @test result.convergence == 0
        @test all(abs.(result.par) .< 0.1)
    end

    @testset "L-BFGS-B bounded numgrad stays in bounds (bug fix)" begin
        f_sqrt(x) = sum(sqrt.(x))
        result = optimize([4.0, 4.0], f_sqrt; method="L-BFGS-B",
                       lower=[0.001, 0.001], upper=[10.0, 10.0])
        @test all(result.par .>= 0.0)
        @test result.convergence == 0
    end

    @testset "numerical_hessian with parscale (bug fix)" begin
        f_cross(x) = x[1] * x[2]
        H = numerical_hessian(f_cross, [2.0, 3.0]; parscale=[2.0, 5.0])
        @test isapprox(H[1,2], 1.0, atol=0.01)
        @test isapprox(H[2,1], 1.0, atol=0.01)
        @test isapprox(H[1,1], 0.0, atol=0.01)
        @test isapprox(H[2,2], 0.0, atol=0.01)
    end

    @testset "Unknown control key warning (bug fix)" begin
        @test_warn "unknown names in control" optimize([1.0, 1.0], sphere;
                                                     control=Dict("bogus" => 42))
    end

    @testset "Brent requires finite bounds (bug fix)" begin
        @test_throws ErrorException optimize([1.0], x -> x[1]^2;
                                          method="Brent", lower=-Inf, upper=Inf)
        @test_throws ErrorException optimize([1.0], x -> x[1]^2;
                                          method="Brent", lower=-Inf, upper=10.0)
    end

    @testset "Brent handles non-finite evaluations (bug fix)" begin
        f_nan(x) = x < 2.0 ? NaN : (x - 3.0)^2
        result = @test_warn r"NaN|Inf" brent(f_nan, 0.0, 5.0)
        @test isfinite(result.f_opt)
    end

    @testset "L-BFGS-B convergence message (bug fix)" begin
        result = optimize([5.0, 3.0], sphere; method="L-BFGS-B")
        @test result.message !== nothing
        @test occursin("CONVERGENCE", result.message)
    end

    @testset "L-BFGS-B infeasible bounds returns code 52 (bug fix)" begin
        result = optimize([1.0, 1.0], sphere; method="L-BFGS-B",
                       lower=[5.0, 5.0], upper=[0.0, 0.0])
        @test result.convergence == 52
        @test occursin("NO FEASIBLE SOLUTION", result.message)
    end

    @testset "L-BFGS-B maxit convergence code (bug fix)" begin
        result = optimize([-1.2, 1.0], rosenbrock; method="L-BFGS-B",
                       lower=[-5.0, -5.0], upper=[5.0, 5.0],
                       control=Dict("maxit" => 1))
        @test result.convergence == 1
    end

    @testset "L-BFGS-B errors on non-finite fn" begin
        f_nan(x) = NaN
        @test_throws ErrorException optimize([1.0, 1.0], f_nan; method="L-BFGS-B")
    end

    @testset "L-BFGS-B errors on mid-iteration non-finite fn" begin
        calls = Ref(0)
        f_delayed_nan(x) = begin
            calls[] += 1
            calls[] > 2 ? NaN : sum(x .^ 2)
        end
        gr_delayed_nan(x) = 2.0 .* x
        @test_throws ErrorException optimize([5.0, 3.0], f_delayed_nan; gr=gr_delayed_nan,
                                          method="L-BFGS-B")
    end

    @testset "L-BFGS-B errors on non-finite gradient" begin
        f_sphere(x) = sum(x .^ 2)
        gr_inf(x) = [Inf, Inf]
        @test_throws ErrorException optimize([1.0, 1.0], f_sphere; gr=gr_inf,
                                          method="L-BFGS-B")
    end

    @testset "Brent returns convergence=0 and nothing counts" begin
        f1d(x) = (x[1] - 2.0)^2
        result = optimize([3.0], f1d; method="Brent", lower=-10.0, upper=10.0)
        @test result.convergence == 0
        @test result.counts.function_ === nothing
        @test result.counts.gradient === nothing
    end

    @testset "Brent convergence=0 even with small maxit" begin
        f1d(x) = (x[1] - 2.0)^2
        result = optimize([3.0], f1d; method="Brent", lower=-10.0, upper=10.0,
                       control=Dict("maxit" => 2))
        @test result.convergence == 0
    end

    @testset "Bounds length recycling" begin
        result = optimize([5.0, 5.0, 5.0], sphere; method="L-BFGS-B",
                       lower=[0.0], upper=[10.0, 10.0, 10.0])
        @test length(result.par) == 3
        @test all(result.par .>= 0.0)

        result2 = optimize([5.0, 5.0], sphere; method="L-BFGS-B",
                        lower=[-10.0, -10.0, -10.0], upper=[10.0, 10.0, 10.0])
        @test length(result2.par) == 2

        result3 = optimize([5.0, 5.0], sphere; method="L-BFGS-B",
                        lower=Float64[], upper=[10.0, 10.0])
        @test result3.convergence == 0
        @test all(abs.(result3.par) .< 0.1)

        @test_throws ErrorException optimize([0.0], x -> x[1]^2;
                       method="Brent", lower=Float64[], upper=[5.0])
    end

    @testset "parscale/ndeps wrong length errors" begin
        @test_throws ErrorException optimize([5.0, 5.0], sphere; method="BFGS",
                       control=Dict("parscale" => [1.0]))
        @test_throws ErrorException optimize([5.0, 5.0], sphere; method="BFGS",
                       control=Dict("ndeps" => [1e-4]))

        result = optimize([5.0, 5.0], sphere; method="BFGS",
                       control=Dict("parscale" => 2.0))
        @test result.convergence == 0
        result2 = optimize([5.0, 5.0], sphere; method="BFGS",
                        control=Dict("ndeps" => 1e-4))
        @test result2.convergence == 0
    end

    @testset "L-BFGS-B catch does not swallow user errors (bug fix)" begin
        f_user_error(x) = error("finite values in custom code")
        @test_throws ErrorException optimize([1.0, 1.0], f_user_error; method="L-BFGS-B")
    end

    @testset "Integer par and bounds accepted" begin
        result = optimize([5, 3], sphere; method="BFGS")
        @test all(abs.(result.par) .< 0.1)

        result2 = optimize([5, 5], sphere; method="L-BFGS-B",
                        lower=[0, 0], upper=[10, 10])
        @test all(result2.par .>= 0.0)
    end

    @testset "Brent errors when lower >= upper" begin
        @test_throws ErrorException optimize([0.0], x -> x[1]^2;
                       method="Brent", lower=5.0, upper=1.0)
        @test_throws ErrorException optimize([0.0], x -> x[1]^2;
                       method="Brent", lower=3.0, upper=3.0)
    end

    @testset "Gradient length validated" begin
        bad_gr(x) = [1.0]
        @test_throws ErrorException optimize([1.0, 1.0], sphere;
                       gr=bad_gr, method="BFGS")
        @test_throws ErrorException optimize([1.0, 1.0], sphere;
                       gr=bad_gr, method="L-BFGS-B",
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
    end

    @testset "L-BFGS-B zero-parameter returns NOTHING TO DO" begin
        f_empty(x) = 42.0
        result = optimize(Float64[], f_empty; method="L-BFGS-B")
        @test result.convergence == 0
        @test result.value == 42.0
        @test result.counts.function_ == 1
        @test result.counts.gradient == 0
        @test result.message == "NOTHING TO DO"
        @test isempty(result.par)
    end

    @testset "Objective returning non-scalar errors clearly" begin
        fn_vec(x) = [x[1]^2, x[1]]
        @test_throws ErrorException optimize([1.0, 1.0], fn_vec)
        @test_throws ErrorException optimize([1.0, 1.0], fn_vec; method="BFGS")
        @test_throws ErrorException optimize([1.0, 1.0], fn_vec; method="L-BFGS-B",
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
    end

    @testset "Direct brent errors when lower >= upper" begin
        f(x) = (x - 2.0)^2
        @test_throws ErrorException brent(f, 5.0, -5.0)
        @test_throws ErrorException brent(f, 3.0, 3.0)
        result = brent(f, 0.0, 5.0)
        @test abs(result.x_opt - 2.0) < 0.01
    end

    @testset "L-BFGS-B error on delayed NaN fn throws" begin
        calls = Ref(0)
        f_delayed_nan2(x) = begin
            calls[] += 1
            calls[] > 2 ? NaN : sum(x .^ 2)
        end
        gr_for_count(x) = 2.0 .* x
        @test_throws ErrorException optimize([5.0, 3.0], f_delayed_nan2; gr=gr_for_count,
                                          method="L-BFGS-B")
    end

    @testset "Length-1 vector objective accepted" begin
        fn_vec1(x) = [sum(x .^ 2)]
        result = optimize([5.0, 3.0], fn_vec1)
        @test result.value < 0.1

        result2 = optimize([5.0, 3.0], fn_vec1; method="BFGS")
        @test result2.value < 0.01

        result3 = optimize([5.0, 3.0], fn_vec1; method="L-BFGS-B",
                        lower=[-10.0, -10.0], upper=[10.0, 10.0])
        @test result3.value < 0.1

        fn_vec2(x) = [x[1]^2, x[1]]
        @test_throws ErrorException optimize([1.0, 1.0], fn_vec2)
    end

    @testset "Brent ignores maxit from control" begin
        f1d(x) = (x[1] - 2.0)^2
        result_small = optimize([0.0], f1d; method="Brent", lower=-10.0, upper=10.0,
                             control=Dict("maxit" => 1))
        result_big = optimize([0.0], f1d; method="Brent", lower=-10.0, upper=10.0,
                           control=Dict("maxit" => 10000))
        @test abs(result_small.par[1] - 2.0) < 0.01
        @test abs(result_big.par[1] - 2.0) < 0.01
    end

    @testset "Direct nelder_mead with negative objective (abstol fix)" begin
        f_neg(x) = (x[1] - 1.0)^2 + (x[2] - 1.0)^2 - 10.0
        opts = NelderMeadOptions()
        result = nelder_mead(f_neg, [0.0, 0.0], opts)
        @test result.f_opt < -9.99
    end

    @testset "ndeps wrong length only errors for gradient methods" begin
        result_nm = optimize([5.0, 5.0], sphere;
                          control=Dict("ndeps" => [1e-3]))
        @test result_nm.convergence == 0 || result_nm.value < 0.1

        f1d(x) = (x[1] - 2.0)^2
        result_brent = optimize([0.0], f1d; method="Brent", lower=-10.0, upper=10.0,
                             control=Dict("ndeps" => [1e-3, 1e-3, 1e-3]))
        @test abs(result_brent.par[1] - 2.0) < 0.01

        @test_throws ErrorException optimize([5.0, 5.0], sphere; method="BFGS",
                       control=Dict("ndeps" => [1e-4]))
    end

    @testset "Brent ignores parscale" begin
        f1d(x) = (x[1] - 2.0)^2
        result = optimize([0.0], f1d; method="Brent", lower=-10.0, upper=10.0,
                       control=Dict("parscale" => [1.0, 2.0, 3.0]))
        @test abs(result.par[1] - 2.0) < 0.01
    end

    @testset "NM gradient count is nothing" begin
        result = optimize([5.0, 5.0], sphere)
        @test result.counts.gradient === nothing
    end

    @testset "warn.1d.NelderMead control suppresses warning" begin
        @test_warn "Nelder-Mead is unreliable" optimize([5.0], sphere)

        @test_nowarn optimize([5.0], sphere; control=Dict("warn.1d.NelderMead" => false))
    end

    @testset "fn returning nothing/missing gives clear error" begin
        @test_throws ErrorException optimize([1.0, 1.0], x -> nothing)
        @test_throws ErrorException optimize([1.0, 1.0], x -> missing)
    end

    @testset "gr returning nothing gives clear error" begin
        @test_throws ErrorException optimize([1.0, 1.0], sphere;
                       gr=x -> nothing, method="BFGS")
    end

    @testset "Symbol keys in control accepted (convenience)" begin
        result = optimize([5.0, 5.0], sphere; control=Dict(:maxit => 1000))
        @test result.convergence == 0

        @test_warn "unknown names in control" optimize([5.0, 5.0], sphere;
                                                     control=Dict(:bogus => 42))
    end

    @testset "L-BFGS-B converges with numerical gradient (aliasing fix)" begin
        result = optimize([-1.2, 1.0], rosenbrock; method="L-BFGS-B",
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
        @test result.convergence == 0
        @test result.value < 1e-4
        @test abs(result.par[1] - 1.0) < 0.01
        @test abs(result.par[2] - 1.0) < 0.01

        result2 = optimize([-1.2, 1.0], rosenbrock; method="L-BFGS-B")
        @test result2.convergence == 0
        @test result2.value < 1e-4
    end

    @testset "L-BFGS-B Beale from [4,4] bounded (GCP fix)" begin
        beale(x) = (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1] + x[1]*x[2]^2)^2 +
                   (2.625 - x[1] + x[1]*x[2]^3)^2
        r = optimize([4.0, 4.0], beale; method="L-BFGS-B",
                  lower=[-4.5, -4.5], upper=[4.5, 4.5])
        @test r.convergence == 0
        @test r.value < 1e-6
        @test abs(r.par[1] - 3.0) < 0.01
        @test abs(r.par[2] - 0.5) < 0.01
    end

    @testset "L-BFGS-B Rosenbrock from [0,0] bounded (GCP fix)" begin
        r = optimize([0.0, 0.0], rosenbrock; method="L-BFGS-B",
                  lower=[-5.0, -5.0], upper=[5.0, 5.0])
        @test r.value < 1e-4
        @test abs(r.par[1] - 1.0) < 0.01
        @test abs(r.par[2] - 1.0) < 0.01
    end

    @testset "L-BFGS-B unbounded Rosenbrock (GCP fix)" begin
        r = optimize([-1.2, 1.0], rosenbrock; method="L-BFGS-B")
        @test r.convergence == 0
        @test r.value < 1e-4
        @test abs(r.par[1] - 1.0) < 0.01
        @test abs(r.par[2] - 1.0) < 0.01
    end

    @testset "lbfgsb NaN gradient does not falsely converge (bug fix)" begin
        using Durbyn.Optimize: lbfgsb
        f_ok(n, x, _) = sum(x .^ 2)
        g_nan(n, x, _) = [NaN, NaN]
        r = lbfgsb(f_ok, g_nan, [1.0, 1.0])
        @test r.fail != 0
        @test occursin("NON-FINITE", r.message)
    end

    @testset "lbfgsb Inf gradient returns non-finite error (bug fix)" begin
        using Durbyn.Optimize: lbfgsb
        f_ok(n, x, _) = sum(x .^ 2)
        g_inf(n, x, _) = [Inf, Inf]
        r = lbfgsb(f_ok, g_inf, [1.0, 1.0])
        @test r.fail != 0
        @test occursin("NON-FINITE", r.message)
    end

    @testset "bfgs NaN gradient does not falsely converge (bug fix)" begin
        using Durbyn.Optimize: bfgs
        f_bfgs(n, x, ex) = sum(x .^ 2)
        g_nan_bfgs(n, x, gvec, ex) = (gvec .= NaN; nothing)
        r = bfgs(f_bfgs, g_nan_bfgs, [1.0, 1.0])
        @test r.fail != 0
    end

    @testset "bfgs Inf gradient does not falsely converge (bug fix)" begin
        using Durbyn.Optimize: bfgs
        f_bfgs2(n, x, ex) = sum(x .^ 2)
        g_inf_bfgs(n, x, gvec, ex) = (gvec .= Inf; nothing)
        r = bfgs(f_bfgs2, g_inf_bfgs, [1.0, 1.0])
        @test r.fail != 0
    end

    @testset "NM strict: Rosenbrock from [-1.2,1.0]" begin
        result = optimize([-1.2, 1.0], rosenbrock)
        @test result.convergence == 0
        @test result.counts.function_ == 195
        @test result.value < 1e-6
        @test abs(result.par[1] - 1.0) < 0.01
        @test abs(result.par[2] - 1.0) < 0.01
    end

    @testset "NM strict: Beale from [0,0]" begin
        beale(x) = (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1] + x[1]*x[2]^2)^2 + (2.625 - x[1] + x[1]*x[2]^3)^2
        result = optimize([0.0, 0.0], beale)
        @test result.convergence == 0
        @test result.counts.function_ < 100
        @test result.value < 1e-6
        @test abs(result.par[1] - 3.0) < 0.01
        @test abs(result.par[2] - 0.5) < 0.01
    end

    @testset "NM strict: Booth from [0,0]" begin
        booth(x) = (x[1] + 2*x[2] - 7)^2 + (2*x[1] + x[2] - 5)^2
        result = optimize([0.0, 0.0], booth)
        @test result.convergence == 0
        @test result.counts.function_ == 75
        @test result.value < 1e-5
        @test abs(result.par[1] - 1.0) < 0.01
        @test abs(result.par[2] - 3.0) < 0.01
    end

    @testset "NM strict: maxit behavior" begin
        r10 = optimize([-1.2, 1.0], rosenbrock; control=Dict("maxit" => 10))
        @test r10.convergence == 1
        @test r10.counts.function_ == 11

        r50 = optimize([-1.2, 1.0], rosenbrock; control=Dict("maxit" => 50))
        @test r50.convergence == 1
        @test r50.counts.function_ == 51

        r200 = optimize([-1.2, 1.0], rosenbrock; control=Dict("maxit" => 200))
        @test r200.convergence == 0
        @test r200.counts.function_ == 195
    end

    @testset "Brent scalar-style callback - basic" begin
        f_scalar(x) = (x - 2)^2
        result = optimize([3.0], f_scalar; method="Brent", lower=-10.0, upper=10.0)
        @test abs(result.par[1] - 2.0) <= EPS_OPT
        @test result.value < 1e-8
    end

    @testset "Brent scalar-style callback - negative minimum" begin
        f_scalar(x) = (x + 3.0)^2
        result = optimize([0.0], f_scalar; method="Brent", lower=-10.0, upper=10.0)
        @test abs(result.par[1] - (-3.0)) <= EPS_OPT
        @test result.value < 1e-8
    end

    @testset "Brent scalar-style callback - sin" begin
        result = optimize([1.0], sin; method="Brent", lower=0.0, upper=2π)
        @test abs(result.par[1] - 3π/2) <= EPS_OPT
        @test abs(result.value - (-1.0)) < 1e-8
    end

    @testset "Brent scalar-style callback - with fnscale=-1 (maximize)" begin
        f_scalar(x) = x * (1 - x)
        result = optimize([0.1], f_scalar; method="Brent", lower=0.0, upper=1.0,
                        control=Dict("fnscale" => -1.0))
        @test abs(result.par[1] - 0.5) <= EPS_OPT
        @test abs(result.value - 0.25) < 1e-8
    end

    @testset "Brent vector-style callback still works" begin
        f_vec(x) = (x[1] - 2.0)^2
        result = optimize([3.0], f_vec; method="Brent", lower=-10.0, upper=10.0)
        @test abs(result.par[1] - 2.0) <= EPS_OPT
        @test result.value < 1e-8
    end

    @testset "Nelder-Mead (nelder_mead)" begin

        @testset "Basic optimization" begin
            x0 = [0.0, 0.0]
            opts = NelderMeadOptions()
            result = nelder_mead(rosenbrock, x0, opts)

            @test abs(result.x_opt[1] - 1.0) <= 0.1
            @test abs(result.x_opt[2] - 1.0) <= 0.1
            @test result.f_opt < 0.01
        end

        @testset "Sphere function" begin
            x0 = [5.0, 5.0, 5.0]
            opts = NelderMeadOptions()
            result = nelder_mead(sphere, x0, opts)

            @test all(abs.(result.x_opt) .<= 0.01)
            @test result.f_opt < 1e-6
        end

        @testset "Custom options" begin
            x0 = [0.0, 0.0]
            opts = NelderMeadOptions(abstol=1e-10, maxit=2000)
            result = nelder_mead(rosenbrock, x0, opts)

            @test abs(result.x_opt[1] - 1.0) <= 0.01
        end
    end

    @testset "1D Minimization (brent)" begin

        @testset "Simple quadratic" begin
            f(x) = (x - 3.0)^2
            opts = BrentOptions()
            result = brent(f, -10.0, 10.0; options=opts)

            @test abs(result.x_opt - 3.0) <= EPS_OPT
            @test result.f_opt < 1e-8
        end

        @testset "Asymmetric bounds" begin
            f(x) = (x - 5.0)^2
            opts = BrentOptions()
            result = brent(f, 0.0, 10.0; options=opts)

            @test abs(result.x_opt - 5.0) <= EPS_OPT
        end

        @testset "Minimum at boundary" begin
            f(x) = x^2
            opts = BrentOptions()
            result = brent(f, 1.0, 10.0; options=opts)

            @test abs(result.x_opt - 1.0) <= 0.01
        end
    end

    @testset "Unified optimize interface" begin

        @testset "Nelder-Mead method" begin
            x0 = [0.0, 0.0]
            result = optimize(x0, rosenbrock; method="Nelder-Mead")

            @test haskey(result, :x_opt) || haskey(result, :par)

            x_opt = haskey(result, :x_opt) ? result.x_opt : result.par
            @test abs(x_opt[1] - 1.0) <= 0.1
            @test abs(x_opt[2] - 1.0) <= 0.1
        end

        @testset "Control parameters" begin
            x0 = [0.0, 0.0]
            control = Dict("maxit" => 1000, "abstol" => 1e-8)
            result = optimize(x0, rosenbrock; method="Nelder-Mead", control=control)

            x_opt = haskey(result, :x_opt) ? result.x_opt : result.par
            @test abs(x_opt[1] - 1.0) <= 0.1
        end

        @testset "Default method" begin
            x0 = [0.0, 0.0]
            result = optimize(x0, sphere)

            x_opt = haskey(result, :x_opt) ? result.x_opt : result.par
            @test all(abs.(x_opt) .<= 0.1)
        end
    end

    @testset "numerical_hessian" begin
        x = [1.0, 1.0]
        H = numerical_hessian(rosenbrock, x)

        @test size(H) == (2, 2)
        @test isapprox(H, H', atol=0.1)
    end

    @testset "Scalers" begin

        @testset "Basic scaling" begin
            x = [1.0, 100.0, 10000.0]
            scale = [1.0, 100.0, 10000.0]
            scaled = scaler(x, scale)

            @test all(scaled .≈ 1.0)
        end

        @testset "Descaling" begin
            x_orig = [1.0, 100.0, 10000.0]
            scale = [1.0, 100.0, 10000.0]
            scaled = scaler(x_orig, scale)
            descaled = descaler(scaled, scale)

            @test all(isapprox.(descaled, x_orig, atol=EPS_OPT))
        end

        @testset "Round-trip consistency" begin
            x = randn(5) .* [1, 10, 100, 1000, 10000]
            scale = [1.0, 10.0, 100.0, 1000.0, 10000.0]
            scaled = scaler(x, scale)
            back = descaler(scaled, scale)

            @test all(isapprox.(back, x, rtol=1e-10))
        end
    end

end
