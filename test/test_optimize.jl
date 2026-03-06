using Test
using Durbyn.Optimize

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
        result = optimize(rosenbrock, x0, :bfgs; gradient=rosenbrock_grad)
        @test abs(result.minimizer[1] - 1.0) <= 0.01
        @test abs(result.minimizer[2] - 1.0) <= 0.01
        @test result.minimum < 1e-8
        @test result.converged
    end

    @testset "BFGS via optimize - Sphere with gradient" begin
        x0 = [5.0, -3.0, 2.0]
        result = optimize(sphere, x0, :bfgs; gradient=sphere_grad)
        @test all(abs.(result.minimizer) .<= 0.01)
        @test result.minimum < 1e-6
    end

    @testset "BFGS via optimize - without gradient (numerical)" begin
        x0 = [5.0, -3.0]
        result = optimize(sphere, x0, :bfgs)
        @test all(abs.(result.minimizer) .<= 0.01)
    end

    @testset "bfgs - Sphere with analytic gradient" begin
        f_bfgs(x) = sum(x .^ 2)
        function g_bfgs!(grad, x)
            grad .= 2.0 .* x
            return nothing
        end
        x0 = [5.0, -3.0]
        result = bfgs(f_bfgs, g_bfgs!, x0)
        @test all(abs.(result.x_opt) .<= 0.01)
        @test result.f_opt < 1e-6
        @test result.fail == 0
    end

    @testset "bfgs - with mask (partial optimization)" begin
        f_bfgs(x) = (x[1] - 3.0)^2 + (x[2] - 5.0)^2
        function g_bfgs!(grad, x)
            grad[1] = 2.0 * (x[1] - 3.0)
            grad[2] = 2.0 * (x[2] - 5.0)
            return nothing
        end
        x0 = [0.0, 0.0]
        mask = BitVector([true, false])
        result = bfgs(f_bfgs, g_bfgs!, x0; mask=mask)
        @test abs(result.x_opt[1] - 3.0) <= 0.01
        @test result.x_opt[2] == 0.0
    end

    @testset "L-BFGS-B via optimize - Rosenbrock with bounds" begin
        x0 = [-1.2, 1.0]
        result = optimize(rosenbrock, x0, :lbfgsb;
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
        @test abs(result.minimizer[1] - 1.0) <= 0.1
        @test abs(result.minimizer[2] - 1.0) <= 0.1
    end

    @testset "L-BFGS-B via optimize - Sphere with active lower bounds" begin
        x0 = [2.0, 2.0]
        result = optimize(sphere, x0, :lbfgsb;
                       lower=[1.0, 1.0], upper=[10.0, 10.0])
        @test abs(result.minimizer[1] - 1.0) <= 0.1
        @test abs(result.minimizer[2] - 1.0) <= 0.1
    end

    @testset "L-BFGS-B via optimize - with gradient" begin
        x0 = [5.0, -3.0]
        result = optimize(sphere, x0, :lbfgsb; gradient=sphere_grad,
                       lower=[-10.0, -10.0], upper=[10.0, 10.0])
        @test all(abs.(result.minimizer) .<= 0.01)
    end

    @testset "Brent via optimize - simple quadratic" begin
        f1d(x) = (x[1] - 2.0)^2
        result = optimize(f1d, [3.0], :brent; lower=-10.0, upper=10.0)
        @test abs(result.minimizer[1] - 2.0) <= EPS_OPT
        @test result.minimum < 1e-8
    end

    @testset "Brent via optimize - minimum at non-zero" begin
        f1d(x) = (x[1] + 3.0)^2
        result = optimize(f1d, [0.0], :brent; lower=-10.0, upper=10.0)
        @test abs(result.minimizer[1] - (-3.0)) <= EPS_OPT
    end

    @testset "Brent via optimize - minimum at boundary" begin
        f1d(x) = x[1]^2
        result = optimize(f1d, [5.0], :brent; lower=2.0, upper=10.0)
        @test abs(result.minimizer[1] - 2.0) <= 0.01
    end

    @testset "ITP root finding - sqrt(2)" begin
        f_root(x) = x^2 - 2.0
        result = itp(f_root, 0.0, 2.0)
        @test result.fail == 0
        @test abs(result.x_root - sqrt(2.0)) <= 1e-7
        @test abs(result.f_root) <= 1e-7
    end

    @testset "ITP root finding - endpoint root" begin
        f_root(x) = x - 1.0
        result = itp(f_root, 1.0, 3.0)
        @test result.fail == 0
        @test result.x_root == 1.0
        @test result.f_root == 0.0
        @test result.n_iter == 0
    end

    @testset "ITP validates bracket sign change" begin
        @test_throws ArgumentError itp(x -> x^2 + 1.0, -1.0, 1.0)
    end

    @testset "ITP handles reversed endpoint signs" begin
        result = itp(x -> 1.0 - x, 0.0, 2.0)
        @test result.fail == 0
        @test abs(result.x_root - 1.0) <= 1e-8
    end

    @testset "ITP n0=0 respects bisection iteration bound" begin
        tol = 1e-6
        opts = ITPOptions(tol=tol, n0=0, maxit=10_000)
        result = itp(x -> x, -1.0, 1.0; options=opts)
        n_half = ceil(Int, log2((2.0) / (2.0 * tol)))
        @test result.fail == 0
        @test result.n_iter <= n_half
    end

    @testset "ITP maxit fail code" begin
        opts = ITPOptions(tol=1e-14, maxit=1)
        result = itp(x -> x^3 - 2.0, 0.0, 2.0; options=opts)
        @test result.fail == 1
    end

    @testset "ITP via optimize - scalar callback" begin
        result = optimize(x -> x^2 - 2.0, [1.0], :itp; lower=0.0, upper=2.0)
        @test result.converged
        @test abs(result.minimizer[1] - sqrt(2.0)) <= 1e-7
        @test abs(result.minimum) <= 1e-7
    end

    @testset "ITP via optimize - vector callback" begin
        result = optimize(x -> x[1]^2 - 2.0, [1.0], :itp; lower=0.0, upper=2.0)
        @test result.converged
        @test abs(result.minimizer[1] - sqrt(2.0)) <= 1e-7
    end

    @testset "ITP via optimize validates bracket/sign change" begin
        @test_throws ArgumentError optimize(x -> x[1]^2 + 1.0, [0.0], :itp; lower=-1.0, upper=1.0)
    end

    @testset "ITP via optimize does not support hessian" begin
        @test_throws ArgumentError optimize(x -> x[1]^2 - 2.0, [1.0], :itp; lower=0.0, upper=2.0, hessian=true)
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

    @testset "numerical_hessian - gradient differencing path" begin
        x = [2.0, -3.0, 4.0]
        H = numerical_hessian(sphere, x; gradient=sphere_grad)
        expected_hessian = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
        @test isapprox(H, expected_hessian, atol=1e-6)
    end

    @testset "optimize(hessian=true) threads provided gradient into Hessian" begin
        result = optimize(sphere, [3.0, -2.0], :bfgs; gradient=sphere_grad, hessian=true)
        expected_hessian = [2.0 0.0; 0.0 2.0]
        @test isapprox(result.hessian, expected_hessian, atol=1e-6)
    end

    @testset "L-BFGS-B unbounded convergence (bug fix)" begin
        result = optimize(sphere, [5.0, 3.0], :lbfgsb)
        @test result.converged
        @test all(abs.(result.minimizer) .< 0.1)
        @test result.minimum < 0.01
    end

    @testset "L-BFGS-B one-sided bounds convergence (bug fix)" begin
        result = optimize(sphere, [5.0, 5.0], :lbfgsb;
                       lower=[0.0, 0.0], upper=[Inf, Inf])
        @test result.converged
        @test all(abs.(result.minimizer) .< 0.1)
    end

    @testset "L-BFGS-B bounded numerical gradient stays in bounds (bug fix)" begin
        f_sqrt(x) = sum(sqrt.(x))
        result = optimize(f_sqrt, [4.0, 4.0], :lbfgsb;
                       lower=[0.001, 0.001], upper=[10.0, 10.0])
        @test all(result.minimizer .>= 0.0)
        @test result.converged
    end

    @testset "numerical_hessian with parscale (bug fix)" begin
        f_cross(x) = x[1] * x[2]
        H = numerical_hessian(f_cross, [2.0, 3.0]; parscale=[2.0, 5.0])
        @test isapprox(H[1,2], 1.0, atol=0.01)
        @test isapprox(H[2,1], 1.0, atol=0.01)
        @test isapprox(H[1,1], 0.0, atol=0.01)
        @test isapprox(H[2,2], 0.0, atol=0.01)
    end

    @testset "Brent requires finite bounds (bug fix)" begin
        @test_throws ArgumentError optimize(x -> x[1]^2, [1.0], :brent;
                                          lower=-Inf, upper=Inf)
        @test_throws ArgumentError optimize(x -> x[1]^2, [1.0], :brent;
                                          lower=-Inf, upper=10.0)
    end

    @testset "Brent handles non-finite evaluations (bug fix)" begin
        f_nan(x) = x < 2.0 ? NaN : (x - 3.0)^2
        result = @test_warn r"NaN|Inf" brent(f_nan, 0.0, 5.0)
        @test isfinite(result.f_opt)
    end

    @testset "L-BFGS-B convergence message (bug fix)" begin
        result = optimize(sphere, [5.0, 3.0], :lbfgsb)
        @test result.message !== nothing
        @test length(result.message) > 0
    end

    @testset "L-BFGS-B infeasible bounds returns not converged (bug fix)" begin
        result = optimize(sphere, [1.0, 1.0], :lbfgsb;
                       lower=[5.0, 5.0], upper=[0.0, 0.0])
        @test !result.converged
        @test occursin("no feasible", result.message)
    end

    @testset "L-BFGS-B max_iterations convergence (bug fix)" begin
        result = optimize(rosenbrock, [-1.2, 1.0], :lbfgsb;
                       lower=[-5.0, -5.0], upper=[5.0, 5.0],
                       max_iterations=1)
        @test !result.converged
    end

    @testset "L-BFGS-B errors on non-finite fn" begin
        f_nan(x) = NaN
        @test_throws ErrorException optimize(f_nan, [1.0, 1.0], :lbfgsb)
    end

    @testset "L-BFGS-B errors on mid-iteration non-finite fn" begin
        calls = Ref(0)
        f_delayed_nan(x) = begin
            calls[] += 1
            calls[] > 2 ? NaN : sum(x .^ 2)
        end
        gr_delayed_nan(x) = 2.0 .* x
        @test_throws ErrorException optimize(f_delayed_nan, [5.0, 3.0], :lbfgsb;
                                          gradient=gr_delayed_nan)
    end

    @testset "L-BFGS-B errors on non-finite gradient" begin
        f_sphere(x) = sum(x .^ 2)
        gr_inf(x) = [Inf, Inf]
        @test_throws ErrorException optimize(f_sphere, [1.0, 1.0], :lbfgsb;
                                          gradient=gr_inf)
    end

    @testset "Brent returns converged with correct counts" begin
        f1d(x) = (x[1] - 2.0)^2
        result = optimize(f1d, [3.0], :brent; lower=-10.0, upper=10.0)
        @test result.converged
        @test result.f_calls >= 1
        @test result.g_calls == 0
    end

    @testset "Bounds length recycling" begin
        result = optimize(sphere, [5.0, 5.0, 5.0], :lbfgsb;
                       lower=[0.0], upper=[10.0, 10.0, 10.0])
        @test length(result.minimizer) == 3
        @test all(result.minimizer .>= 0.0)

        result2 = optimize(sphere, [5.0, 5.0], :lbfgsb;
                        lower=[-10.0, -10.0, -10.0], upper=[10.0, 10.0, 10.0])
        @test length(result2.minimizer) == 2

        result3 = optimize(sphere, [5.0, 5.0], :lbfgsb;
                        lower=Float64[], upper=[10.0, 10.0])
        @test result3.converged
        @test all(abs.(result3.minimizer) .< 0.1)

        @test_throws ArgumentError optimize(x -> x[1]^2, [0.0], :brent;
                       lower=Float64[], upper=[5.0])
    end

    @testset "param_scale/step_sizes wrong length errors" begin
        @test_throws ArgumentError optimize(sphere, [5.0, 5.0], :bfgs;
                       param_scale=[1.0])
        @test_throws ArgumentError optimize(sphere, [5.0, 5.0], :bfgs;
                       step_sizes=[1e-4])

        result = optimize(sphere, [5.0, 5.0], :bfgs;
                       param_scale=[2.0, 2.0])
        @test result.converged
    end

    @testset "L-BFGS-B catch does not swallow user errors (bug fix)" begin
        f_user_error(x) = error("finite values in custom code")
        @test_throws ErrorException optimize(f_user_error, [1.0, 1.0], :lbfgsb)
    end

    @testset "Integer par and bounds accepted" begin
        result = optimize(sphere, [5, 3], :bfgs)
        @test all(abs.(result.minimizer) .< 0.1)

        result2 = optimize(sphere, [5, 5], :lbfgsb;
                        lower=[0, 0], upper=[10, 10])
        @test all(result2.minimizer .>= 0.0)
    end

    @testset "Brent errors when lower >= upper" begin
        @test_throws ArgumentError optimize(x -> x[1]^2, [0.0], :brent;
                       lower=5.0, upper=1.0)
        @test_throws ArgumentError optimize(x -> x[1]^2, [0.0], :brent;
                       lower=3.0, upper=3.0)
    end

    @testset "Gradient length validated" begin
        bad_gr(x) = [1.0]
        @test_throws ArgumentError optimize(sphere, [1.0, 1.0], :bfgs;
                       gradient=bad_gr)
        @test_throws ArgumentError optimize(sphere, [1.0, 1.0], :lbfgsb;
                       gradient=bad_gr,
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
    end

    @testset "L-BFGS-B zero-parameter returns NOTHING TO DO" begin
        f_empty(x) = 42.0
        result = optimize(f_empty, Float64[], :lbfgsb)
        @test result.converged
        @test result.minimum == 42.0
        @test result.f_calls == 1
        @test result.g_calls == 0
        @test result.message == "NOTHING TO DO"
        @test isempty(result.minimizer)
    end

    @testset "Objective returning non-scalar errors clearly" begin
        fn_vec(x) = [x[1]^2, x[1]]
        @test_throws ArgumentError optimize(fn_vec, [1.0, 1.0])
        @test_throws ArgumentError optimize(fn_vec, [1.0, 1.0], :bfgs)
        @test_throws ArgumentError optimize(fn_vec, [1.0, 1.0], :lbfgsb;
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
    end

    @testset "Direct brent errors when lower >= upper" begin
        f(x) = (x - 2.0)^2
        @test_throws ArgumentError brent(f, 5.0, -5.0)
        @test_throws ArgumentError brent(f, 3.0, 3.0)
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
        @test_throws ErrorException optimize(f_delayed_nan2, [5.0, 3.0], :lbfgsb;
                                          gradient=gr_for_count)
    end

    @testset "Length-1 vector objective accepted" begin
        fn_vec1(x) = [sum(x .^ 2)]
        result = optimize(fn_vec1, [5.0, 3.0])
        @test result.minimum < 0.1

        result2 = optimize(fn_vec1, [5.0, 3.0], :bfgs)
        @test result2.minimum < 0.01

        result3 = optimize(fn_vec1, [5.0, 3.0], :lbfgsb;
                        lower=[-10.0, -10.0], upper=[10.0, 10.0])
        @test result3.minimum < 0.1

        fn_vec2(x) = [x[1]^2, x[1]]
        @test_throws ArgumentError optimize(fn_vec2, [1.0, 1.0])
    end

    @testset "Brent honors max_iterations" begin
        f1d(x) = (x[1] - 2.0)^2
        result_small = optimize(f1d, [0.0], :brent; lower=-10.0, upper=10.0,
                             max_iterations=1)
        result_big = optimize(f1d, [0.0], :brent; lower=-10.0, upper=10.0,
                           max_iterations=10000)
        @test result_big.converged
        @test abs(result_big.minimizer[1] - 2.0) < 0.01
        # With max_iterations=1, Brent barely iterates
        @test result_big.iterations >= result_small.iterations
    end

    @testset "Direct nelder_mead with negative objective (abstol fix)" begin
        f_neg(x) = (x[1] - 1.0)^2 + (x[2] - 1.0)^2 - 10.0
        opts = NelderMeadOptions()
        result = nelder_mead(f_neg, [0.0, 0.0], opts)
        @test result.f_opt < -9.99
    end

    @testset "step_sizes wrong length only errors for gradient methods" begin
        f1d(x) = (x[1] - 2.0)^2
        @test_throws ArgumentError optimize(sphere, [5.0, 5.0], :bfgs;
                       step_sizes=[1e-4])
    end

    @testset "1D Nelder-Mead warning" begin
        @test_warn "Nelder-Mead is unreliable" optimize(sphere, [5.0])
    end

    @testset "fn returning nothing/missing gives clear error" begin
        @test_throws ArgumentError optimize(x -> nothing, [1.0, 1.0])
        @test_throws ArgumentError optimize(x -> missing, [1.0, 1.0])
    end

    @testset "grad returning nothing gives clear error" begin
        @test_throws ArgumentError optimize(sphere, [1.0, 1.0], :bfgs;
                       gradient=x -> nothing)
    end

    @testset "L-BFGS-B converges with numerical gradient (aliasing fix)" begin
        result = optimize(rosenbrock, [-1.2, 1.0], :lbfgsb;
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
        @test result.converged
        @test result.minimum < 1e-4
        @test abs(result.minimizer[1] - 1.0) < 0.01
        @test abs(result.minimizer[2] - 1.0) < 0.01

        result2 = optimize(rosenbrock, [-1.2, 1.0], :lbfgsb)
        @test result2.converged
        @test result2.minimum < 1e-4
    end

    @testset "L-BFGS-B Beale from [4,4] bounded (GCP fix)" begin
        beale(x) = (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1] + x[1]*x[2]^2)^2 +
                   (2.625 - x[1] + x[1]*x[2]^3)^2
        r = optimize(beale, [4.0, 4.0], :lbfgsb;
                  lower=[-4.5, -4.5], upper=[4.5, 4.5])
        @test r.converged
        @test r.minimum < 1e-6
        @test abs(r.minimizer[1] - 3.0) < 0.01
        @test abs(r.minimizer[2] - 0.5) < 0.01
    end

    @testset "L-BFGS-B Rosenbrock from [0,0] bounded (GCP fix)" begin
        r = optimize(rosenbrock, [0.0, 0.0], :lbfgsb;
                  lower=[-5.0, -5.0], upper=[5.0, 5.0])
        @test r.minimum < 1e-4
        @test abs(r.minimizer[1] - 1.0) < 0.01
        @test abs(r.minimizer[2] - 1.0) < 0.01
    end

    @testset "L-BFGS-B unbounded Rosenbrock (GCP fix)" begin
        r = optimize(rosenbrock, [-1.2, 1.0], :lbfgsb)
        @test r.converged
        @test r.minimum < 1e-4
        @test abs(r.minimizer[1] - 1.0) < 0.01
        @test abs(r.minimizer[2] - 1.0) < 0.01
    end

    @testset "lbfgsb NaN gradient does not falsely converge (bug fix)" begin
        using Durbyn.Optimize: lbfgsb
        f_ok(x) = sum(x .^ 2)
        g_nan(x) = [NaN, NaN]
        r = lbfgsb(f_ok, g_nan, [1.0, 1.0])
        @test r.fail != 0
        @test occursin("non-finite", r.message)
    end

    @testset "lbfgsb Inf gradient returns non-finite error (bug fix)" begin
        using Durbyn.Optimize: lbfgsb
        f_ok(x) = sum(x .^ 2)
        g_inf(x) = [Inf, Inf]
        r = lbfgsb(f_ok, g_inf, [1.0, 1.0])
        @test r.fail != 0
        @test occursin("non-finite", r.message)
    end

    @testset "bfgs NaN gradient does not falsely converge (bug fix)" begin
        using Durbyn.Optimize: bfgs
        f_bfgs(x) = sum(x .^ 2)
        g_nan_bfgs(gvec, x) = (gvec .= NaN; nothing)
        r = bfgs(f_bfgs, g_nan_bfgs, [1.0, 1.0])
        @test r.fail != 0
    end

    @testset "bfgs Inf gradient does not falsely converge (bug fix)" begin
        using Durbyn.Optimize: bfgs
        f_bfgs2(x) = sum(x .^ 2)
        g_inf_bfgs(gvec, x) = (gvec .= Inf; nothing)
        r = bfgs(f_bfgs2, g_inf_bfgs, [1.0, 1.0])
        @test r.fail != 0
    end

    @testset "NM strict: Rosenbrock from [-1.2,1.0]" begin
        result = optimize(rosenbrock, [-1.2, 1.0])
        @test result.converged
        @test result.f_calls > 0
        @test result.minimum < 1e-6
        @test abs(result.minimizer[1] - 1.0) < 0.01
        @test abs(result.minimizer[2] - 1.0) < 0.01
    end

    @testset "NM strict: Beale from [0,0]" begin
        beale(x) = (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1] + x[1]*x[2]^2)^2 + (2.625 - x[1] + x[1]*x[2]^3)^2
        result = optimize(beale, [0.0, 0.0])
        @test result.converged
        @test result.f_calls < 100
        @test result.minimum < 1e-6
        @test abs(result.minimizer[1] - 3.0) < 0.01
        @test abs(result.minimizer[2] - 0.5) < 0.01
    end

    @testset "NM strict: Booth from [0,0]" begin
        booth(x) = (x[1] + 2*x[2] - 7)^2 + (2*x[1] + x[2] - 5)^2
        result = optimize(booth, [0.0, 0.0])
        @test result.converged
        @test result.f_calls > 0
        @test result.minimum < 1e-5
        @test abs(result.minimizer[1] - 1.0) < 0.01
        @test abs(result.minimizer[2] - 3.0) < 0.01
    end

    @testset "NM strict: max_iterations behavior" begin
        r10 = optimize(rosenbrock, [-1.2, 1.0]; max_iterations=10)
        @test !r10.converged
        @test r10.f_calls > 0

        r50 = optimize(rosenbrock, [-1.2, 1.0]; max_iterations=50)
        @test !r50.converged
        @test r50.f_calls >= r10.f_calls

        r200 = optimize(rosenbrock, [-1.2, 1.0]; max_iterations=200)
        @test r200.converged
        @test r200.f_calls >= r50.f_calls
    end

    @testset "Brent scalar-style callback - basic" begin
        f_scalar(x) = (x - 2)^2
        result = optimize(f_scalar, [3.0], :brent; lower=-10.0, upper=10.0)
        @test abs(result.minimizer[1] - 2.0) <= EPS_OPT
        @test result.minimum < 1e-8
    end

    @testset "Brent scalar-style callback - negative minimum" begin
        f_scalar(x) = (x + 3.0)^2
        result = optimize(f_scalar, [0.0], :brent; lower=-10.0, upper=10.0)
        @test abs(result.minimizer[1] - (-3.0)) <= EPS_OPT
        @test result.minimum < 1e-8
    end

    @testset "Brent scalar-style callback - sin" begin
        result = optimize(sin, [1.0], :brent; lower=0.0, upper=2π)
        @test abs(result.minimizer[1] - 3π/2) <= EPS_OPT
        @test abs(result.minimum - (-1.0)) < 1e-8
    end

    @testset "Brent scalar-style callback - with fn_scale=-1 (maximize)" begin
        f_scalar(x) = x * (1 - x)
        result = optimize(f_scalar, [0.1], :brent; lower=0.0, upper=1.0,
                        fn_scale=-1.0)
        @test abs(result.minimizer[1] - 0.5) <= EPS_OPT
        @test abs(result.minimum - 0.25) < 1e-8
    end

    @testset "Brent vector-style callback still works" begin
        f_vec(x) = (x[1] - 2.0)^2
        result = optimize(f_vec, [3.0], :brent; lower=-10.0, upper=10.0)
        @test abs(result.minimizer[1] - 2.0) <= EPS_OPT
        @test result.minimum < 1e-8
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
            result = optimize(rosenbrock, x0, :nelder_mead)

            @test abs(result.minimizer[1] - 1.0) <= 0.1
            @test abs(result.minimizer[2] - 1.0) <= 0.1
        end

        @testset "With keyword arguments" begin
            x0 = [0.0, 0.0]
            result = optimize(rosenbrock, x0, :nelder_mead;
                            max_iterations=1000, abstol=1e-8)

            @test abs(result.minimizer[1] - 1.0) <= 0.1
        end

        @testset "Default method" begin
            x0 = [0.0, 0.0]
            result = optimize(sphere, x0)

            @test all(abs.(result.minimizer) .<= 0.1)
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
