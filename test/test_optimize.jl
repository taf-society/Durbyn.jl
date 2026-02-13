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

    # ── BFGS via optim() ────────────────────────────────────────────────────
    @testset "BFGS via optim - Rosenbrock with analytic gradient" begin
        x0 = [-1.2, 1.0]
        result = optim(x0, rosenbrock; gr=rosenbrock_grad, method="BFGS")
        @test abs(result.par[1] - 1.0) <= 0.01
        @test abs(result.par[2] - 1.0) <= 0.01
        @test result.value < 1e-8
        @test result.convergence == 0
    end

    @testset "BFGS via optim - Sphere with gradient" begin
        x0 = [5.0, -3.0, 2.0]
        result = optim(x0, sphere; gr=sphere_grad, method="BFGS")
        @test all(abs.(result.par) .<= 0.01)
        @test result.value < 1e-6
    end

    @testset "BFGS via optim - without gradient (numerical)" begin
        x0 = [5.0, -3.0]
        result = optim(x0, sphere; method="BFGS")
        @test all(abs.(result.par) .<= 0.01)
    end

    # ── BFGS direct (bfgsmin) ───────────────────────────────────────────────
    @testset "bfgsmin - Sphere with analytic gradient" begin
        # bfgsmin expects f(n, x, ex) and g(n, x, grad, ex) signatures
        f_bfgs(n, x, ex) = sum(x .^ 2)
        function g_bfgs(n, x, grad, ex)
            grad .= 2.0 .* x
            return nothing
        end
        x0 = [5.0, -3.0]
        result = bfgsmin(f_bfgs, g_bfgs, x0)
        @test all(abs.(result.x_opt) .<= 0.01)
        @test result.f_opt < 1e-6
        @test result.fail == 0
    end

    @testset "bfgsmin - with mask (partial optimization)" begin
        f_bfgs(n, x, ex) = (x[1] - 3.0)^2 + (x[2] - 5.0)^2
        function g_bfgs(n, x, grad, ex)
            grad[1] = 2.0 * (x[1] - 3.0)
            grad[2] = 2.0 * (x[2] - 5.0)
            return nothing
        end
        x0 = [0.0, 0.0]
        mask = BitVector([true, false])  # only optimize first param
        result = bfgsmin(f_bfgs, g_bfgs, x0; mask=mask)
        @test abs(result.x_opt[1] - 3.0) <= 0.01
        @test result.x_opt[2] == 0.0  # second param should not move
    end

    # ── L-BFGS-B via optim() ────────────────────────────────────────────────
    @testset "L-BFGS-B via optim - Rosenbrock with bounds" begin
        x0 = [-1.2, 1.0]
        result = optim(x0, rosenbrock; method="L-BFGS-B",
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
        @test abs(result.par[1] - 1.0) <= 0.1
        @test abs(result.par[2] - 1.0) <= 0.1
    end

    @testset "L-BFGS-B via optim - Sphere with active lower bounds" begin
        x0 = [2.0, 2.0]
        result = optim(x0, sphere; method="L-BFGS-B",
                       lower=[1.0, 1.0], upper=[10.0, 10.0])
        # Optimum at lower bounds since unconstrained optimum is (0,0)
        @test abs(result.par[1] - 1.0) <= 0.1
        @test abs(result.par[2] - 1.0) <= 0.1
    end

    @testset "L-BFGS-B via optim - with gradient" begin
        x0 = [5.0, -3.0]
        result = optim(x0, sphere; gr=sphere_grad, method="L-BFGS-B",
                       lower=[-10.0, -10.0], upper=[10.0, 10.0])
        @test all(abs.(result.par) .<= 0.01)
    end

    # ── Brent via optim() ───────────────────────────────────────────────────
    @testset "Brent via optim - simple quadratic" begin
        f1d(x) = (x[1] - 2.0)^2
        result = optim([3.0], f1d; method="Brent", lower=-10.0, upper=10.0)
        @test abs(result.par[1] - 2.0) <= EPS_OPT
        @test result.value < 1e-8
    end

    @testset "Brent via optim - minimum at non-zero" begin
        f1d(x) = (x[1] + 3.0)^2
        result = optim([0.0], f1d; method="Brent", lower=-10.0, upper=10.0)
        @test abs(result.par[1] - (-3.0)) <= EPS_OPT
    end

    @testset "Brent via optim - minimum at boundary" begin
        f1d(x) = x[1]^2
        result = optim([5.0], f1d; method="Brent", lower=2.0, upper=10.0)
        @test abs(result.par[1] - 2.0) <= 0.01
    end

    # ── Numerical gradient (numgrad) ────────────────────────────────────────
    @testset "numgrad - Sphere gradient" begin
        # numgrad expects f(n, x, ex) signature
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

    # ── optim_hessian expanded ──────────────────────────────────────────────
    @testset "optim_hessian - Sphere at origin" begin
        x = [0.0, 0.0]
        H = optim_hessian(sphere, x)
        @test isapprox(H, 2.0 * [1.0 0.0; 0.0 1.0], atol=0.1)
    end

    @testset "optim_hessian - Rosenbrock at (1,1)" begin
        x = [1.0, 1.0]
        H = optim_hessian(rosenbrock, x)
        # Analytic Hessian at (1,1): [[802, -400], [-400, 200]]
        @test isapprox(H[1,1], 802.0, rtol=0.05)
        @test isapprox(H[1,2], -400.0, rtol=0.05)
        @test isapprox(H[2,1], -400.0, rtol=0.05)
        @test isapprox(H[2,2], 200.0, rtol=0.05)
    end

    # ── Bug fix regression tests ──────────────────────────────────────────
    @testset "L-BFGS-B unbounded convergence (bug fix)" begin
        # Previously: line search rejected alpha_max=Inf, returning unchanged params
        result = optim([5.0, 3.0], sphere; method="L-BFGS-B")
        @test result.convergence == 0
        @test all(abs.(result.par) .< 0.1)
        @test result.value < 0.01
    end

    @testset "L-BFGS-B one-sided bounds convergence (bug fix)" begin
        # lower finite, upper=Inf
        result = optim([5.0, 5.0], sphere; method="L-BFGS-B",
                       lower=[0.0, 0.0], upper=[Inf, Inf])
        @test result.convergence == 0
        @test all(abs.(result.par) .< 0.1)
    end

    @testset "L-BFGS-B bounded numgrad stays in bounds (bug fix)" begin
        # Previously: unbounded numgrad! could evaluate outside feasible region
        f_sqrt(x) = sum(sqrt.(x))
        result = optim([4.0, 4.0], f_sqrt; method="L-BFGS-B",
                       lower=[0.001, 0.001], upper=[10.0, 10.0])
        @test all(result.par .>= 0.0)
        @test result.convergence == 0
    end

    @testset "optim_hessian with parscale (bug fix)" begin
        # Previously: mixed scaled/unscaled coordinates corrupted mixed partials
        f_cross(x) = x[1] * x[2]
        H = optim_hessian(f_cross, [2.0, 3.0]; parscale=[2.0, 5.0])
        @test isapprox(H[1,2], 1.0, atol=0.01)
        @test isapprox(H[2,1], 1.0, atol=0.01)
        @test isapprox(H[1,1], 0.0, atol=0.01)
        @test isapprox(H[2,2], 0.0, atol=0.01)
    end

    @testset "Unknown control key warning (bug fix)" begin
        # Previously: warning never triggered due to merge! before setdiff
        @test_warn "unknown names in control" optim([1.0, 1.0], sphere;
                                                     control=Dict("bogus" => 42))
    end

    @testset "Brent requires finite bounds (bug fix)" begin
        @test_throws ErrorException optim([1.0], x -> x[1]^2;
                                          method="Brent", lower=-Inf, upper=Inf)
        @test_throws ErrorException optim([1.0], x -> x[1]^2;
                                          method="Brent", lower=-Inf, upper=10.0)
    end

    @testset "Brent handles non-finite evaluations (bug fix)" begin
        # Function that returns NaN at the initial evaluation point.
        # Previously fmin would hard-error; now it substitutes like R's optimize().
        f_nan(x) = x < 2.0 ? NaN : (x - 3.0)^2
        # Initial point for [0,5] is at ~1.91 which returns NaN
        result = @test_warn r"NaN|Inf" fmin(f_nan, 0.0, 5.0)
        @test isfinite(result.f_opt)
    end

    @testset "L-BFGS-B convergence message (bug fix)" begin
        # Previously: always returned message=nothing
        result = optim([5.0, 3.0], sphere; method="L-BFGS-B")
        @test result.message !== nothing
        @test occursin("CONVERGENCE", result.message)
    end

    @testset "L-BFGS-B infeasible bounds returns code 52 (bug fix)" begin
        # R returns convergence=52, message="ERROR: NO FEASIBLE SOLUTION"
        result = optim([1.0, 1.0], sphere; method="L-BFGS-B",
                       lower=[5.0, 5.0], upper=[0.0, 0.0])  # lower > upper
        @test result.convergence == 52
        @test occursin("NO FEASIBLE SOLUTION", result.message)
    end

    @testset "L-BFGS-B maxit convergence code (bug fix)" begin
        # Force maxit with very few iterations
        result = optim([-1.2, 1.0], rosenbrock; method="L-BFGS-B",
                       lower=[-5.0, -5.0], upper=[5.0, 5.0],
                       control=Dict("maxit" => 1))
        @test result.convergence == 1
    end

    # ── Existing tests (preserved) ──────────────────────────────────────────
    @testset "Nelder-Mead (nmmin)" begin

        @testset "Basic optimization" begin
            x0 = [0.0, 0.0]
            opts = NelderMeadOptions()
            result = nmmin(rosenbrock, x0, opts)

            @test abs(result.x_opt[1] - 1.0) <= 0.1
            @test abs(result.x_opt[2] - 1.0) <= 0.1
            @test result.f_opt < 0.01
        end

        @testset "Sphere function" begin
            x0 = [5.0, 5.0, 5.0]
            opts = NelderMeadOptions()
            result = nmmin(sphere, x0, opts)

            @test all(abs.(result.x_opt) .<= 0.01)
            @test result.f_opt < 1e-6
        end

        @testset "Custom options" begin
            x0 = [0.0, 0.0]
            opts = NelderMeadOptions(abstol=1e-10, maxit=2000)
            result = nmmin(rosenbrock, x0, opts)

            @test abs(result.x_opt[1] - 1.0) <= 0.01
        end
    end

    @testset "1D Minimization (fmin)" begin

        @testset "Simple quadratic" begin
            f(x) = (x - 3.0)^2
            opts = FminOptions()
            result = fmin(f, -10.0, 10.0; options=opts)

            @test abs(result.x_opt - 3.0) <= EPS_OPT
            @test result.f_opt < 1e-8
        end

        @testset "Asymmetric bounds" begin
            f(x) = (x - 5.0)^2
            opts = FminOptions()
            result = fmin(f, 0.0, 10.0; options=opts)

            @test abs(result.x_opt - 5.0) <= EPS_OPT
        end

        @testset "Minimum at boundary" begin
            f(x) = x^2
            opts = FminOptions()
            result = fmin(f, 1.0, 10.0; options=opts)

            @test abs(result.x_opt - 1.0) <= 0.01
        end
    end

    @testset "Unified optim interface" begin

        @testset "Nelder-Mead method" begin
            x0 = [0.0, 0.0]
            result = optim(x0, rosenbrock; method="Nelder-Mead")

            @test haskey(result, :x_opt) || haskey(result, :par)

            x_opt = haskey(result, :x_opt) ? result.x_opt : result.par
            @test abs(x_opt[1] - 1.0) <= 0.1
            @test abs(x_opt[2] - 1.0) <= 0.1
        end

        @testset "Control parameters" begin
            x0 = [0.0, 0.0]
            control = Dict("maxit" => 1000, "abstol" => 1e-8)
            result = optim(x0, rosenbrock; method="Nelder-Mead", control=control)

            x_opt = haskey(result, :x_opt) ? result.x_opt : result.par
            @test abs(x_opt[1] - 1.0) <= 0.1
        end

        @testset "Default method" begin
            x0 = [0.0, 0.0]
            result = optim(x0, sphere)

            x_opt = haskey(result, :x_opt) ? result.x_opt : result.par
            @test all(abs.(x_opt) .<= 0.1)
        end
    end

    @testset "optim_hessian" begin
        x = [1.0, 1.0]
        H = optim_hessian(rosenbrock, x)

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
