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

    @testset "L-BFGS-B errors on non-finite fn (R compat)" begin
        # R throws: "L-BFGS-B needs finite values of 'fn'"
        f_nan(x) = NaN
        @test_throws ErrorException optim([1.0, 1.0], f_nan; method="L-BFGS-B")
    end

    @testset "L-BFGS-B errors on mid-iteration non-finite fn (R compat)" begin
        # R throws when fn returns NaN mid-optimization
        calls = Ref(0)
        f_delayed_nan(x) = begin
            calls[] += 1
            calls[] > 2 ? NaN : sum(x .^ 2)
        end
        gr_delayed_nan(x) = 2.0 .* x
        @test_throws ErrorException optim([5.0, 3.0], f_delayed_nan; gr=gr_delayed_nan,
                                          method="L-BFGS-B")
    end

    @testset "L-BFGS-B errors on non-finite gradient (R compat)" begin
        # R throws: "non-finite value supplied by optim"
        f_sphere(x) = sum(x .^ 2)
        gr_inf(x) = [Inf, Inf]
        @test_throws ErrorException optim([1.0, 1.0], f_sphere; gr=gr_inf,
                                          method="L-BFGS-B")
    end

    @testset "Brent returns convergence=0 and nothing counts (R compat)" begin
        # R's optim() Brent path always returns convergence=0 and counts=NA
        f1d(x) = (x[1] - 2.0)^2
        result = optim([3.0], f1d; method="Brent", lower=-10.0, upper=10.0)
        @test result.convergence == 0
        @test result.counts.function_ === nothing
        @test result.counts.gradient === nothing
    end

    @testset "Brent convergence=0 even with small maxit (R compat)" begin
        # R always returns 0 for Brent; Julia used to return fail=1
        f1d(x) = (x[1] - 2.0)^2
        result = optim([3.0], f1d; method="Brent", lower=-10.0, upper=10.0,
                       control=Dict("maxit" => 2))
        @test result.convergence == 0
    end

    @testset "Bounds length recycling (R rep_len compat)" begin
        # Scalar lower with vector upper should work (scalar already handled)
        # Mismatched vector lengths: short vector recycled to npar
        result = optim([5.0, 5.0, 5.0], sphere; method="L-BFGS-B",
                       lower=[0.0], upper=[10.0, 10.0, 10.0])
        @test length(result.par) == 3
        @test all(result.par .>= 0.0)

        # Long vector truncated to npar
        result2 = optim([5.0, 5.0], sphere; method="L-BFGS-B",
                        lower=[-10.0, -10.0, -10.0], upper=[10.0, 10.0, 10.0])
        @test length(result2.par) == 2

        # Empty vector bounds → treated as unbounded (R: rep_len(numeric(0),n) → NA → unbounded)
        result3 = optim([5.0, 5.0], sphere; method="L-BFGS-B",
                        lower=Float64[], upper=[10.0, 10.0])
        @test result3.convergence == 0
        @test all(abs.(result3.par) .< 0.1)

        # Empty bounds for Brent → error (NaN bounds are not finite)
        @test_throws ErrorException optim([0.0], x -> x[1]^2;
                       method="Brent", lower=Float64[], upper=[5.0])
    end

    @testset "parscale/ndeps wrong length errors (R compat)" begin
        # R's optim.c errors: "'parscale' is of the wrong length"
        @test_throws ErrorException optim([5.0, 5.0], sphere; method="BFGS",
                       control=Dict("parscale" => [1.0]))
        @test_throws ErrorException optim([5.0, 5.0], sphere; method="BFGS",
                       control=Dict("ndeps" => [1e-4]))

        # Scalar parscale/ndeps should be broadcast (not error)
        result = optim([5.0, 5.0], sphere; method="BFGS",
                       control=Dict("parscale" => 2.0))
        @test result.convergence == 0
        result2 = optim([5.0, 5.0], sphere; method="BFGS",
                        control=Dict("ndeps" => 1e-4))
        @test result2.convergence == 0
    end

    @testset "L-BFGS-B catch does not swallow user errors (bug fix)" begin
        # A user fn that errors with a message containing "finite values" should
        # NOT be silently caught — only the exact optimizer message is caught.
        f_user_error(x) = error("finite values in custom code")
        @test_throws ErrorException optim([1.0, 1.0], f_user_error; method="L-BFGS-B")
    end

    @testset "Integer par and bounds accepted (R as.double compat)" begin
        # R silently coerces: optim(c(1L,2L), ...) works
        result = optim([5, 3], sphere; method="BFGS")
        @test all(abs.(result.par) .< 0.1)

        result2 = optim([5, 5], sphere; method="L-BFGS-B",
                        lower=[0, 0], upper=[10, 10])
        @test all(result2.par .>= 0.0)
    end

    @testset "Brent errors when lower >= upper (R compat)" begin
        # R's optimize.c:267 checks: xmin >= xmax → error
        @test_throws ErrorException optim([0.0], x -> x[1]^2;
                       method="Brent", lower=5.0, upper=1.0)
        # Equal bounds also invalid
        @test_throws ErrorException optim([0.0], x -> x[1]^2;
                       method="Brent", lower=3.0, upper=3.0)
    end

    @testset "Gradient length validated (R compat)" begin
        # R's optim.c:109-111: "gradient in optim evaluated to length X not Y"
        bad_gr(x) = [1.0]  # returns length 1 for a 2-param problem
        @test_throws ErrorException optim([1.0, 1.0], sphere;
                       gr=bad_gr, method="BFGS")
        @test_throws ErrorException optim([1.0, 1.0], sphere;
                       gr=bad_gr, method="L-BFGS-B",
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
    end

    @testset "L-BFGS-B zero-parameter returns NOTHING TO DO (R compat)" begin
        # R's optim.c:653-659 special-cases n=0
        f_empty(x) = 42.0
        result = optim(Float64[], f_empty; method="L-BFGS-B")
        @test result.convergence == 0
        @test result.value == 42.0
        @test result.counts.function_ == 1
        @test result.counts.gradient == 0
        @test result.message == "NOTHING TO DO"
        @test isempty(result.par)
    end

    @testset "Objective returning non-scalar errors clearly (R compat)" begin
        # R's optim.c:82-84: "objective function in optim evaluates to length N not 1"
        fn_vec(x) = [x[1]^2, x[1]]
        @test_throws ErrorException optim([1.0, 1.0], fn_vec)
        @test_throws ErrorException optim([1.0, 1.0], fn_vec; method="BFGS")
        @test_throws ErrorException optim([1.0, 1.0], fn_vec; method="L-BFGS-B",
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
    end

    @testset "Direct fmin errors when lower >= upper (R compat)" begin
        # R's optimize.c:267: xmin >= xmax → error
        f(x) = (x - 2.0)^2
        @test_throws ErrorException fmin(f, 5.0, -5.0)
        @test_throws ErrorException fmin(f, 3.0, 3.0)
        # Valid interval still works
        result = fmin(f, 0.0, 5.0)
        @test abs(result.x_opt - 2.0) < 0.01
    end

    @testset "L-BFGS-B error on delayed NaN fn throws (R compat)" begin
        # fn that becomes NaN after a few calls — R throws an error
        calls = Ref(0)
        f_delayed_nan2(x) = begin
            calls[] += 1
            calls[] > 2 ? NaN : sum(x .^ 2)
        end
        gr_for_count(x) = 2.0 .* x
        @test_throws ErrorException optim([5.0, 3.0], f_delayed_nan2; gr=gr_for_count,
                                          method="L-BFGS-B")
    end

    @testset "Length-1 vector objective accepted (R compat)" begin
        # R's optim.c:80-81 coerces to REALSXP then checks LENGTH==1.
        # A length-1 vector passes in R; only length>1 errors.
        fn_vec1(x) = [sum(x .^ 2)]  # returns length-1 vector
        result = optim([5.0, 3.0], fn_vec1)
        @test result.value < 0.1

        result2 = optim([5.0, 3.0], fn_vec1; method="BFGS")
        @test result2.value < 0.01

        result3 = optim([5.0, 3.0], fn_vec1; method="L-BFGS-B",
                        lower=[-10.0, -10.0], upper=[10.0, 10.0])
        @test result3.value < 0.1

        # Length-2 vector still errors
        fn_vec2(x) = [x[1]^2, x[1]]
        @test_throws ErrorException optim([1.0, 1.0], fn_vec2)
    end

    @testset "Brent ignores maxit from control (R compat)" begin
        # R's optimize() has no maxit parameter — Brent runs until tol convergence.
        f1d(x) = (x[1] - 2.0)^2
        result_small = optim([0.0], f1d; method="Brent", lower=-10.0, upper=10.0,
                             control=Dict("maxit" => 1))
        result_big = optim([0.0], f1d; method="Brent", lower=-10.0, upper=10.0,
                           control=Dict("maxit" => 10000))
        # Both should find the same minimum regardless of maxit
        @test abs(result_small.par[1] - 2.0) < 0.01
        @test abs(result_big.par[1] - 2.0) < 0.01
    end

    @testset "Direct nmmin with negative objective (abstol fix)" begin
        # nmmin default abstol was 0.0, causing early stop for negative objectives.
        # R's default abstol=-Inf (from optim.R:37).
        f_neg(x) = (x[1] - 1.0)^2 + (x[2] - 1.0)^2 - 10.0
        opts = NelderMeadOptions()  # should now default to abstol=-Inf
        result = nmmin(f_neg, [0.0, 0.0], opts)
        @test result.f_opt < -9.99  # should reach near -10.0
    end

    @testset "ndeps wrong length only errors for gradient methods (R compat)" begin
        # R only validates ndeps for BFGS/L-BFGS-B when gr=NULL.
        # NM and Brent should ignore wrong-length ndeps.
        result_nm = optim([5.0, 5.0], sphere;
                          control=Dict("ndeps" => [1e-3]))
        @test result_nm.convergence == 0 || result_nm.value < 0.1

        f1d(x) = (x[1] - 2.0)^2
        result_brent = optim([0.0], f1d; method="Brent", lower=-10.0, upper=10.0,
                             control=Dict("ndeps" => [1e-3, 1e-3, 1e-3]))
        @test abs(result_brent.par[1] - 2.0) < 0.01

        # But BFGS without gr still errors on wrong ndeps
        @test_throws ErrorException optim([5.0, 5.0], sphere; method="BFGS",
                       control=Dict("ndeps" => [1e-4]))
    end

    @testset "Brent ignores parscale (R compat)" begin
        # R's Brent path bypasses C_optim, so parscale is never used/validated.
        f1d(x) = (x[1] - 2.0)^2
        # Wrong-length parscale should NOT error for Brent
        result = optim([0.0], f1d; method="Brent", lower=-10.0, upper=10.0,
                       control=Dict("parscale" => [1.0, 2.0, 3.0]))
        @test abs(result.par[1] - 2.0) < 0.01
    end

    @testset "NM gradient count is nothing (R NA compat)" begin
        # R's optim.c:276 sets grcount=NA_INTEGER for NM
        result = optim([5.0, 5.0], sphere)
        @test result.counts.gradient === nothing
    end

    @testset "warn.1d.NelderMead control suppresses warning (R compat)" begin
        # Default: warning fires for 1D Nelder-Mead
        @test_warn "Nelder-Mead is unreliable" optim([5.0], sphere)

        # R's key is "warn.1d.NelderMead"
        @test_nowarn optim([5.0], sphere; control=Dict("warn.1d.NelderMead" => false))
    end

    @testset "fn returning nothing/missing gives clear error" begin
        @test_throws ErrorException optim([1.0, 1.0], x -> nothing)
        @test_throws ErrorException optim([1.0, 1.0], x -> missing)
    end

    @testset "gr returning nothing gives clear error" begin
        @test_throws ErrorException optim([1.0, 1.0], sphere;
                       gr=x -> nothing, method="BFGS")
    end

    @testset "Symbol keys in control accepted (convenience)" begin
        # R's list names are strings, but Julia users may pass Symbol keys
        result = optim([5.0, 5.0], sphere; control=Dict(:maxit => 1000))
        @test result.convergence == 0

        # Unknown Symbol keys still warn
        @test_warn "unknown names in control" optim([5.0, 5.0], sphere;
                                                     control=Dict(:bogus => 42))
    end

    @testset "L-BFGS-B converges with numerical gradient (aliasing fix)" begin
        # Bug: numgrad_bounded! returns pre-allocated cache.df buffer. lbfgsbmin stored
        # gx = g(...) as a reference to this buffer. Subsequent g() calls in the line
        # search overwrote it, making y = gnew - gx ≈ 0 and destroying L-BFGS curvature.
        # Fix: lbfgsbmin now copies the initial gradient. R: convergence=0, (46,46).
        result = optim([-1.2, 1.0], rosenbrock; method="L-BFGS-B",
                       lower=[-5.0, -5.0], upper=[5.0, 5.0])
        @test result.convergence == 0
        @test result.value < 1e-4
        @test abs(result.par[1] - 1.0) < 0.01
        @test abs(result.par[2] - 1.0) < 0.01

        # Unbounded case should also converge
        result2 = optim([-1.2, 1.0], rosenbrock; method="L-BFGS-B")
        @test result2.convergence == 0
        @test result2.value < 1e-4
    end

    # ── L-BFGS-B full algorithm (GCP + subspace minimization) ────────────────
    @testset "L-BFGS-B Beale from [4,4] bounded (GCP fix)" begin
        # Bug: old projected-gradient L-BFGS-B converged to boundary local min
        # [0.023, -4.5] value≈9.39. Full GCP+subspace algorithm finds global opt.
        # R: par≈[3.0, 0.5], value≈1.5e-10, convergence=0.
        beale(x) = (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1] + x[1]*x[2]^2)^2 +
                   (2.625 - x[1] + x[1]*x[2]^3)^2
        r = optim([4.0, 4.0], beale; method="L-BFGS-B",
                  lower=[-4.5, -4.5], upper=[4.5, 4.5])
        @test r.convergence == 0
        @test r.value < 1e-6
        @test abs(r.par[1] - 3.0) < 0.01
        @test abs(r.par[2] - 0.5) < 0.01
    end

    @testset "L-BFGS-B Rosenbrock from [0,0] bounded (GCP fix)" begin
        # R returns convergence=52 (false alarm), Julia converges properly.
        # Both reach near-optimum. Julia's result is at least as good.
        r = optim([0.0, 0.0], rosenbrock; method="L-BFGS-B",
                  lower=[-5.0, -5.0], upper=[5.0, 5.0])
        @test r.value < 1e-4
        @test abs(r.par[1] - 1.0) < 0.01
        @test abs(r.par[2] - 1.0) < 0.01
    end

    @testset "L-BFGS-B unbounded Rosenbrock (GCP fix)" begin
        # Unbounded case should converge well with full algorithm
        r = optim([-1.2, 1.0], rosenbrock; method="L-BFGS-B")
        @test r.convergence == 0
        @test r.value < 1e-4
        @test abs(r.par[1] - 1.0) < 0.01
        @test abs(r.par[2] - 1.0) < 0.01
    end

    # ── Direct lbfgsbmin API: non-finite gradient handling ───────────────────
    @testset "lbfgsbmin NaN gradient does not falsely converge (bug fix)" begin
        # Bug: NaN in gradient → _proj_grad! returned 0.0 (NaN > m is false),
        # causing immediate false convergence with fail=0.
        using Durbyn.Optimize: lbfgsbmin
        f_ok(n, x, _) = sum(x .^ 2)
        g_nan(n, x, _) = [NaN, NaN]
        r = lbfgsbmin(f_ok, g_nan, [1.0, 1.0])
        @test r.fail != 0
        @test occursin("NON-FINITE", r.message)
    end

    @testset "lbfgsbmin Inf gradient returns non-finite error (bug fix)" begin
        # Bug: Inf gradient → line search failed with ABNORMAL_TERMINATION
        # instead of a clear non-finite gradient message.
        using Durbyn.Optimize: lbfgsbmin
        f_ok(n, x, _) = sum(x .^ 2)
        g_inf(n, x, _) = [Inf, Inf]
        r = lbfgsbmin(f_ok, g_inf, [1.0, 1.0])
        @test r.fail != 0
        @test occursin("NON-FINITE", r.message)
    end

    # ── Strict NM regression tests (R parity) ────────────────────────────────
    @testset "NM strict: Rosenbrock from [-1.2,1.0] (R parity)" begin
        result = optim([-1.2, 1.0], rosenbrock)
        @test result.convergence == 0
        @test result.counts.function_ == 195  # exact R match
        @test result.value < 1e-6
        @test abs(result.par[1] - 1.0) < 0.01
        @test abs(result.par[2] - 1.0) < 0.01
    end

    @testset "NM strict: Beale from [0,0] (R parity)" begin
        beale(x) = (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1] + x[1]*x[2]^2)^2 + (2.625 - x[1] + x[1]*x[2]^3)^2
        result = optim([0.0, 0.0], beale)
        @test result.convergence == 0
        @test result.counts.function_ < 100
        @test result.value < 1e-6
        @test abs(result.par[1] - 3.0) < 0.01
        @test abs(result.par[2] - 0.5) < 0.01
    end

    @testset "NM strict: Booth from [0,0] (R parity)" begin
        booth(x) = (x[1] + 2*x[2] - 7)^2 + (2*x[1] + x[2] - 5)^2
        result = optim([0.0, 0.0], booth)
        @test result.convergence == 0
        @test result.counts.function_ == 75  # exact R match
        @test result.value < 1e-5
        @test abs(result.par[1] - 1.0) < 0.01
        @test abs(result.par[2] - 3.0) < 0.01
    end

    @testset "NM strict: maxit behavior (R parity)" begin
        # maxit=10: convergence=1, fncount=maxit+1 (R behavior)
        r10 = optim([-1.2, 1.0], rosenbrock; control=Dict("maxit" => 10))
        @test r10.convergence == 1
        @test r10.counts.function_ == 11

        # maxit=50: convergence=1
        r50 = optim([-1.2, 1.0], rosenbrock; control=Dict("maxit" => 50))
        @test r50.convergence == 1
        @test r50.counts.function_ == 51

        # maxit=200: converges with same fncount as default (maxit=500)
        r200 = optim([-1.2, 1.0], rosenbrock; control=Dict("maxit" => 200))
        @test r200.convergence == 0
        @test r200.counts.function_ == 195
    end

    # ── Brent scalar-style callback (R compat) ──────────────────────────────
    @testset "Brent scalar-style callback - basic" begin
        # R's Brent passes scalar to fn, not a 1-element vector.
        # This must work: f(x) = (x - 2)^2 where x is a scalar.
        f_scalar(x) = (x - 2)^2
        result = optim([3.0], f_scalar; method="Brent", lower=-10.0, upper=10.0)
        @test abs(result.par[1] - 2.0) <= EPS_OPT
        @test result.value < 1e-8
    end

    @testset "Brent scalar-style callback - negative minimum" begin
        f_scalar(x) = (x + 3.0)^2
        result = optim([0.0], f_scalar; method="Brent", lower=-10.0, upper=10.0)
        @test abs(result.par[1] - (-3.0)) <= EPS_OPT
        @test result.value < 1e-8
    end

    @testset "Brent scalar-style callback - sin" begin
        # Minimum of sin(x) on [0, 2π] is at x = 3π/2
        result = optim([1.0], sin; method="Brent", lower=0.0, upper=2π)
        @test abs(result.par[1] - 3π/2) <= EPS_OPT
        @test abs(result.value - (-1.0)) < 1e-8
    end

    @testset "Brent scalar-style callback - with fnscale=-1 (maximize)" begin
        # Maximize x*(1-x) on [0,1] → max at x=0.5, value=0.25
        f_scalar(x) = x * (1 - x)
        result = optim([0.1], f_scalar; method="Brent", lower=0.0, upper=1.0,
                        control=Dict("fnscale" => -1.0))
        @test abs(result.par[1] - 0.5) <= EPS_OPT
        # value is returned in original scale (fnscale applied back)
        @test abs(result.value - 0.25) < 1e-8
    end

    @testset "Brent vector-style callback still works" begin
        # Verify existing x[1] style still works after the fix
        f_vec(x) = (x[1] - 2.0)^2
        result = optim([3.0], f_vec; method="Brent", lower=-10.0, upper=10.0)
        @test abs(result.par[1] - 2.0) <= EPS_OPT
        @test result.value < 1e-8
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
