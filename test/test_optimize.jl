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
