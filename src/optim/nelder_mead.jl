using Optim, BenchmarkTools

# === Define the 10D Rosenbrock Function ===
rosenbrock_nd(x) = sum((1 .- x[1:end-1]).^2 .+ 100.0 .* (x[2:end] .- x[1:end-1].^2).^2)

# === Initial Setup ===
n = 10
x0 = fill(0.0, n)
lower = fill(-5.0, n)
upper = fill(5.0, n)

# === Your nelder_mead implementation (assumed defined elsewhere) ===
# Make sure you have nelder_mead(F, x0, lower, upper; ...) defined

# === Define Wrapper for optim_nm ===
function run_optim_nm()
    result = optim_nm(rosenbrock_nd, x0; iter_max=1000, abs_tol=1e-8)
    return result.value
end

# === Define Wrapper for Optim.jl ===
function run_optimjl()
    result = optimize(rosenbrock_nd, x0, NelderMead(),
                      Optim.Options(iterations=1000, f_tol=1e-8))
    return Optim.minimum(result)
end

# === Define Wrapper for custom nelder_mead ===
function run_nelder_mead()
    result = nelder_mead(rosenbrock_nd, x0, lower, upper;
                         max_iter=3000, tol_std=1e-10,
                         init_step=0.1, adaptive=true)
    return result.f_opt
end

# === Run Benchmarks ===
println("Benchmarking all optimizers on 10D Rosenbrock...\n")

println("Optim.jl:")
@btime run_optimjl()

println("\noptim_nm (multi-start):")
@btime run_optim_nm()

println("\nCustom nelder_mead:")
@btime run_nelder_mead()
