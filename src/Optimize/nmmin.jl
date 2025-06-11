"""
	NelderMeadOptions(; abstol=1e-4, intol=1e-4, alpha=1.0, beta=0.5, gamma=2.0, trace=false, maxit=100)

A configuration container for the Nelder-Mead optimization algorithm.

# Keyword Arguments

- `abstol::Float64`: Absolute tolerance on the function value for stopping. Default is `1e-4`.
- `intol::Float64`: Relative tolerance between the best and worst function values. Default is `1e-4`.
- `alpha::Float64`: Reflection coefficient. Controls how far to reflect. Default is `1.0`.
- `beta::Float64`: Contraction coefficient. Controls step size during contraction. Default is `0.5`.
- `gamma::Float64`: Expansion coefficient. Determines step size during expansion. Default is `2.0`.
- `trace::Bool`: If `true`, prints diagnostic output. Default is `false`.
- `maxit::Int`: Maximum number of iterations. Default is `500`.

# Example

```julia
opts = NelderMeadOptions(abstol=1e-4, maxit=100, trace=true)
````
"""
struct NelderMeadOptions
    abstol::Float64
    intol::Float64
    alpha::Float64
    beta::Float64
    gamma::Float64
    trace::Bool
    maxit::Int
end

NelderMeadOptions(;
    abstol = 1e-4,
    intol = 1e-4,
    alpha = 1.0,
    beta = 0.5,
    gamma = 2.0,
    trace = false,
    maxit = 100,
) = NelderMeadOptions(abstol, intol, alpha, beta, gamma, trace, maxit)


"""
	nmmin(f, x0, options::NelderMeadOptions)

Minimizes a real-valued function `f` using the **Nelder-Mead simplex algorithm** starting from an initial guess `x0`. 
The algorithm is derivative-free and useful for optimizing non-smooth or noisy functions.

# Arguments

- `f::Function`: The objective function to minimize. Must take a vector of Float64s and return a scalar Float64.
- `x0::Vector{Float64}`: Initial point in the parameter space. The algorithm builds its initial simplex around this.

- `options::NelderMeadOptions`: A struct containing the control parameters:
  - `abstol::Float64`: Absolute tolerance on the function value. Algorithm stops when best value is below this.
  - `intol::Float64`: Tolerance on relative difference between best and worst values. Stops if small enough.
  - `alpha::Float64`: Reflection coefficient. Controls how far to reflect away from worst point.
  - `beta::Float64`: Contraction coefficient. Used when reflection fails.
  - `gamma::Float64`: Expansion coefficient. Controls how far to expand promising moves.
  - `trace::Bool`: If `true`, prints progress at each step.
  - `maxit::Int`: Maximum number of iterations to run.

# Behavior

The algorithm iteratively updates a simplex of `n+1` points (for `n`-dimensional input) using reflection, expansion, contraction, 
	and shrink operations, guided by the coefficients `alpha`, `beta`, and `gamma`. It terminates when either:

- The maximum number of iterations (`maxit`) is reached,
- The best function value is below `abstol`, or
- The relative function value spread is below `intol`.

# Tuning

| Option    |   Increase                        |   Decrease                  |
|-----------|----------------------------------|------------------------------|
| `abstol`  | Faster but less precise          | Slower, more accurate        |
| `intol`   | Looser convergence               | Tighter convergence          |
| `alpha`   | More aggressive reflection       | More conservative movement   |
| `beta`    | Gentler contractions             | Stronger contractions        |
| `gamma`   | Larger expansions                | Smaller steps                |
| `maxit`   | Allows longer convergence        | May stop prematurely         |
| `trace`   | Verbose output                   | Cleaner, silent run          |

# Example

```julia
f(x) = sum((x .- 3).^2)  # simple quadratic minimum at x = [3, 3]
x0 = [0.0, 0.0]
opt = NelderMeadOptions(abstol=1e-6, maxit=2000, trace=true)
res = nmmin(f, x0, opt)
````

# Returns
A named tuple with the following fields:
- `x_opt`: The estimated minimizer of the function.
- `f_opt`: The function value at `x_opt`.
- `n_iter`: The maximum number of iterations attempted.
- `fail`: A flag that is `0` if the method converged, or `10` if shrinking failed to reduce the simplex size.
- `evals`: The number of function evaluations performed.

"""
function nmmin(f::Function, x0::Vector{Float64}, options::NelderMeadOptions)
	abstol = options.abstol
	intol = options.intol
	alpha = options.alpha
	beta = options.beta
	gamma = options.gamma
	trace = options.trace
	maxit = options.maxit

	n = length(x0)
	funcount = 0
	fail = 0
	fmin = 0.0

	# === Initialize simplex ===
	simplex = Matrix{Float64}(undef, n, n + 1)
	fvals = Vector{Float64}(undef, n + 1)

	# Vertex 1 = x0
	simplex[:, 1] .= x0
	fvals[1] = f(x0)
	funcount += 1

	if !isfinite(fvals[1])
		error("Function cannot be evaluated at initial parameters")
	end

	step = maximum(0.1 * abs.(x0))
	if step == 0.0
		step = 0.1
	end

	if trace
		println("Nelder-Mead (R-style): Initial f(x) = ", fvals[1])
		println("Step size = ", step)
	end

	# Build the rest of the simplex
	for j in 2:n+1
		simplex[:, j] .= x0
		simplex[j-1, j] += step
		fvals[j] = f(simplex[:, j])
		funcount += 1
	end

	oldsize = step * n

	for iter in 1:maxit
		# Order simplex by function value
		idx = sortperm(fvals)
		simplex .= simplex[:, idx]
		fvals .= fvals[idx]

		# Identify best, worst
		VL = fvals[1]
		VH = fvals[end]
		L = 1
		H = n + 1

		convtol = intol * (abs(VL) + intol)
		if VH <= VL + convtol || VL <= abstol
			break
		end

		if trace
			println("Iter $iter: best = $VL, worst = $VH, evals = $funcount")
		end

		# Compute centroid (excluding worst point)
		centroid = sum(simplex[:, 1:n], dims = 2)[:, 1] / n

		# === Reflection ===
		x_r = (1 + alpha) .* centroid .- alpha .* simplex[:, H]
		f_r = f(x_r)
		funcount += 1

		if f_r < VL
			# === Expansion ===
			x_e = gamma .* x_r .+ (1 - gamma) .* centroid
			f_e = f(x_e)
			funcount += 1
			if f_e < f_r
				simplex[:, H] .= x_e
				fvals[H] = f_e
				trace && println("Expansion")
			else
				simplex[:, H] .= x_r
				fvals[H] = f_r
				trace && println("Reflection (better)")
			end
		elseif f_r < fvals[n]
			# Accept reflection
			simplex[:, H] .= x_r
			fvals[H] = f_r
			trace && println("Reflection")
		else
			# === Contraction ===
			if f_r < VH
				x_c = beta .* x_r .+ (1 - beta) .* centroid
			else
				x_c = (1 - beta) .* centroid .+ beta .* simplex[:, H]
			end
			f_c = f(x_c)
			funcount += 1

			if f_c < VH
				simplex[:, H] .= x_c
				fvals[H] = f_c
				trace && println("Contraction")
			else
				# === Shrink simplex ===
				trace && println("Shrink")
				for j in 2:n+1
					simplex[:, j] .= simplex[:, L] .+ 0.5 .* (simplex[:, j] .- simplex[:, L])
					fvals[j] = f(simplex[:, j])
					funcount += 1
				end
				newsize = sum(abs.(simplex[:, 2:end] .- simplex[:, 1]))
				if newsize >= oldsize
					trace && println("Polytope size did not decrease on shrink")
					fail = 10
					break
				end
				oldsize = newsize
			end
		end
	end

	fmin = fvals[1]
	xopt = copy(simplex[:, 1])
	return (x_opt = xopt, f_opt = fmin, n_iter = maxit, fail = fail, evals = funcount)
end