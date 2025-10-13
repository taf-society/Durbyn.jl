"""
    BFGSOptions(; abstol=1e-8, reltol=1e-8, trace=false, maxit=1000, nREPORT=10)

Options for the BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimization algorithm.

# Keyword Arguments

- `abstol::Float64=1e-8`:  
    Absolute tolerance for function value convergence.  
    The algorithm stops if the objective function is less than or equal to this value.

- `reltol::Float64=1e-8`:  
    Relative tolerance for function value changes.  
    Optimization stops if the change in function value is below this relative threshold.

- `trace::Bool=false`:  
    If `true`, progress and diagnostics are printed during optimization.

- `maxit::Int=1000`:  
    Maximum number of optimization iterations.

- `nREPORT::Int=10`:  
    Reporting interval (print progress every `nREPORT` iterations if `trace` is true).

# Example

```julia
options = BFGSOptions(abstol=1e-6, reltol=1e-6, trace=true, maxit=500)
````

You can then pass this `options` object to the optimizer:

```julia
result = bfgsmin(f, g, x0; options=options)
```
"""
struct BFGSOptions
    abstol::Float64
    reltol::Float64
    trace::Bool
    maxit::Int
    nREPORT::Int
end

BFGSOptions(; abstol=1e-8, reltol=1e-8, trace=false, maxit=1000, nREPORT=10) =
    BFGSOptions(abstol, reltol, trace, maxit, nREPORT)

"""
    bfgsmin(f, g, x0; mask=trues(length(x0)), options=BFGSOptions())

Minimizes a multivariate function using the BFGS (Broyden-Fletcher-Goldfarb-Shanno) quasi-Newton algorithm with optional variable masking and options struct.

# Arguments

- `f::Function`: Objective function to minimize, called as `f(n, x, ex)`, where `n` is the dimension, `x` the parameter vector, and `ex` an (optional) extra argument.
- `g::Union{Function,Nothing}`: Gradient function, called as `g(n, x, ex)`, returning a vector. If `nothing`, numerical gradients will be computed using central differences.
- `x0::Vector{Float64}`: Initial guess for the variables.

# Keyword Arguments

- `mask::Vector{Bool}`: Optional. Specifies which variables are optimized (`true`) or held fixed (`false`). Defaults to all `true`.
- `options::BFGSOptions`: Optional. Struct with solver settings (tolerances, tracing, max iterations, etc).

# Returns

A named tuple with the following fields:
- `x_opt`: Optimal parameter vector found.
- `f_opt`: Function value at optimum.
- `n_iter`: Number of iterations performed.
- `fail`: Status flag (`0` if converged, otherwise indicates failure).
- `fn_evals`: Number of function evaluations.
- `gr_evals`: Number of gradient evaluations.

# Examples

## Rosenbrock Function

```julia
rosenbrock(n, x, ex) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
rosenbrock_grad(n, x, ex) = [
    -400 * (x[2] - x[1]^2) * x[1] - 2 * (1 - x[1]),
    200 * (x[2] - x[1]^2)
]

x0 = [-1.2, 1.0]
mask = [true, true]
options = BFGSOptions(trace=true, maxit=1000, nREPORT=10)

result = bfgsmin(
    rosenbrock, rosenbrock_grad, x0;
    mask=mask,
    options=options
)
println("Optimal x: ", result.x_opt)
````

## Quadratic Example

```julia
quad(n, x, ex) = (x[1] - 3)^2 + (x[2] + 1)^2
quad_grad(n, x, ex) = [2 * (x[1] - 3), 2 * (x[2] + 1)]
x0 = [0.0, 0.0]
mask = [true, true]
options = BFGSOptions()

result = bfgsmin(
    quad, quad_grad, x0;
    mask=mask,
    options=options
)

println("Optimal x: ", result.x_opt)
```

## Himmelblau Function

```julia
himmelblau(n, x, ex) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
himmelblau_grad(n, x, ex) = [
    4 * x[1] * (x[1]^2 + x[2] - 11) + 2 * (x[1] + x[2]^2 - 7),
    2 * (x[1]^2 + x[2] - 11) + 4 * x[2] * (x[1] + x[2]^2 - 7)
]
x0 = [0.0, 0.0]
mask = [true, true]
options = BFGSOptions(maxit=100)

result = bfgsmin(
    himmelblau, himmelblau_grad, x0;
    mask=mask,
    options=options
)
println("Optimal x: ", result.x_opt)
```

## Masked Variable Example (Optimize Only x_1)

```julia
quad(n, x, ex) = (x[1] - 3)^2 + (x[2] + 1)^2
quad_grad(n, x, ex) = [2 * (x[1] - 3), 2 * (x[2] + 1)]
x0 = [0.0, 0.0]
mask = [true, false]   # Only optimize x[1], keep x[2] fixed
options = BFGSOptions()

result = bfgsmin(
    quad, quad_grad, x0;
    mask=mask,
    options=options
)
println("Optimal x: ", result.x_opt)  # x[2] will remain at 0
```

## Numerical Gradients (No Gradient Function)

```julia
# When gradient is not available, pass nothing and numerical gradients will be computed
rosenbrock(n, x, ex) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
x0 = [-1.2, 1.0]

# Use numerical gradients (central differences with default step size 1e-3)
result = bfgsmin(rosenbrock, nothing, x0)

# Or specify custom step sizes for each parameter
result = bfgsmin(rosenbrock, nothing, x0; ndeps=[1e-4, 1e-4])

println("Optimal x: ", result.x_opt)
```

# See Also

* [`BFGSOptions`](@ref): Options struct for configuring the optimizer.
* [`numgrad`](@ref): Numerical gradient computation function.
"""
function bfgsmin(
    f::Function, g::Union{Function,Nothing}, x0::Vector{Float64};
    mask=trues(length(x0)),
    options::BFGSOptions=BFGSOptions(),
    ndeps::Union{Nothing,Vector{Float64}}=nothing
)
    # Extract options to locals for clarity
    abstol  = options.abstol
    reltol  = options.reltol
    trace   = options.trace
    maxit   = options.maxit
    nREPORT = options.nREPORT

    n0 = length(x0)
    l = findall(mask)
    n = length(l)
    b = copy(x0)
    gvec = zeros(n0)
    t = zeros(n)
    X = zeros(n)
    c = zeros(n)
    B = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n] # Hessian approx

    # Setup gradient function (analytical or numerical)
    if isnothing(g)
        # Use numerical gradients with central differences
        ndeps_actual = isnothing(ndeps) ? fill(1e-3, n0) : ndeps
        if length(ndeps_actual) != n0
            error("ndeps must have length $n0")
        end
        # numgrad is already loaded at module level in Optimize.jl
        gfunc = (n, x, ex) -> numgrad(f, n, x, ex, ndeps_actual)
    else
        gfunc = g
    end

    fval = f(n0, b, nothing)
    if !isfinite(fval)
        error("Initial value in 'bfgsmin' is not finite")
    end
    if trace
        println("initial value $fval")
    end

    Fmin = fval
    funcount = 1
    gradcount = 1
    gvec .= gfunc(n0, b, nothing)
    iter = 1
    ilast = gradcount
    fail = 1
    count = n

    while true
        # Restart Hessian if required
        if ilast == gradcount
            B .= 0.0
            for i in 1:n
                B[i,i] = 1.0
            end
        end

        for i in 1:n
            X[i] = b[l[i]]
            c[i] = gvec[l[i]]
        end

        gradproj = 0.0
        for i in 1:n
            s = 0.0
            for j in 1:i
                s -= B[i,j] * gvec[l[j]]
            end
            for j in i+1:n
                s -= B[j,i] * gvec[l[j]]
            end
            t[i] = s
            gradproj += s * gvec[l[i]]
        end

        if gradproj < 0.0
            steplength = 1.0
            accpoint = false
            while true
                count = 0
                for i in 1:n
                    tmp = X[i] + steplength * t[i]
                    if 1.0 + X[i] == 1.0 + tmp
                        count += 1
                    end
                    b[l[i]] = tmp
                end
                if count < n
                    fval = f(n0, b, nothing)
                    funcount += 1
                    accpoint = isfinite(fval) && (fval <= Fmin + gradproj * steplength * 1e-4)
                    if !accpoint
                        steplength *= 0.2
                    end
                end
                if count == n || accpoint
                    break
                end
            end

            enough = (fval > abstol) && abs(fval - Fmin) > reltol * (abs(Fmin) + reltol)
            if !enough
                count = n
                Fmin = fval
            end
            if count < n
                Fmin = fval
                gnew = gfunc(n0, b, nothing)
                gradcount += 1
                iter += 1
                D1 = 0.0
                for i in 1:n
                    t[i] = steplength * t[i]
                    c[i] = gnew[l[i]] - c[i]
                    D1 += t[i] * c[i]
                end
                if D1 > 0
                    X_ = zeros(n)
                    D2 = 0.0
                    for i in 1:n
                        s = 0.0
                        for j in 1:i
                            s += B[i,j] * c[j]
                        end
                        for j in i+1:n
                            s += B[j,i] * c[j]
                        end
                        X_[i] = s
                        D2 += s * c[i]
                    end
                    D2 = 1.0 + D2 / D1
                    for i in 1:n
                        for j in 1:i
                            B[i,j] += (D2 * t[i] * t[j] - X_[i] * t[j] - t[i] * X_[j]) / D1
                        end
                    end
                else
                    ilast = gradcount
                end
                gvec .= gnew
            else
                if ilast < gradcount
                    count = 0
                    ilast = gradcount
                end
            end
        else
            count = 0
            if ilast == gradcount
                count = n
            else
                ilast = gradcount
            end
        end

        if trace && (iter % nREPORT == 0)
            println("iter $iter value $fval")
        end

        if iter >= maxit
            break
        end
        if gradcount - ilast > 2 * n
            ilast = gradcount
        end
        if count == n && ilast == gradcount
            break
        end
    end

    if trace
        println("final value $Fmin")
        if iter < maxit
            println("converged")
        else
            println("stopped after $iter iterations")
        end
    end
    fail = (iter < maxit) ? 0 : 1

    return (
        x_opt = copy(b),
        f_opt = Fmin,
        n_iter = iter,
        fail = fail,
        fn_evals = funcount,
        gr_evals = gradcount
    )
end
