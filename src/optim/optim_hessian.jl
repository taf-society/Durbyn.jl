export optim_hessian

function compute_gradient(p, fn, gr, fnscale, parscale, ndeps; kwargs...)
    n = length(p)
    df = zeros(n)

    if gr !== nothing
        x = p .* parscale
        df = gr(x; kwargs...) .* parscale ./ fnscale
        return df
    end

    x = p
    for i in 1:n
        eps = ndeps[i]
        xp = copy(x); xp[i] = (p[i] + eps) * parscale[i]
        val1 = fn(xp; kwargs...) / fnscale

        xm = copy(x); xm[i] = (p[i] - eps) * parscale[i]
        val2 = fn(xm; kwargs...) / fnscale

        df[i] = (val1 - val2) / (2 * eps)

        if !isfinite(df[i])
            error("non-finite finite-difference value at index $(i)")
        end
    end

    return df
end


"""
    optim_hessian(par, fn, gr=nothing; fnscale=1, parscale=nothing, ndeps=nothing, kwargs...)

Compute the Hessian matrix of a given function `fn` at the point `par`.

The Hessian is computed using the second-order finite difference method for numerical gradient estimation unless a gradient function `gr` is provided.

### Arguments
- `par::Vector{T}`: A vector of parameters at which the Hessian is evaluated.
- `fn::Function`: The function whose Hessian is to be computed. It must accept a vector of parameters and return a scalar.
- `gr::Function`: (optional) A user-provided gradient function. If `nothing`, the gradient is estimated numerically.
- `fnscale::Number`: (optional, default = `1`) A scaling factor for the function. This is used in the Hessian calculation.
- `parscale::Vector{T}`: (optional, default = `ones(length(par))`) A vector of scaling factors for the parameters, used in the finite difference method.
- `ndeps::Vector{T}`: (optional, default = `0.001 * ones(length(par))`) A vector of step sizes for the finite difference approximation.
- `kwargs...`: Additional arguments passed to `fn` and `gr`.

### Returns
- `hessian::Matrix{T}`: A square matrix of size `length(par) x length(par)` representing the Hessian matrix of `fn` at `par`.

### Details
The function computes the Hessian matrix using a second-order finite difference method for numerical gradient estimation if no gradient function is provided. If a gradient function is provided, it is used to calculate the Hessian matrix directly. The Hessian is returned as a symmetric matrix.

### Example Usage
```julia

# Rosenbrock function (Banana function)
function rosenbrock(x::Vector{T}) where T
    x1, x2 = x
    return 100 * (x2 - x1^2)^2 + (1 - x1)^2
end

# Gradient of the Rosenbrock function
function rosenbrock_grad(x::Vector{T}) where T
    x1, x2 = x
    return [
        -400 * x1 * (x2 - x1^2) - 2 * (1 - x1),  # Partial derivative wrt x1
        200 * (x2 - x1^2)  # Partial derivative wrt x2
    ]
end


# Ackley function
function ackley(x::Vector{T}) where T
    n = length(x)
    sum1 = sum(xi^2 for xi in x)
    sum2 = sum(cos(2 * π * xi) for xi in x)
    return -20 * exp(-0.2 * sqrt(sum1 / n)) - exp(sum2 / n) + exp(1) + 20
end

# Gradient of the Ackley function
function ackley_grad(x::Vector{T}) where T
    n = length(x)
    sum1 = sum(xi^2 for xi in x)
    sum2 = sum(cos(2 * π * xi) for xi in x)

    grad = Float64[]
    for xi in x
        grad = vcat(grad, -0.4 * xi * exp(-0.2 * sqrt(sum1 / n)) / sqrt(sum1 / n) - 2 * π * sin(2 * π * xi) * exp(sum2 / n) / n)
    end
    return grad
end


# Sphere function
function sphere(x::Vector{T}) where T
    return sum(xi^2 for xi in x)
end

# Gradient of the Sphere function
function sphere_grad(x::Vector{T}) where T
    return 2 * x  # Derivative of x^2 is 2x
end

# Himmelblau's function
function himmelblau(x::Vector{T}) where T
    x1, x2 = x
    return (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2
end

# Gradient of Himmelblau's function
function himmelblau_grad(x::Vector{T}) where T
    x1, x2 = x
    return [
        4 * x1 * (x1^2 + x2 - 11) + 2 * (x1 + x2^2 - 7),  # Partial derivative wrt x1
        2 * (x1^2 + x2 - 11) + 4 * x2 * (x1 + x2^2 - 7)  # Partial derivative wrt x2
    ]
end


# Rosenbrock Test
hessian_rosenbrock = optim_hessian([1.0, 1.0], rosenbrock, rosenbrock_grad)
println("Hessian of the Rosenbrock function:")
println(hessian_rosenbrock)

hessian_rosenbrock2 = optim_hessian([1.0, 1.0], rosenbrock, nothing)
println("Hessian of the Rosenbrock function:")
println(hessian_rosenbrock2)

# Ackley Test
hessian_ackley = optim_hessian([0.0, 0.0], ackley, ackley_grad)
println("Hessian of the Ackley function:")
println(hessian_ackley)

hessian_ackley2 = optim_hessian([0.0, 0.0], ackley, nothing)
println("Hessian of the Ackley function:")
println(hessian_ackley2)


# Sphere Test
hessian_sphere = optim_hessian([1.0, 1.0], sphere, sphere_grad)
println("Hessian of the Sphere function:")
println(hessian_sphere)

hessian_sphere2 = optim_hessian([1.0, 1.0], sphere, nothing)
println("Hessian of the Sphere function:")
println(hessian_sphere2)

# Himmelblau Test
hessian_himmelblau = optim_hessian([1.0, 1.0], himmelblau, himmelblau_grad)
println("Hessian of the Himmelblau function:")
println(hessian_himmelblau)

```

### Notes
- If `gr` is `nothing`, the gradient is estimated numerically using finite differences with a small step size (`1e-6`).
- The `gr` function must return a vector of gradients for each parameter, i.e., it should have the same length as `par`.

"""
function optim_hessian(par, fn, gr = nothing; fnscale = 1.0, parscale = nothing, ndeps = nothing, kwargs...)
    npar = length(par)
    fn1(par) = fn(par; kwargs...)

    parscale = parscale === nothing ? ones(npar) : parscale
    ndeps    = ndeps === nothing    ? fill(0.001, npar) : ndeps

    hessian = zeros(npar, npar)
    dpar = par ./ parscale

    for i in 1:npar
        eps = ndeps[i] / parscale[i]

        dpar[i] += eps
        df1 = compute_gradient(dpar, fn1, gr, fnscale, parscale, ndeps; kwargs...)

        dpar[i] -= 2 * eps
        df2 = compute_gradient(dpar, fn1, gr, fnscale, parscale, ndeps; kwargs...)

        for j in 1:npar
            hessian[i, j] = fnscale * (df1[j] - df2[j]) /
                            (2 * eps * parscale[i] * parscale[j])
        end

        dpar[i] += eps
    end

    for i in 1:npar
        for j in 1:(i - 1)
            tmp = 0.5 * (hessian[i, j] + hessian[j, i])
            hessian[i, j] = hessian[j, i] = tmp
        end
    end
    return hessian
end

