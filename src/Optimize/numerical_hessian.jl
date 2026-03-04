"""
    numerical_hessian(fn, x; fnscale=1.0, parscale=nothing, step_sizes=nothing, kwargs...)

Compute the Hessian matrix of `fn` at `x` using Optim/NLSolversBase finite differences.

This routine delegates Hessian estimation to `Optim.TwiceDifferentiable` with
`AutoFiniteDiff`.

# Arguments

- `fn::Function`: Scalar-valued objective function.
- `x::Vector`: Parameter vector at which to evaluate the Hessian.
- `fnscale::Number`: Function scaling factor (default: 1.0).
- `parscale::Vector`: Parameter scaling factors (default: ones).
- `step_sizes::Vector`: Finite difference step sizes (default: 0.001).

# Returns

Symmetric Hessian matrix of size `(n, n)`.

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Section 8.1.
  Springer.

# Examples

```julia
rosenbrock(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
H = numerical_hessian(rosenbrock, [1.0, 1.0])
```
"""
function numerical_hessian(fn, x; fnscale = 1.0, parscale = nothing, step_sizes = nothing, kwargs...)
    n = length(x)
    x0 = Float64.(x)
    fscale = Float64(fnscale)
    (isfinite(fscale) && fscale != 0.0) || throw(ArgumentError("fnscale must be finite and non-zero"))

    ps = if isnothing(parscale)
        ones(n)
    elseif parscale isa Number
        fill(Float64(parscale), n)
    else
        v = Float64.(parscale)
        length(v) == n || throw(ArgumentError("parscale must have length $n"))
        v
    end

    h = if isnothing(step_sizes)
        fill(1e-3, n)
    elseif step_sizes isa Number
        fill(Float64(step_sizes), n)
    else
        v = Float64.(step_sizes)
        length(v) == n || throw(ArgumentError("step_sizes must have length $n"))
        v
    end

    # Desired perturbation in original coordinates.
    # FiniteDiff in Optim takes scalar abs/rel step, so we reparameterize with a diagonal scale:
    #   x = scale .* y, compute H_y(g), then recover H_x = D^{-1} * H_y * D^{-1}.
    scale = abs.(h .* ps)
    any(scale .<= 0.0) && throw(ArgumentError("all finite-difference scales must be positive"))
    all(isfinite, scale) || throw(ArgumentError("all finite-difference scales must be finite"))

    inv_scale = 1.0 ./ scale
    y0 = x0 .* inv_scale

    fn_k = isempty(kwargs) ? fn : (xv -> fn(xv; kwargs...))

    x_work = similar(x0)
    f_scaled = function (y)
        @inbounds @simd for i in eachindex(y)
            x_work[i] = y[i] * scale[i]
        end
        return Float64(fn_k(x_work)) / fscale
    end

    ad = Optim.ADTypes.AutoFiniteDiff(; relstep = 0.0, absstep = 1.0)
    obj = Optim.TwiceDifferentiable(f_scaled, y0; autodiff = ad)

    Optim.hessian!(obj, y0)
    H = Matrix{Float64}(Optim.NLSolversBase.hessian(obj))

    @inbounds for i in 1:n, j in 1:n
        H[i, j] *= inv_scale[i] * inv_scale[j]
    end

    return H
end
