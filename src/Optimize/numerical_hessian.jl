"""
    numerical_hessian(fn, x; gradient=nothing, fnscale=1.0, parscale=nothing, step_sizes=nothing, kwargs...)

Compute the Hessian matrix of `fn` at `x` using finite-difference formulas from
Nocedal and Wright (2006), Section 8.1.

- If `gradient` is provided, the Hessian is built by forward differencing the
  gradient column-wise (Eq. 8.7).
- If `gradient` is `nothing`, second-order finite differences of `fn` are used
  (Eq. 8.9).

# Arguments

- `fn::Function`: Scalar-valued objective function.
- `x::Vector`: Parameter vector at which to evaluate the Hessian.
- `gradient::Union{Function,Nothing}`: Optional gradient function `gradient(x)`.
- `fnscale::Number`: Function scaling factor (default: 1.0).
- `parscale::Vector`: Parameter scaling factors (default: ones).
- `step_sizes::Vector`: Base finite-difference steps. Actual per-parameter
  perturbation is `abs(step_sizes .* parscale)`.

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
function numerical_hessian(fn, x; gradient = nothing, fnscale = 1.0, parscale = nothing,
                           step_sizes = nothing, kwargs...)
    x0 = Float64.(x)
    n_parameters = length(x0)
    function_scale = Float64(fnscale)
    (isfinite(function_scale) && function_scale != 0.0) || throw(ArgumentError("fnscale must be finite and non-zero"))
    inverse_function_scale = inv(function_scale)

    parameter_scale = _resolve_parameter_scale(parscale, n_parameters)
    finite_difference_steps = _resolve_finite_difference_steps(
        step_sizes, x0, parameter_scale, !isnothing(gradient),
    )

    fn_with_kwargs = isempty(kwargs) ? fn : (xv -> fn(xv; kwargs...))
    objective_fn = xv -> _evaluate_scaled_objective(fn_with_kwargs, xv, inverse_function_scale)

    if isnothing(gradient)
        return _hessian_from_objective(objective_fn, x0, finite_difference_steps)
    end

    gradient_fn = isempty(kwargs) ? gradient : (xv -> gradient(xv; kwargs...))
    return _hessian_from_gradient(gradient_fn, x0, finite_difference_steps, inverse_function_scale)
end

@inline function _evaluate_scaled_objective(fn::Function, x::AbstractVector{<:Real}, inverse_fnscale::Float64)
    value = _as_scalar_float(fn(x))
    isfinite(value) || throw(ErrorException("objective function returned non-finite value"))
    return value * inverse_fnscale
end

function _resolve_parameter_scale(parscale, n_parameters::Int)
    scale_values = if isnothing(parscale)
        ones(Float64, n_parameters)
    elseif parscale isa Number
        fill(Float64(parscale), n_parameters)
    else
        values = Float64.(parscale)
        length(values) == n_parameters || throw(ArgumentError("parscale must have length $n_parameters"))
        values
    end

    @inbounds for i in eachindex(scale_values)
        scale = abs(scale_values[i])
        (isfinite(scale) && scale > 0.0) || throw(ArgumentError("all parscale entries must be finite and non-zero"))
        scale_values[i] = scale
    end

    return scale_values
end

function _resolve_finite_difference_steps(step_sizes, x0::Vector{Float64},
                                          parameter_scale::Vector{Float64},
                                          use_gradient_formula::Bool)
    n_parameters = length(x0)
    step_values = Vector{Float64}(undef, n_parameters)

    if isnothing(step_sizes)
        base_step = use_gradient_formula ? sqrt(eps(Float64)) : eps(Float64)^(0.25)
        @inbounds for i in 1:n_parameters
            step_scale = max(abs(x0[i]), parameter_scale[i], 1.0)
            step_values[i] = base_step * step_scale
        end
    elseif step_sizes isa Number
        base_step = abs(Float64(step_sizes))
        @inbounds for i in 1:n_parameters
            step_values[i] = base_step * parameter_scale[i]
        end
    else
        raw_steps = Float64.(step_sizes)
        length(raw_steps) == n_parameters || throw(ArgumentError("step_sizes must have length $n_parameters"))
        @inbounds for i in 1:n_parameters
            step_values[i] = abs(raw_steps[i]) * parameter_scale[i]
        end
    end

    @inbounds for i in eachindex(step_values)
        step = step_values[i]
        (isfinite(step) && step > 0.0) || throw(ArgumentError("all finite-difference steps must be finite and positive"))
    end

    return step_values
end

function _evaluate_gradient!(gradient_values::Vector{Float64}, gradient_fn::Function,
                             x::Vector{Float64}, inverse_fnscale::Float64)
    raw_gradient = gradient_fn(x)
    n_parameters = length(gradient_values)
    if raw_gradient isa Number
        n_parameters == 1 || throw(ArgumentError("gradient output length must match parameter length"))
        grad_value = Float64(raw_gradient) * inverse_fnscale
        isfinite(grad_value) || throw(ErrorException("gradient function returned non-finite value"))
        gradient_values[1] = grad_value
        return nothing
    end

    length(raw_gradient) == n_parameters || throw(ArgumentError("gradient output length must match parameter length"))

    @inbounds for i in 1:n_parameters
        grad_value = Float64(raw_gradient[i]) * inverse_fnscale
        isfinite(grad_value) || throw(ErrorException("gradient function returned non-finite value"))
        gradient_values[i] = grad_value
    end
    return nothing
end

function _hessian_from_gradient(gradient_fn::Function, x0::Vector{Float64},
                                step_values::Vector{Float64}, inverse_fnscale::Float64)
    n_parameters = length(x0)
    hessian_matrix = Matrix{Float64}(undef, n_parameters, n_parameters)
    gradient_at_x = Vector{Float64}(undef, n_parameters)
    gradient_at_shift = similar(gradient_at_x)
    x_shifted = copy(x0)

    _evaluate_gradient!(gradient_at_x, gradient_fn, x0, inverse_fnscale)

    @inbounds for j in 1:n_parameters
        step_j = step_values[j]
        x_shifted[j] = x0[j] + step_j
        _evaluate_gradient!(gradient_at_shift, gradient_fn, x_shifted, inverse_fnscale)

        inv_step_j = inv(step_j)
        for i in 1:n_parameters
            hessian_matrix[i, j] = (gradient_at_shift[i] - gradient_at_x[i]) * inv_step_j
        end
        x_shifted[j] = x0[j]
    end

    _symmetrize_hessian!(hessian_matrix)
    return hessian_matrix
end

function _hessian_from_objective(objective_fn::Function, x0::Vector{Float64}, step_values::Vector{Float64})
    n_parameters = length(x0)
    hessian_matrix = zeros(Float64, n_parameters, n_parameters)
    x_shifted = copy(x0)

    objective_at_x = objective_fn(x0)
    objective_plus = Vector{Float64}(undef, n_parameters)
    objective_minus = Vector{Float64}(undef, n_parameters)

    @inbounds for i in 1:n_parameters
        xi = x0[i]
        step_i = step_values[i]

        x_shifted[i] = xi + step_i
        objective_plus[i] = objective_fn(x_shifted)

        x_shifted[i] = xi - step_i
        objective_minus[i] = objective_fn(x_shifted)

        x_shifted[i] = xi
    end

    @inbounds for i in 1:n_parameters
        step_i = step_values[i]
        hessian_matrix[i, i] = (objective_plus[i] - 2.0 * objective_at_x + objective_minus[i]) / (step_i * step_i)
    end

    @inbounds for i in 1:(n_parameters - 1)
        xi = x0[i]
        step_i = step_values[i]
        x_shifted[i] = xi + step_i

        for j in (i + 1):n_parameters
            xj = x0[j]
            step_j = step_values[j]
            x_shifted[j] = xj + step_j

            objective_ij = objective_fn(x_shifted)
            mixed_derivative = (objective_ij - objective_plus[i] - objective_plus[j] + objective_at_x) / (step_i * step_j)
            hessian_matrix[i, j] = mixed_derivative
            hessian_matrix[j, i] = mixed_derivative

            x_shifted[j] = xj
        end

        x_shifted[i] = xi
    end

    return hessian_matrix
end

function _symmetrize_hessian!(hessian_matrix::Matrix{Float64})
    n_parameters = size(hessian_matrix, 1)
    @inbounds for j in 1:(n_parameters - 1)
        for i in (j + 1):n_parameters
            symmetric_value = 0.5 * (hessian_matrix[i, j] + hessian_matrix[j, i])
            hessian_matrix[i, j] = symmetric_value
            hessian_matrix[j, i] = symmetric_value
        end
    end
    return hessian_matrix
end

@inline function _as_scalar_float(value::Number)
    return Float64(value)
end

@inline function _as_scalar_float(value)
    if value === nothing
        throw(ArgumentError("objective function returned nothing instead of a scalar"))
    elseif value === missing
        throw(ArgumentError("objective function returned missing instead of a scalar"))
    end

    length(value) == 1 || throw(ArgumentError("objective function returned $(length(value)) values instead of 1"))
    return Float64(first(value))
end
