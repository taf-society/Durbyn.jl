"""
    NelderMeadOptions(;
        abstol=-Inf,
        reltol=sqrt(eps(Float64)),
        alpha=1.0,
        beta=0.5,
        gamma=2.0,
        trace=false,
        maxit=500,
        invalid_penalty=1e35,
        project_to_bounds=false,
        lower=nothing,
        upper=nothing,
        init_step_cap=nothing,
    )

Options for direct Nelder-Mead simplex minimization.

# Parameters
- `abstol`: absolute objective stopping level (`f_best <= abstol`).
- `reltol`: simplex function-spread stopping tolerance.
- `alpha`: reflection coefficient.
- `beta`: contraction coefficient; also used for shrink.
- `gamma`: expansion coefficient.
- `trace`: print per-iteration diagnostics.
- `maxit`: maximum objective evaluations.
- `invalid_penalty`: replacement value for non-finite objective values.
- `project_to_bounds`: clamp trial points to `[lower, upper]`.
- `lower`, `upper`: bound vectors used when `project_to_bounds=true`.
- `init_step_cap`: optional cap on initial axial simplex perturbations.

# References
- Nelder, J. A. & Mead, R. (1965). *A simplex method for function minimization*.
- Lagarias, J. C. et al. (1998). *Convergence properties of the Nelder-Mead simplex method in low dimensions*.
"""
struct NelderMeadOptions
    abstol::Float64
    reltol::Float64
    alpha::Float64
    beta::Float64
    gamma::Float64
    trace::Bool
    maxit::Int
    invalid_penalty::Float64
    project_to_bounds::Bool
    lower::Union{Nothing,AbstractVector{<:Real}}
    upper::Union{Nothing,AbstractVector{<:Real}}
    init_step_cap::Union{Nothing,Float64}
end

@enum NelderMeadStatus::Int begin
    NMConverged = 0
    NMMaxEvaluations = 1
    NMStagnated = 10
end

@inline function _nm_result(
    x_opt::Vector{Float64},
    f_opt::Float64,
    fncount::Int,
    status::NelderMeadStatus,
)
    return (x_opt = x_opt, f_opt = f_opt, fncount = fncount, fail = Int(status))
end

NelderMeadOptions(;
    abstol = -Inf,
    reltol = sqrt(eps(Float64)),
    alpha = 1.0,
    beta = 0.5,
    gamma = 2.0,
    trace = false,
    maxit = 500,
    invalid_penalty = 1.0e35,
    project_to_bounds = false,
    lower = nothing,
    upper = nothing,
    init_step_cap = nothing,
) = NelderMeadOptions(
    abstol,
    reltol,
    alpha,
    beta,
    gamma,
    trace,
    maxit,
    invalid_penalty,
    project_to_bounds,
    lower,
    upper,
    init_step_cap,
)

@inline function _nm_clamp!(point::Vector{Float64}, lower::Vector{Float64}, upper::Vector{Float64})
    @inbounds @simd for index in eachindex(point, lower, upper)
        value = point[index]
        if value < lower[index]
            point[index] = lower[index]
        elseif value > upper[index]
            point[index] = upper[index]
        end
    end
    return point
end

function _nm_bounds(options::NelderMeadOptions, dimension::Int)
    has_bounds = options.project_to_bounds &&
                 !isnothing(options.lower) &&
                 !isnothing(options.upper)
    if !has_bounds
        return nothing, nothing
    end

    lower = Float64.(options.lower)
    upper = Float64.(options.upper)
    length(lower) == dimension || throw(ArgumentError("lower must have length $dimension"))
    length(upper) == dimension || throw(ArgumentError("upper must have length $dimension"))
    all(lower .<= upper) || throw(ArgumentError("all lower bounds must be <= upper bounds"))
    return lower, upper
end

@inline function _nm_safe_eval(
    objective,
    trial_point::Vector{Float64},
    invalid_penalty::Float64,
    lower::Union{Nothing,Vector{Float64}},
    upper::Union{Nothing,Vector{Float64}},
)::Float64
    if !isnothing(lower)
        _nm_clamp!(trial_point, lower, upper)
    end
    objective_value = Float64(objective(trial_point))
    return isfinite(objective_value) ? objective_value : invalid_penalty
end

@inline function _nm_initial_step(starting_point::Vector{Float64})::Float64
    step_size = 0.0
    @inbounds for coordinate_value in starting_point
        step_size = max(step_size, 0.1 * abs(coordinate_value))
    end
    return step_size == 0.0 ? 0.1 : step_size
end

@inline function _nm_spread_std(simplex_values::Vector{Float64}, dimension::Int)::Float64
    n_vertices = length(simplex_values)
    mean_value = sum(simplex_values) / n_vertices
    centered_sum_squares = 0.0
    @inbounds for value in simplex_values
        centered = value - mean_value
        centered_sum_squares += centered * centered
    end
    return sqrt(centered_sum_squares / max(dimension, 1))
end

function _nm_best_worst_second(simplex_values::Vector{Float64})
    n_vertices = length(simplex_values)
    best_index = 1
    worst_index = 1
    best_value = simplex_values[1]
    worst_value = simplex_values[1]

    @inbounds for vertex_index in 2:n_vertices
        value = simplex_values[vertex_index]
        if value < best_value
            best_value = value
            best_index = vertex_index
        end
        if value > worst_value
            worst_value = value
            worst_index = vertex_index
        end
    end

    second_worst_index = best_index
    second_worst_value = -Inf
    @inbounds for vertex_index in 1:n_vertices
        if vertex_index == worst_index
            continue
        end
        value = simplex_values[vertex_index]
        if value > second_worst_value
            second_worst_value = value
            second_worst_index = vertex_index
        end
    end
    return best_index, worst_index, second_worst_index
end

@inline function _nm_centroid_excluding_worst!(
    centroid::Vector{Float64},
    simplex_vertices::Matrix{Float64},
    worst_index::Int,
)
    dimension, n_vertices = size(simplex_vertices)
    @inbounds for coordinate_index in 1:dimension
        coordinate_sum = 0.0
        for vertex_index in 1:n_vertices
            if vertex_index != worst_index
                coordinate_sum += simplex_vertices[coordinate_index, vertex_index]
            end
        end
        centroid[coordinate_index] = coordinate_sum / (n_vertices - 1)
    end
    return centroid
end

@inline function _nm_reflect!(
    reflected_point::Vector{Float64},
    centroid::Vector{Float64},
    worst_vertex::AbstractVector{<:Real},
    alpha::Float64,
)
    @inbounds @simd for coordinate_index in eachindex(reflected_point, centroid, worst_vertex)
        reflected_point[coordinate_index] =
            (1.0 + alpha) * centroid[coordinate_index] - alpha * worst_vertex[coordinate_index]
    end
    return reflected_point
end

@inline function _nm_expand!(
    expanded_point::Vector{Float64},
    reflected_point::Vector{Float64},
    centroid::Vector{Float64},
    gamma::Float64,
)
    @inbounds @simd for coordinate_index in eachindex(expanded_point, reflected_point, centroid)
        expanded_point[coordinate_index] =
            gamma * reflected_point[coordinate_index] + (1.0 - gamma) * centroid[coordinate_index]
    end
    return expanded_point
end

@inline function _nm_contract!(
    contracted_point::Vector{Float64},
    contracted_vertex::Vector{Float64},
    centroid::Vector{Float64},
    beta::Float64,
)
    @inbounds @simd for coordinate_index in eachindex(contracted_point, contracted_vertex, centroid)
        contracted_point[coordinate_index] =
            beta * contracted_vertex[coordinate_index] + (1.0 - beta) * centroid[coordinate_index]
    end
    return contracted_point
end

function _nm_shrink!(
    simplex_vertices::Matrix{Float64},
    simplex_values::Vector{Float64},
    best_index::Int,
    objective,
    invalid_penalty::Float64,
    lower::Union{Nothing,Vector{Float64}},
    upper::Union{Nothing,Vector{Float64}},
    fncount::Int,
    trial_point::Vector{Float64},
)
    n_vertices = size(simplex_vertices, 2)
    dimension = size(simplex_vertices, 1)
    best_vertex = @view simplex_vertices[:, best_index]
    simplex_diameter = 0.0

    @inbounds for vertex_index in 1:n_vertices
        if vertex_index == best_index
            continue
        end
        for coordinate_index in 1:dimension
            old_value = simplex_vertices[coordinate_index, vertex_index]
            new_value = 0.5 * (old_value + best_vertex[coordinate_index])
            simplex_vertices[coordinate_index, vertex_index] = new_value
            simplex_diameter += abs(new_value - best_vertex[coordinate_index])
        end
        copyto!(trial_point, @view(simplex_vertices[:, vertex_index]))
        simplex_values[vertex_index] = _nm_safe_eval(objective, trial_point, invalid_penalty, lower, upper)
        simplex_vertices[:, vertex_index] .= trial_point
        fncount += 1
    end
    return simplex_diameter, fncount
end

raw"""
    nelder_mead(f, x0, options::NelderMeadOptions)

Direct Nelder-Mead minimization derived from the original simplex equations.

Implemented equations:

```math
\bar{P} = \frac{1}{n}\sum_{i \ne h} P_i,\qquad
P^* = (1 + \alpha)\bar{P} - \alpha P_h
```
```math
P^{**}_{\text{expand}} = \gamma P^* + (1-\gamma)\bar{P}
```
```math
P^{**}_{\text{contract}} = \beta P_h + (1-\beta)\bar{P}
```
```math
P_i \leftarrow \frac{P_i + P_l}{2}\quad(\text{shrink})
```

Stopping checks simplex function-value spread:

```math
\sigma = \sqrt{\frac{1}{n}\sum_{i=0}^{n}(y_i - \bar{y})^2}
```

Returns `(x_opt, f_opt, fncount, fail)`:
- `fail == 0`: converged
- `fail == 1`: reached function-evaluation limit
- `fail == 10`: shrink failed to reduce simplex diameter
"""
function nelder_mead(
    objective,
    initial_point::AbstractVector{<:Real},
    options::NelderMeadOptions,
)
    dimension = length(initial_point)
    starting_point = Float64.(initial_point)
    lower, upper = _nm_bounds(options, dimension)

    if !isnothing(lower)
        _nm_clamp!(starting_point, lower, upper)
    end

    options.maxit < 0 && throw(ArgumentError("maxit must be non-negative"))

    if options.maxit == 0
        objective_value = Float64(objective(starting_point))
        isfinite(objective_value) || error("function cannot be evaluated at initial parameters")
        return _nm_result(copy(starting_point), objective_value, 0, NMConverged)
    end

    if dimension == 0
        objective_value = Float64(objective(starting_point))
        isfinite(objective_value) || error("function cannot be evaluated at initial parameters")
        return _nm_result(Float64[], objective_value, 1, NMConverged)
    end

    n_vertices = dimension + 1
    simplex_vertices = Matrix{Float64}(undef, dimension, n_vertices)
    simplex_values = Vector{Float64}(undef, n_vertices)

    simplex_vertices[:, 1] .= starting_point
    initial_value = Float64(objective(starting_point))
    isfinite(initial_value) || error("function cannot be evaluated at initial parameters")
    simplex_values[1] = initial_value
    fncount = 1
    spread_tolerance = 0.5 * options.reltol * (abs(initial_value) + options.reltol)

    initial_step = _nm_initial_step(starting_point)
    @inbounds for coordinate_index in 1:dimension
        simplex_vertices[:, coordinate_index + 1] .= starting_point
        base_value = starting_point[coordinate_index]
        trial_value = base_value
        trial_step = initial_step
        for _ in 1:16
            proposed = base_value + trial_step
            if !isnothing(options.init_step_cap)
                proposed = base_value + min(trial_step, options.init_step_cap)
            end
            if !isnothing(lower)
                proposed = clamp(proposed, lower[coordinate_index], upper[coordinate_index])
            end
            if proposed != base_value
                trial_value = proposed
                break
            end
            trial_step *= 10.0
        end
        if trial_value == base_value
            proposed = nextfloat(base_value)
            if !isnothing(lower)
                proposed = clamp(proposed, lower[coordinate_index], upper[coordinate_index])
            end
            trial_value = proposed
        end
        simplex_vertices[coordinate_index, coordinate_index + 1] = trial_value
    end

    centroid = zeros(Float64, dimension)
    reflected_point = similar(centroid)
    expanded_point = similar(centroid)
    contracted_point = similar(centroid)
    contracted_vertex = similar(centroid)
    trial_point = similar(centroid)

    @inbounds for vertex_index in 2:n_vertices
        copyto!(trial_point, @view(simplex_vertices[:, vertex_index]))
        simplex_values[vertex_index] =
            _nm_safe_eval(objective, trial_point, options.invalid_penalty, lower, upper)
        simplex_vertices[:, vertex_index] .= trial_point
        fncount += 1
    end

    previous_shrink_diameter = Inf
    iteration = 0

    while fncount <= options.maxit
        iteration += 1

        best_index, worst_index, second_worst_index = _nm_best_worst_second(simplex_values)
        f_best = simplex_values[best_index]
        f_worst = simplex_values[worst_index]
        simplex_spread = _nm_spread_std(simplex_values, dimension)
        if f_worst <= f_best + spread_tolerance || simplex_spread <= spread_tolerance || f_best <= options.abstol
            return _nm_result(copy(@view(simplex_vertices[:, best_index])), f_best, fncount, NMConverged)
        end

        _nm_centroid_excluding_worst!(centroid, simplex_vertices, worst_index)

        worst_vertex = @view simplex_vertices[:, worst_index]
        _nm_reflect!(reflected_point, centroid, worst_vertex, options.alpha)
        f_reflect = _nm_safe_eval(objective, reflected_point, options.invalid_penalty, lower, upper)
        fncount += 1

        action::Symbol = :reflect
        if f_reflect < f_best
            _nm_expand!(expanded_point, reflected_point, centroid, options.gamma)
            f_expand = _nm_safe_eval(objective, expanded_point, options.invalid_penalty, lower, upper)
            fncount += 1
            if f_expand < f_reflect
                simplex_vertices[:, worst_index] .= expanded_point
                simplex_values[worst_index] = f_expand
                action = :expand
            else
                simplex_vertices[:, worst_index] .= reflected_point
                simplex_values[worst_index] = f_reflect
            end
        elseif f_reflect < simplex_values[second_worst_index]
            simplex_vertices[:, worst_index] .= reflected_point
            simplex_values[worst_index] = f_reflect
        else
            if f_reflect < f_worst
                contracted_vertex .= reflected_point
                action = :contract_outside
            else
                contracted_vertex .= worst_vertex
                action = :contract_inside
            end

            _nm_contract!(contracted_point, contracted_vertex, centroid, options.beta)
            f_contract = _nm_safe_eval(objective, contracted_point, options.invalid_penalty, lower, upper)
            fncount += 1

            contraction_threshold = f_reflect < f_worst ? f_reflect : f_worst
            if f_contract <= contraction_threshold
                simplex_vertices[:, worst_index] .= contracted_point
                simplex_values[worst_index] = f_contract
            else
                action = :shrink
                shrink_diameter, fncount = _nm_shrink!(
                    simplex_vertices,
                    simplex_values,
                    best_index,
                    objective,
                    options.invalid_penalty,
                    lower,
                    upper,
                    fncount,
                    trial_point,
                )
                if shrink_diameter >= previous_shrink_diameter
                    return _nm_result(
                        copy(@view(simplex_vertices[:, best_index])),
                        simplex_values[best_index],
                        fncount,
                        NMStagnated,
                    )
                end
                previous_shrink_diameter = shrink_diameter
            end
        end

        if options.trace
            current_best_index, _, _ = _nm_best_worst_second(simplex_values)
            println(
                "iter=", iteration,
                " evals=", fncount,
                " action=", action,
                " best=", simplex_values[current_best_index],
            )
        end
    end

    best_index, _, _ = _nm_best_worst_second(simplex_values)
    return _nm_result(
        copy(@view(simplex_vertices[:, best_index])),
        simplex_values[best_index],
        fncount,
        NMMaxEvaluations,
    )
end

function nelder_mead(objective, initial_point::AbstractVector{<:Real}; kwargs...)
    return nelder_mead(objective, initial_point, NelderMeadOptions(; kwargs...))
end
