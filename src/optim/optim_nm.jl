export optim_nm

function reflect_point(centroid::Vector{T}, worst_point::Vector{T}, alpha::T = 1.0) where T<:Real
    return centroid .+ alpha .* (centroid .- worst_point)
end

function expand_point(centroid::Vector{T}, worst_point::Vector{T}, gamma::T = 2.0) where T<:Real
    return centroid .+ gamma .* (centroid .- worst_point)
end

function contract_point(centroid::Vector{T}, worst_point::Vector{T}, rho::T = -0.5) where T<:Real
    return centroid .+ rho .* (centroid .- worst_point)
end

function shrink_simplex(simplex::Matrix{T}, best_point::Vector{T}, sigma::T = 0.5) where T<:Real
    new_vertices = [best_point .+ sigma .* (simplex[i, :] .- best_point) for i in 1:size(simplex, 1)]
    return reduce(vcat, [transpose(v) for v in new_vertices])
end

function optim_nm_base(simplex_vertices::Matrix{T}, params_init::Vector{T},
                       objective_fn::Function;
                       iter_max::Int = 250, abs_tol::T = 0.0001,
                       alpha::T = 1.0, gamma::T = 2.0, rho::T = -0.5, sigma::T = 0.5) where T<:Real

    n = length(params_init)  # dimension of the parameter space
    iter = 0
    simplex = copy(simplex_vertices)

    while iter < iter_max
        # Evaluate objective function at each vertex (each row)
        vertex_values = [objective_fn(view(simplex, i, :)) for i in 1:size(simplex, 1)]
        
        # Order simplex vertices by their function values (lowest first)
        sorted_indices = sortperm(vertex_values)
        simplex = simplex[sorted_indices, :]
        vertex_values = vertex_values[sorted_indices]

        # Check convergence: difference between best and worst of the best n vertices
        if abs(objective_fn(simplex[n, :]) - objective_fn(simplex[1, :])) < abs_tol
            break
        end

        # Calculate centroid of the best n vertices (all except the worst vertex)
        centroid = vec(mean(simplex[1:n, :], dims = 1))

        worst_point = vec(simplex[n+1, :])
        # Reflection step
        reflected = reflect_point(centroid, worst_point, alpha)
        f_reflected = objective_fn(reflected)

        if objective_fn(simplex[1, :]) <= f_reflected < objective_fn(simplex[n, :])
            simplex[n+1, :] .= reflected
        elseif f_reflected < objective_fn(simplex[1, :])
            # Expansion step
            expanded = expand_point(centroid, worst_point, gamma)
            if objective_fn(expanded) < f_reflected
                simplex[n+1, :] .= expanded
            else
                simplex[n+1, :] .= reflected
            end
        else
            # Contraction step
            contracted = contract_point(centroid, worst_point, rho)
            if objective_fn(contracted) < objective_fn(worst_point)
                simplex[n+1, :] .= contracted
            else
                # Shrinkage step: shrink all vertices toward the best vertex.
                simplex = shrink_simplex(simplex, vec(simplex[1, :]), sigma)
            end
        end

        iter += 1
    end

    best_params = vec(simplex[1, :])
    best_value = objective_fn(best_params)
    return (optimal_params = best_params, optimal_value = best_value)
end

"""
		optim_nm(objective_fn::Function, params_init::Vector; iter_max::Int = 250, abs_tol::Real = 0.0001)
	
	Performs optimization using the Nelder-Mead simplex method on a given objective function.
	
	# Overview
	
	The `optim_nm` function attempts to find the minimum of the `objective_fn` starting from an initial guess
	`params_init` by iteratively updating a simplex of candidate points. It uses a multi-start approach where three
	different initial simplexes (with low, zero, and up perturbations) are evaluated to pick a promising starting
	point, which is then refined using a new simplex. The function handles one-dimensional problems by augmenting
	the parameter vector with a dummy parameter.
	
	# Arguments
	
	- `objective_fn::Function`: A function that takes a vector of parameters and returns a scalar objective value.
	- `params_init::Vector`: The initial guess for the parameters as a vector.
	- `iter_max::Int=250`: *(Optional)* The maximum number of iterations for the Nelder-Mead algorithm.
	- `abs_tol::Real=0.0001`: *(Optional)* The absolute tolerance used as the convergence criterion. The optimization
	  stops if the difference in the objective function value between the best and worst points of the current simplex
	  (among the best n points) is less than `abs_tol`.
	
	# Returns
	
	A named tuple with the following fields:
	- `par`: A vector representing the optimized parameters.
	- `value`: The value of the objective function at the optimized parameters.
	
	# Details
	
	Internally, `optim_nm` constructs three initial simplexes using different perturbations:
	- **lo:** A downward perturbation.
	- **zero:** No perturbation.
	- **up:** An upward perturbation.
	
	Each of these simplexes is refined using a base Nelder-Mead optimizer (`optim_nm_base`). The best
	result is selected to generate a new simplex, which is further optimized to produce the final result.
	
	If the optimization problem is one-dimensional, the function automatically augments the parameter vector
	with a dummy parameter and adapts the objective function accordingly.
	
	# Example
	
	```julia
	# Define a sample objective function, e.g., a simple quadratic function.
	objective_fn(x) = sum((x .- 3).^2)
	
	# Initial guess for a 2-dimensional problem.
	params_init = [0.0, 0.0]
	
	# Run the Nelder-Mead optimization.
	result = optim_nm(objective_fn, params_init; iter_max=500, abs_tol=1e-6)
	
	println("Optimized parameters: ", result.par)
	println("Objective function value: ", result.value)
	````
"""
function optim_nm(objective_fn::Function, params_init::Vector{T};
                  iter_max::Int = 250, abs_tol::T = 0.0001,
                  alpha::T = 1.0, gamma::T = 2.0, rho::T = -0.5, sigma::T = 0.5) where T<:Real
                  
    if length(params_init) == 1
        original_objective = objective_fn
        params_init = [params_init[1], zero(T)]
        objective_fn = x -> original_objective(x[1])
    end

    n = length(params_init)

    function make_simplex(delta::Function)
        vertices = [params_init]
        for i in 1:n
            vertex = copy(params_init)
            vertex[i] += delta(params_init[i])
            push!(vertices, vertex)
        end
        return reduce(vcat, [transpose(v) for v in vertices])
    end

    lo_delta = x -> -0.50 * x - 0.05
    zero_delta = x -> 0.0
    up_delta = x -> 0.50 * x + 0.05

    initial_simplexes = Dict(
        "lo"   => make_simplex(lo_delta),
        "zero" => make_simplex(zero_delta),
        "up"   => make_simplex(up_delta)
    )

    pocket = Array{T}(undef, 0, n+1)
    for (name, simplex) in initial_simplexes
        result = optim_nm_base(simplex, params_init, objective_fn;
                               iter_max = iter_max, abs_tol = abs_tol,
                               alpha = alpha, gamma = gamma, rho = rho, sigma = sigma)
        row = [result.optimal_value; result.optimal_params]
        pocket = vcat(pocket, transpose(row))
    end

    sort_order = sortperm(pocket[:, 1])
    pocket = pocket[sort_order, :]
    initial_guess = pocket[1, 2:end]

    function make_simplex_from_guess(guess::Vector{T})
        vertices = [guess]
        for i in 1:n
            vertex = copy(guess)
            vertex[i] += 0.05 * guess[i] + 0.01
            push!(vertices, vertex)
        end
        return reduce(vcat, [transpose(v) for v in vertices])
    end

    simplex_vertices = make_simplex_from_guess(initial_guess)
    final_result = optim_nm_base(simplex_vertices, initial_guess, objective_fn;
                                 iter_max = iter_max, abs_tol = abs_tol,
                                 alpha = alpha, gamma = gamma, rho = rho, sigma = sigma)

    return (par = final_result.optimal_params, value = final_result.optimal_value)
end
