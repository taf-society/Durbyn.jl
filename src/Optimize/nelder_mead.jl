"""
	NelderMeadOptions(; abstol=-Inf, reltol=sqrt(eps(Float64)), alpha=1.0, beta=0.5,
                        gamma=2.0, trace=false, maxit=500, invalid_penalty=1e35,
                        project_to_bounds=false, lower=nothing, upper=nothing,
                        init_step_cap=nothing)

A configuration container for the Nelder-Mead optimization algorithm.

# Keyword Arguments

- `abstol::Float64`: Absolute tolerance on the function value for stopping. Default is `-Inf`.
- `reltol::Float64`: Relative tolerance between the best and worst function values. Default is `sqrt(eps(Float64))`.
- `alpha::Float64`: Reflection coefficient. Controls how far to reflect. Default is `1.0`.
- `beta::Float64`: Contraction coefficient. Controls step size during contraction. Default is `0.5`.
- `gamma::Float64`: Expansion coefficient. Determines step size during expansion. Default is `2.0`.
- `trace::Bool`: If `true`, prints diagnostic output. Default is `false`.
- `maxit::Int`: Maximum number of function evaluations. Default is `500`.
- `invalid_penalty::Float64`: Penalty value for non-finite function evaluations. Default is `1e35`.
- `project_to_bounds::Bool`: If `true`, projects trial points to bounds before evaluation. Default is `false`.
- `lower::Union{Nothing,AbstractVector{<:Real}}`: Lower bounds (only used if `project_to_bounds=true`). Default is `nothing`.
- `upper::Union{Nothing,AbstractVector{<:Real}}`: Upper bounds (only used if `project_to_bounds=true`). Default is `nothing`.
- `init_step_cap::Union{Nothing,Float64}`: Maximum initial step size. Default is `nothing` (uncapped).

# References

- Nelder, J. A. & Mead, R. (1965). A simplex method for function minimization.
  *The Computer Journal*, 7(4), 308–313.
- Lagarias, J. C. et al. (1998). Convergence properties of the Nelder-Mead simplex
  method in low dimensions. *SIAM J. Optim.*, 9(1), 112–147.

# Example

```julia
opts = NelderMeadOptions(abstol=1e-4, maxit=100, trace=true)

# With bounds
opts = NelderMeadOptions(project_to_bounds=true, lower=[0.0, 0.0], upper=[10.0, 10.0])
```
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

"""
Simplex action identifiers for the Nelder-Mead algorithm.

Following the terminology of Lagarias et al. (1998):
- `BUILD`: Initial simplex construction
- `REFLECT`: Reflection step accepted
- `EXPAND`: Expansion step accepted
- `CONTRACT_OUTSIDE`: Outside contraction (reflected point better than worst)
- `CONTRACT_INSIDE`: Inside contraction (reflected point worse than worst)
- `SHRINK`: Shrink toward best vertex
"""
@enum SimplexAction begin
    SIMPLEX_BUILD
    SIMPLEX_REFLECT
    SIMPLEX_EXPAND
    SIMPLEX_CONTRACT_OUTSIDE
    SIMPLEX_CONTRACT_INSIDE
    SIMPLEX_SHRINK
end

const _SIMPLEX_ACTION_NAMES = Dict{SimplexAction,String}(
    SIMPLEX_BUILD             => "BUILD",
    SIMPLEX_REFLECT           => "REFLECT",
    SIMPLEX_EXPAND            => "EXPAND",
    SIMPLEX_CONTRACT_OUTSIDE  => "LO-REDUCTION",
    SIMPLEX_CONTRACT_INSIDE   => "HI-REDUCTION",
    SIMPLEX_SHRINK            => "SHRINK",
)

"""
    SimplexState

Internal workspace for the Nelder-Mead simplex algorithm. Stores the simplex
vertices and their corresponding function values.

# Fields
- `vertices::Matrix{Float64}` — `n × (n+1)` matrix; each column is a vertex
- `fvals::Vector{Float64}` — Function value at each vertex
- `idx_best::Int` — Column index of the best (lowest f) vertex
- `idx_worst::Int` — Column index of the worst (highest f) vertex

# References

- Nelder, J. A. & Mead, R. (1965). A simplex method for function minimization.
  *The Computer Journal*, 7(4), 308–313.
"""
mutable struct SimplexState
    vertices::Matrix{Float64}
    fvals::Vector{Float64}
    idx_best::Int
    idx_worst::Int
end

"""
    _simplex_centroid!(centroid, simplex, idx_worst, n)

Compute the centroid of all simplex vertices except the worst, storing
the result in `centroid`.
"""
@inline function _simplex_centroid!(centroid::AbstractVector{Float64},
    simplex::SimplexState, n::Int)
    idx_worst = simplex.idx_worst
    n_vertices = n + 1
    @inbounds for i in 1:n
        s = -simplex.vertices[i, idx_worst]
        for j in 1:n_vertices
            s += simplex.vertices[i, j]
        end
        centroid[i] = s / n
    end
end

"""
    _simplex_reflect!(trial, simplex, centroid, alpha, n)

Compute the reflection point: trial = (1 + α) * centroid - α * worst.
"""
@inline function _simplex_reflect!(trial::AbstractVector{Float64},
    simplex::SimplexState, centroid::AbstractVector{Float64},
    alpha::Float64, n::Int)
    idx_worst = simplex.idx_worst
    @inbounds for i in 1:n
        trial[i] = (1 + alpha) * centroid[i] - alpha * simplex.vertices[i, idx_worst]
    end
end

"""
    _simplex_expand!(trial, reflect_pt, centroid, gamma, n)

Compute the expansion point: trial = γ * reflect_pt + (1 - γ) * centroid.
"""
@inline function _simplex_expand!(trial::AbstractVector{Float64},
    reflect_pt::AbstractVector{Float64}, centroid::AbstractVector{Float64},
    gamma::Float64, n::Int)
    @inbounds for i in 1:n
        trial[i] = gamma * reflect_pt[i] + (1 - gamma) * centroid[i]
    end
end

"""
    _simplex_contract!(trial, simplex, centroid, beta, n)

Compute the contraction point: trial = (1 - β) * worst + β * centroid.
"""
@inline function _simplex_contract!(trial::AbstractVector{Float64},
    simplex::SimplexState, centroid::AbstractVector{Float64},
    beta::Float64, n::Int)
    idx_worst = simplex.idx_worst
    @inbounds for i in 1:n
        trial[i] = (1 - beta) * simplex.vertices[i, idx_worst] + beta * centroid[i]
    end
end

"""
    _simplex_shrink!(simplex, beta, n) -> Float64

Shrink all vertices toward the best vertex. Returns the new simplex diameter.
"""
function _simplex_shrink!(simplex::SimplexState, beta::Float64, n::Int)
    n_vertices = n + 1
    idx_best = simplex.idx_best
    diameter = 0.0
    @inbounds for j in 1:n_vertices
        if j != idx_best
            for i in 1:n
                simplex.vertices[i, j] = beta * (simplex.vertices[i, j] - simplex.vertices[i, idx_best]) + simplex.vertices[i, idx_best]
                diameter += abs(simplex.vertices[i, j] - simplex.vertices[i, idx_best])
            end
        end
    end
    return diameter
end

"""
    _simplex_update_extremes!(simplex, n)

Find the best and worst vertices in the simplex.
"""
function _simplex_update_extremes!(simplex::SimplexState, n::Int)
    n_vertices = n + 1
    f_best = simplex.fvals[simplex.idx_best]
    f_worst = f_best
    idx_worst = simplex.idx_best
    @inbounds for j in 1:n_vertices
        if j != simplex.idx_best
            fj = simplex.fvals[j]
            if fj < f_best
                f_best = fj
                simplex.idx_best = j
            end
            if fj > f_worst
                f_worst = fj
                idx_worst = j
            end
        end
    end
    simplex.idx_worst = idx_worst
end

"""
	nelder_mead(f, x0, options::NelderMeadOptions)

Minimize a function of several variables using the Nelder-Mead simplex algorithm.

The Nelder-Mead method is a derivative-free optimization algorithm that maintains a simplex
of n+1 vertices in n-dimensional space. At each step it transforms the simplex through
reflection, expansion, contraction, or shrink operations to move toward the minimum.

Optional bounds-aware extensions (`project_to_bounds`, `init_step_cap`) are available but
disabled by default.

# Arguments

- `f::Function`: Objective function mapping `Vector{Float64}` → `Real`. Should return a finite
  value for valid parameters. Non-finite values during iterations are replaced by
  `invalid_penalty`. If the initial evaluation at `x0` is non-finite, the routine returns
  immediately with `fail === true`.
- `x0::AbstractVector{<:Real}`: Starting point. Internally converted to `Vector{Float64}`.
- `options::NelderMeadOptions`: Control parameters (see `NelderMeadOptions` documentation).

# Returns

A named tuple `(x_opt, f_opt, fncount, fail)` where:
- `x_opt::Vector{Float64}`: Best parameters found.
- `f_opt::Float64`: Objective value at `x_opt`.
- `fncount::Int`: Total number of objective evaluations.
- `fail`: Status code:
    * `0` — Converged (scaled tolerance and/or `abstol` satisfied).
    * `1` — Exceeded `maxit` function evaluations.
    * `10` — Shrink step failed to reduce the simplex size (degenerate case).
    * `true` — Initial evaluation at `x0` was non-finite.

# References

- Nelder, J. A. & Mead, R. (1965). A simplex method for function minimization.
  *The Computer Journal*, 7(4), 308–313.
- Lagarias, J. C. et al. (1998). Convergence properties of the Nelder-Mead simplex
  method in low dimensions. *SIAM J. Optim.*, 9(1), 112–147.
- Nash, J. C. (1990). *Compact Numerical Methods for Computers*, 2nd ed., Algorithm 19.
  Adam Hilger.

# Examples

```julia
using Durbyn.Optimize

rosen(x) = (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2
opts = NelderMeadOptions(trace=true, maxit=2000)
res = nelder_mead(rosen, [-1.2, 1.0], opts)
@show res.x_opt res.f_opt res.fncount res.fail

# With box constraints
f(x) = (x[1]-0.8)^2 + (x[2]-0.2)^2
opts_bounded = NelderMeadOptions(
    project_to_bounds=true, lower=[0.0, 0.0], upper=[1.0, 1.0],
    init_step_cap=0.25)
resb = nelder_mead(f, [0.3, 0.3], opts_bounded)
```
"""
function nelder_mead(f::Function, x0::AbstractVector{<:Real}, options::NelderMeadOptions)
    abstol = options.abstol
    reltol = options.reltol
    alpha = options.alpha
    beta = options.beta
    gamma = options.gamma
    trace = options.trace
    maxit = options.maxit
    invalid_penalty = options.invalid_penalty
    project_to_bounds = options.project_to_bounds
    lower = options.lower
    upper = options.upper
    init_step_cap = options.init_step_cap

    n = length(x0)
    trial = collect(float.(x0))

    @inline function clamp_inplace!(
        x::AbstractVector{T},
        lo::AbstractVector,
        hi::AbstractVector,
    ) where {T<:Real}
        @inbounds for i in eachindex(x, lo, hi)
            xi = x[i]
            l = Float64(lo[i])
            h = Float64(hi[i])
            if xi < l
                x[i] = l
            elseif xi > h
                x[i] = h
            end
        end
        return x
    end

    if maxit <= 0
        if project_to_bounds && !isnothing(lower) && !isnothing(upper)
            clamp_inplace!(trial, lower, upper)
        end
        fmin = f(trial)
        return (x_opt = copy(trial), f_opt = fmin, fncount = 0, fail = 0)
    end

    if trace
        println("  Nelder-Mead direct search function minimizer")
    end

    if project_to_bounds && !isnothing(lower) && !isnothing(upper)
        clamp_inplace!(trial, lower, upper)
    end
    f_initial = f(trial)
    if !(isfinite(f_initial))
        error("function cannot be evaluated at initial parameters")
    end

    if trace
        println("function value for initial parameters = ", f_initial)
    end

    fncount = 1
    scaled_tol = reltol * (abs(f_initial) + reltol)
    if trace
        println("  Scaled convergence tolerance is ", scaled_tol)
    end

    n_vertices = n + 1

    # Initialize simplex state
    simplex = SimplexState(
        zeros(Float64, n, n_vertices),
        zeros(Float64, n_vertices),
        1,  # idx_best
        1   # idx_worst (will be updated)
    )
    centroid = zeros(Float64, n)

    simplex.vertices[:, 1] .= trial
    simplex.fvals[1] = f_initial

    # Compute initial step size (10% of largest coordinate magnitude, or 0.1)
    initial_step = 0.0
    @inbounds for i = 1:n
        initial_step = max(initial_step, 0.1 * abs(trial[i]))
    end
    if initial_step == 0.0
        initial_step = 0.1
    end
    if trace
        println("Stepsize computed as ", initial_step)
    end

    # Build remaining n vertices by perturbing each coordinate
    diameter = 0.0
    @inbounds for j = 2:n_vertices
        simplex.vertices[:, j] .= trial
        trystep = initial_step
        while simplex.vertices[j-1, j] == trial[j-1]
            simplex.vertices[j-1, j] = trial[j-1] + trystep

            if !isnothing(init_step_cap) && abs(trystep) > init_step_cap
                trystep = init_step_cap
                simplex.vertices[j-1, j] = trial[j-1] + trystep
            end

            if project_to_bounds && !isnothing(lower) && !isnothing(upper)
                lj = Float64(lower[j-1])
                uj = Float64(upper[j-1])
                v = simplex.vertices[j-1, j]
                if v < lj
                    simplex.vertices[j-1, j] = lj
                end
                if v > uj
                    simplex.vertices[j-1, j] = uj
                end
            end

            trystep *= 10.0

            if !isnothing(init_step_cap) && trystep > init_step_cap
                simplex.vertices[j-1, j] = nextfloat(trial[j-1])
                break
            end
        end
        diameter += trystep
    end
    prev_diameter = diameter
    needs_eval = true

    action = SIMPLEX_BUILD

    while fncount <= maxit
        if needs_eval
            # Evaluate function at all non-best vertices
            @inbounds for j = 1:n_vertices
                if j != simplex.idx_best
                    trial .= simplex.vertices[:, j]
                    if project_to_bounds && !isnothing(lower) && !isnothing(upper)
                        clamp_inplace!(trial, lower, upper)
                    end
                    fval = f(trial)
                    fval = isfinite(fval) ? fval : invalid_penalty
                    fncount += 1
                    simplex.fvals[j] = fval
                end
            end
            needs_eval = false
        end

        # Find best and worst vertices
        _simplex_update_extremes!(simplex, n)
        f_best = simplex.fvals[simplex.idx_best]
        f_worst = simplex.fvals[simplex.idx_worst]

        # Check convergence
        if (f_worst <= f_best + scaled_tol) || (f_best <= abstol)
            break
        end

        if trace
            println(rpad(_SIMPLEX_ACTION_NAMES[action], 15), lpad(string(fncount), 5), " ", f_worst, " ", f_best)
        end

        # Compute centroid of all vertices except worst
        _simplex_centroid!(centroid, simplex, n)

        # === Reflection ===
        _simplex_reflect!(trial, simplex, centroid, alpha, n)
        if project_to_bounds && !isnothing(lower) && !isnothing(upper)
            clamp_inplace!(trial, lower, upper)
        end
        f_reflect = f(trial)
        f_reflect = isfinite(f_reflect) ? f_reflect : invalid_penalty
        fncount += 1

        if f_reflect < f_best
            # Reflected point is best so far — try expansion
            # Save reflected point in centroid temporarily
            reflect_pt = copy(trial)
            f_reflect_saved = f_reflect

            _simplex_expand!(trial, reflect_pt, centroid, gamma, n)
            if project_to_bounds && !isnothing(lower) && !isnothing(upper)
                clamp_inplace!(trial, lower, upper)
            end
            f_expand = f(trial)
            f_expand = isfinite(f_expand) ? f_expand : invalid_penalty
            fncount += 1

            if f_expand < f_reflect_saved
                # === Accept expansion ===
                @inbounds simplex.vertices[:, simplex.idx_worst] .= trial
                simplex.fvals[simplex.idx_worst] = f_expand
                action = SIMPLEX_EXPAND
            else
                # === Accept reflection ===
                @inbounds simplex.vertices[:, simplex.idx_worst] .= reflect_pt
                simplex.fvals[simplex.idx_worst] = f_reflect_saved
                action = SIMPLEX_REFLECT
            end
        else
            # Reflected point is not better than best
            action = SIMPLEX_CONTRACT_INSIDE
            if f_reflect < f_worst
                # Accept reflection, then contract
                @inbounds simplex.vertices[:, simplex.idx_worst] .= trial
                simplex.fvals[simplex.idx_worst] = f_reflect
                action = SIMPLEX_CONTRACT_OUTSIDE
            end

            # === Contraction ===
            _simplex_contract!(trial, simplex, centroid, beta, n)
            if project_to_bounds && !isnothing(lower) && !isnothing(upper)
                clamp_inplace!(trial, lower, upper)
            end
            f_contract = f(trial)
            f_contract = isfinite(f_contract) ? f_contract : invalid_penalty
            fncount += 1

            if f_contract < simplex.fvals[simplex.idx_worst]
                @inbounds simplex.vertices[:, simplex.idx_worst] .= trial
                simplex.fvals[simplex.idx_worst] = f_contract
            else
                if f_reflect >= f_worst
                    # === Shrink ===
                    action = SIMPLEX_SHRINK
                    needs_eval = true
                    diameter = _simplex_shrink!(simplex, beta, n)
                    if diameter < prev_diameter
                        prev_diameter = diameter
                    else
                        if trace
                            println("Polytope size measure not decreased in shrink")
                        end
                        xopt = vec(copy(simplex.vertices[:, simplex.idx_best]))
                        return (x_opt = xopt, f_opt = simplex.fvals[simplex.idx_best], fncount = fncount, fail = 10)
                    end
                end
            end
        end
    end

    if trace
        println("Exiting from Nelder Mead minimizer")
        println("    ", fncount, " function evaluations used")
    end
    fmin = simplex.fvals[simplex.idx_best]
    xopt = vec(copy(simplex.vertices[:, simplex.idx_best]))
    failcode = (fncount > maxit) ? 1 : 0
    return (x_opt = xopt, f_opt = fmin, fncount = fncount, fail = failcode)
end
