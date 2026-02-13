"""
	NelderMeadOptions(; abstol=-Inf, intol=sqrt(eps(Float64)), alpha=1.0, beta=0.5,
                        gamma=2.0, trace=false, maxit=500, invalid_penalty=1e35, 
                        project_to_bounds=false, lower=nothing, upper=nothing, 
                        init_step_cap=nothing)

A configuration container for the Nelder-Mead optimization algorithm.

# Keyword Arguments

- `abstol::Float64`: Absolute tolerance on the function value for stopping. Default is `-Inf` (matching R's `optim()`).
- `intol::Float64`: Relative tolerance between the best and worst function values. Default is `sqrt(eps(Float64))`.
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

# Example

```julia
opts = NelderMeadOptions(abstol=1e-4, maxit=100, trace=true)

# With bounds
opts = NelderMeadOptions(project_to_bounds=true, lower=[0.0, 0.0], upper=[10.0, 10.0])
```
"""
struct NelderMeadOptions
    abstol::Float64
    intol::Float64
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
    intol = sqrt(eps(Float64)),
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
    intol,
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
	nmmin(f, x0, options::NelderMeadOptions)

Nelder-Mead simplex minimization - a line-for-line port of the C `nmmin` from R's `stats::optim`.

This implementation exactly reproduces the C code behavior from Nash (1990) / R source, including:
- Initial simplex construction with ×10 step escalation
- Reflection/expansion/contraction/shrink steps with identical formulas
- Convergence checks matching C tolerances
- Shrink-failure handling
- Contraction/shrink logic: shrink **only** when reflection fails to improve worst AND contraction fails
- Polytope size accumulation matching C's loop structure

It also offers optional, *off-by-default* bounds-aware extensions (project_to_bounds, init_step_cap).

# Arguments

- `f::Function`: Objective function mapping `Vector{Float64}` → `Real`. Should return a finite
  value for valid parameters. If `f` returns non-finite during iterations, it is replaced by
  `invalid_penalty` (see below). If the **initial** evaluation at `x0` is non-finite, the
  routine returns immediately with `fail === true` (no simplex constructed).
- `x0::AbstractVector{<:Real}`: Starting point. Internally converted to `Vector{Float64}`.
- `options::NelderMeadOptions`: Control parameters for the algorithm (see `NelderMeadOptions` documentation).

# Options (via `NelderMeadOptions`)

- `abstol::Float64 = 0.0`: Absolute tolerance on `VL` (best value). `VL ≤ abstol` triggers early exit.
  (Set `0.0` for strict Algorithm 19 behavior.)
- `intol::Float64 = sqrt(eps(Float64))`: **Scaled/relative tolerance** used by Algorithm 19:
  `convtol = intol * (abs(VL) + intol)`. Converged if `VH ≤ VL + convtol`.
- `alpha::Float64 = 1.0`: Reflection coefficient.
- `beta::Float64  = 0.5`: Contraction and **shrink** coefficient (Alg. 19 uses the same `β` for both).
- `gamma::Float64 = 2.0`: Expansion coefficient.
- `trace::Bool = false`: If `true`, prints Algorithm-19 style iteration logs, e.g.:
  `REFLECTION         123 1234.56 1200.78` and exit summary.
- `maxit::Int = 500`: Maximum number of **function evaluations**.
- `invalid_penalty::Float64 = 1e35`: Finite surrogate for non-finite objective values during iterations.
  (Matches the book's "BIG".)
- `project_to_bounds::Bool = false`: If `true`, clamp each trial point into `[lower, upper]` before
  calling `f`. Useful for tight box constraints in practice; **disabled** by default to preserve
  strict Algorithm-19 behavior.
- `lower, upper = nothing`: Vectors of the same length as `x0` giving box bounds (used only when
  `project_to_bounds=true`).
- `init_step_cap::Union{Nothing,Float64} = nothing`: Optional cap on the ×10 step escalation used
  to build the initial simplex. If hitting the cap still fails to change a coordinate in FP, the
  algorithm nudges with `nextfloat(x0[i])` as a last resort.

# Returns

A named tuple `(x_opt, f_opt, fncount, fail)` where:
- `x_opt::Vector{Float64}`: Best parameters found (column `L` of the final simplex).
- `f_opt::Float64`: Objective value at `x_opt`.
- `fncount::Int`: Total number of objective evaluations used.
- `fail`: Status code:
    * `0` — Converged (scaled tolerance and/or `abstol` satisfied).
    * `1` — Exceeded `maxit` function evaluations.
    * `10` — Shrink step failed to reduce the simplex size (degenerate case).
    * `true` — Initial evaluation at `x0` was non-finite; no simplex could be formed.

# Algorithmic details (matches Nash, Alg. 19)

- **Initial simplex**: Start from `x0`, create `n` additional vertices by adding a per-coordinate step
  of size `max(0.1*abs(x0[i]), 0.1)`, escalating the step by ×10 until the floating-point value of the
  coordinate actually changes (robust at large |x0|).
- **Centroid**: Average all vertices except the current worst `H`.
- **Moves**: Reflection → if improved over best, try **expansion**; otherwise accept reflection or perform
  **lo-/hi-reduction** (contraction); if still no improvement, **shrink** about the best by factor `β`.
- **Convergence**: Stop if `VH ≤ VL + convtol` with `convtol = intol*(abs(VL)+intol)` (or `VL ≤ abstol`).
- **Shrink failure**: If the shrink does not decrease a simple polytope size measure, stop with `fail = 10`.

# Notes

- To exactly reproduce R's defaults, use `abstol=-Inf`, `intol=sqrt(eps(Float64))`, `alpha=1.0`, `beta=0.5`,
  `gamma=2.0`, `trace=false`, `maxit=500`.
- `project_to_bounds`, `lower/upper`, and `init_step_cap` are practical extensions for bounded problems
  and do not alter Algorithm-19 unless enabled.
- For likelihood problems, ensure the objective's **scale** (SSE vs MSE vs -loglik) matches downstream IC
  formulas; otherwise AIC/BIC may appear inflated.

# References:
- Nash, J.C. (1990). *Compact Numerical Methods for Computers*, Algorithm 19.

# See also:
- R `stats::optim` with `method="Nelder-Mead"` (derived from Nash's Pascal code).

# Examples

```julia
using Durbyn.Optimize

# Unconstrained Rosenbrock from x0 = [-1.2, 1.0]
rosen(x) = (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2
opts = NelderMeadOptions(trace=true, maxit=2000, intol=sqrt(eps(Float64)), abstol=-Inf)
res = nmmin(rosen, [-1.2, 1.0], opts)
@show res.x_opt res.f_opt res.fncount res.fail

# Bounded quadratic with box constraints
f(x) = (x[1]-0.8)^2 + (x[2]-0.2)^2
lo = [0.0, 0.0]; hi = [1.0, 1.0]
opts_bounded = NelderMeadOptions(
    project_to_bounds=true, lower=lo, upper=hi,
    init_step_cap=0.25, trace=true)

resb = nmmin(f, [0.3, 0.3], opts_bounded)
```
"""
function nmmin(f::Function, x0::AbstractVector{<:Real}, options::NelderMeadOptions)
    abstol = options.abstol
    intol = options.intol
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
    B = collect(float.(x0))

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
            clamp_inplace!(B, lower, upper)
        end
        fmin = f(B)
        return (x_opt = copy(B), f_opt = fmin, fncount = 0, fail = 0)
    end

    if trace
        println("  Nelder-Mead direct search function minimizer")
    end

    if project_to_bounds && !isnothing(lower) && !isnothing(upper)
        clamp_inplace!(B, lower, upper)
    end
    fB = f(B)
    if !(isfinite(fB))
        error("function cannot be evaluated at initial parameters")
    end

    if trace
        println("function value for initial parameters = ", fB)
    end

    fncount = 1
    convtol = intol * (abs(fB) + intol)
    if trace
        println("  Scaled convergence tolerance is ", convtol)
    end

    n1 = n + 1
    C = n + 2  

    P = [zeros(Float64, n, n1 + 1); zeros(Float64, 1, n1 + 1)]

    P[1:n, 1] .= B
    P[end, 1] = fB
    L = 1 

    step = 0.0
    @inbounds for i = 1:n
        step = max(step, 0.1 * abs(B[i]))
    end
    if step == 0.0
        step = 0.1
    end
    if trace
        println("Stepsize computed as ", step)
    end

    size = 0.0
    @inbounds for j = 2:n1
        P[1:n, j] .= B
        trystep = step
        while P[j-1, j] == B[j-1]
            P[j-1, j] = B[j-1] + trystep

            if !isnothing(init_step_cap) && abs(trystep) > init_step_cap
                trystep = init_step_cap
                P[j-1, j] = B[j-1] + trystep
            end

            if project_to_bounds && !isnothing(lower) && !isnothing(upper)
                lj = Float64(lower[j-1])
                uj = Float64(upper[j-1])
                v = P[j-1, j]
                if v < lj
                    P[j-1, j] = lj
                end
                if v > uj
                    P[j-1, j] = uj
                end
            end

            trystep *= 10.0

            if !isnothing(init_step_cap) && trystep > init_step_cap
                P[j-1, j] = nextfloat(B[j-1])
                break
            end
        end
        size += trystep
    end
    oldsize = size
    calcvert = true

    action = "BUILD          "

    while fncount <= maxit
        if calcvert
            
            @inbounds for j = 1:n1
                if j != L
                    B .= P[1:n, j]
                    if project_to_bounds && !isnothing(lower) && !isnothing(upper)
                        clamp_inplace!(B, lower, upper)
                    end
                    fval = f(B)
                    fval = isfinite(fval) ? fval : invalid_penalty
                    fncount += 1
                    P[end, j] = fval
                end
            end
            calcvert = false
        end

        VL = P[end, L]
        VH = VL
        H = L
        @inbounds for j = 1:n1
            if j != L
                fval = P[end, j]
                if fval < VL
                    VL = fval
                    L = j
                end
                if fval > VH
                    VH = fval
                    H = j
                end
            end
        end

        if (VH <= VL + convtol) || (VL <= abstol)
            break
        end

        if trace
            println(action, lpad(string(fncount), 5), " ", VH, " ", VL)
        end

        @inbounds for i = 1:n
            temp = -P[i, H]
            for j = 1:n1
                temp += P[i, j]
            end
            P[i, C] = temp / n
        end
        @inbounds for i = 1:n
            B[i] = (1 + alpha) * P[i, C] - alpha * P[i, H]
        end
        if project_to_bounds && !isnothing(lower) && !isnothing(upper)
            clamp_inplace!(B, lower, upper)
        end
        fR = f(B)
        fR = isfinite(fR) ? fR : invalid_penalty
        fncount += 1
        VR = fR
        if VR < VL
            P[end, C] = fR
            @inbounds for i = 1:n
                tmp = gamma * B[i] + (1 - gamma) * P[i, C]
                P[i, C] = B[i]
                B[i] = tmp
            end
            if project_to_bounds && !isnothing(lower) && !isnothing(upper)
                clamp_inplace!(B, lower, upper)
            end
            fE = f(B)
            fE = isfinite(fE) ? fE : invalid_penalty
            fncount += 1
            if fE < VR
                @inbounds P[1:n, H] .= B
                P[end, H] = fE
                action = "EXTENSION      "
            else
                @inbounds P[1:n, H] .= P[1:n, C]
                P[end, H] = VR
                action = "REFLECTION     "
            end
        else
            action = "HI-REDUCTION   "
            if VR < VH
                @inbounds P[1:n, H] .= B
                P[end, H] = VR
                action = "LO-REDUCTION   "
            end

            # Contraction step
            @inbounds for i = 1:n
                B[i] = (1 - beta) * P[i, H] + beta * P[i, C]
            end
            if project_to_bounds && !isnothing(lower) && !isnothing(upper)
                clamp_inplace!(B, lower, upper)
            end
            fC = f(B)
            fC = isfinite(fC) ? fC : invalid_penalty
            fncount += 1

            if fC < P[end, H]
                @inbounds P[1:n, H] .= B
                P[end, H] = fC
            else
                if VR >= VH
                    action = "SHRINK         "
                    calcvert = true
                    size = 0.0
                    @inbounds for j = 1:n1
                        if j != L
                            for i = 1:n
                                P[i, j] = beta * (P[i, j] - P[i, L]) + P[i, L]
                                size += abs(P[i, j] - P[i, L])
                            end
                        end
                    end
                    if size < oldsize
                        oldsize = size
                    else
                        if trace
                            println("Polytope size measure not decreased in shrink")
                        end
                        xopt = vec(copy(P[1:n, L]))
                        return (x_opt = xopt, f_opt = P[end, L], fncount = fncount, fail = 10)
                    end
                end
            end
        end
    end

    if trace
        println("Exiting from Nelder Mead minimizer")
        println("    ", fncount, " function evaluations used")
    end
    fmin = P[end, L]
    xopt = vec(copy(P[1:n, L]))
    failcode = (fncount > maxit) ? 1 : 0
    return (x_opt = xopt, f_opt = fmin, fncount = fncount, fail = failcode)
end