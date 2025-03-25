function nmmin(f::Function, x0::Vector{Float64};
               abstol=1e-8, intol=1e-8,
               alpha=1.0, beta=0.5, gamma=2.0,
               trace=false, maxit=1000)

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
        simplex[j - 1, j] += step
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
        centroid = sum(simplex[:, 1:n], dims=2)[:, 1] / n

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
