export holt_winters_conventional

function construct_states(level::AbstractArray,
    trend::AbstractArray, season::AbstractArray,
    trendtype::String, seasontype::String, m::Int)

    states = hcat(level)
    state_names = ["l"]

    if trendtype != "N"
        states = hcat(states, trend)
        push!(state_names, "b")
    end

    if seasontype != "N"
        nr = size(states, 1)
        for i in 1:m

            seasonal_column = season[(m-i).+(1:nr)]
            states = hcat(states, seasonal_column)
        end

        seas_names = ["s$i" for i in 1:m]
        append!(state_names, seas_names)
    end

    initstates = states[1, :]

    return states, state_names, initstates
end

function holt_winters_conventional(x::AbstractArray, m::Int;
    alpha::Union{Nothing,Float64,Bool}=nothing,
    beta::Union{Nothing,Float64,Bool}=nothing,
    gamma::Union{Nothing,Float64,Bool}=nothing,
    phi::Union{Nothing,Float64,Bool}=nothing,
    seasonal::String="additive",
    exponential::Bool=false,
    lambda::Union{Nothing,Float64}=nothing,
    biasadj::Bool=false,
    warnings::Bool=true
)
    if !(seasonal in ["additive", "multiplicative"])
        throw(ArgumentError("Invalid seasonal component: must be 'additive' or 'multiplicative'."))
    end

    origx = copy(x)
    lenx = length(x)

    if (lambda == "auto") || (typeof(lambda) == Float64 && !isnothing(lambda))
        x, lambda = box_cox(x, m, lambda=lambda)
    end

    if isnothing(phi) || !(phi isa Number) || (phi isa Bool)
        phi = 1.0
    end

    if !isnothing(alpha) && !(alpha isa Number)
        throw(ArgumentError("Cannot fit models without level ('alpha' must not be 0 or false)."))
    end


    if !all(isnothing.([alpha, beta, gamma])) && any(x -> (x !== nothing && (x < 0 || x > 1)), [alpha, beta, gamma])
        throw(ArgumentError("'alpha', 'beta', and 'gamma' must be within the unit interval (0, 1)."))
    end

    if (isnothing(gamma) || gamma > 0)
        if seasonal == "multiplicative" && any(x .<= 0)
            throw(ArgumentError("Data must be positive for multiplicative Holt-Winters."))
        end
    end

    if m <= 1
        gamma = false
    end

    # Initialize l0, b0, s0
    if !isnothing(gamma) && gamma isa Bool && !gamma
        seasonal = "none"
        l_start = x[1]
        s_start = 0.0
        if isnothing(beta) || !(beta isa Bool) || beta
            if !exponential
                b_start = x[2] - x[1]
            else
                b_start = x[2] / x[1]
            end
        end
    else
        l_start = mean(x[1:m])
        b_start = (mean(x[m+1:m+m]) - l_start) / m
        if seasonal == "additive"
            s_start = x[1:m] .- l_start
        else
            s_start = x[1:m] ./ l_start
        end
    end

    if !@isdefined(b_start)
        b_start = [nothing]
    end

    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [1.0, 1.0, 1.0, 1.0]

    if !isnothing(beta) && isa(beta, Bool) && !beta
        trendtype = "N"
    elseif exponential
        trendtype = "M"
    else
        trendtype = "A"
    end

    if seasonal == "none"
        seasontype = "N"
    elseif seasonal == "multiplicative"
        seasontype = "M"
    else
        seasontype = "A"
    end

    optim_start = initparam(alpha, beta, gamma, 1.0, trendtype, seasontype, false, lower, upper, m, "usual")

    select = Int.(isnothing.([alpha, beta, gamma]))

    if !isa(s_start, AbstractArray)
        s_start = [s_start]
    end

    if !isa(b_start, AbstractArray)
        b_start = [b_start]
    end

    if !isa(l_start, AbstractArray)
        l_start = [l_start]
    end


    if sum(select) > 0
        starting_points = Float64[]

        if select[1] > 0
            alpha2 = optim_start["alpha"]
            push!(starting_points, alpha2)
        else
            alpha2 = alpha
        end

        if select[2] > 0
            beta2 = optim_start["beta"]
            push!(starting_points, beta2)
        else
            beta2 = beta
        end

        if select[3] > 0
            gamma2 = optim_start["gamma"]
            push!(starting_points, gamma2)
        else
            gamma2 = gamma
        end

        cal_opt_sse_closure = p -> calculate_opt_sse(p, select, x, lenx, alpha2, beta2, gamma2, seasonal, m, exponential, phi, l_start, b_start, s_start)
        bound_index = filter(i -> i > 0, select)
        sol = optimize(cal_opt_sse_closure, lower[bound_index], upper[bound_index], starting_points, Fminbox(LBFGS()))

        is_convergence = Optim.converged(sol)
        minimizers = Optim.minimizer(sol)
        iterations = Optim.iterations(sol)



        if !is_convergence | any((minimizers .< 0) .| (minimizers .> 1))
            if iterations > 50
                if warnings
                    @warn "Optimization difficulties!"
                end
            else
                error("Optimization failure")
            end
        end


        if select[1] > 0
            alpha = minimizers[1]
        end
        if select[2] > 0
            beta = minimizers[1+select[1]]
        end
        if select[3] > 0
            gamma = minimizers[1+select[1]+select[2]]
        end
    end

    final_fit = simple_holt_winters(x, lenx,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        phi=phi,
        seasonal=seasonal,
        m=m,
        dotrend=(!isa(beta, Bool) || beta),
        doseasonal=(!isa(gamma, Bool) || gamma),
        l_start=l_start,
        exponential=exponential,
        b_start=b_start,
        s_start=s_start)

    res = final_fit.residuals

    fitted = final_fit.fitted
    if !isnothing(lambda)
        fitted = inv_box_cox(fitted, lambda=lambda, biasadj=biasadj, fvar=var(res))
    end

    states, state_names, initstates = construct_states(final_fit.level, final_fit.trend,
        final_fit.season, trendtype, seasontype, m)

    damped = phi < 1.0

    if seasonal == "additive"
        components = ["A", trendtype, seasontype, damped]
    elseif seasonal == "multiplicative"
        components = ["M", trendtype, seasontype, damped]
    elseif seasonal == "none" && exponential
        components = ["M", trendtype, seasontype, damped]
    else
        components = ["A", trendtype, seasontype, damped]
    end

    param = Dict("alpha" => alpha)
    param["phi"] = NaN

    if trendtype != "N"
        param["beta"] = beta
    end

    if seasontype != "N"
        param["gamma"] = gamma
    end
    if damped
        param["phi"] = phi
    end

    # Calculate sigma2 based on components
    if components[1] == "A"
        sigma2 = mean(res .^ 2)
    else
        sigma2 = mean((res ./ fitted) .^ 2)
    end

    out = HoltWintersConventional(
        fitted,
        res,
        components,
        origx,
        merge(param, Dict("initstate" => initstates)),
        initstates,
        states,
        state_names,
        final_fit.SSE,
        sigma2,
        m,
        lambda,
        biasadj
    )

    return (out)
end