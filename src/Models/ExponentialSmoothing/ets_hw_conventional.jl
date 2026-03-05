function simple_holt_winters(
    x::AbstractArray,
    lenx::Int;
    alpha::Union{Nothing,Float64,Bool} = nothing,
    beta::Union{Nothing,Float64,Bool} = nothing,
    gamma::Union{Nothing,Float64,Bool} = nothing,
    phi::Union{Nothing,Float64,Bool} = nothing,
    seasonal::Symbol = :additive,
    m::Int,
    dotrend::Bool = false,
    doseasonal::Bool = false,
    exponential::Union{Nothing,Bool} = nothing,
    l_start::Union{Nothing,AbstractArray} = nothing,
    b_start::Union{Nothing,AbstractArray} = nothing,
    s_start::Union{Nothing,AbstractArray} = nothing,
)

    if exponential != true || isnothing(exponential)
        exponential = false
    end

    if isnothing(phi) || !isa(phi, Number)
        phi = 1
    end

    # Initialise arrays
    level = zeros(Float64, lenx)
    trend = zeros(Float64, lenx)
    season = zeros(Float64, lenx)
    xfit = zeros(Float64, lenx)
    residuals = zeros(Float64, lenx)
    sse_val = 0.0

    if !dotrend
        beta = 0.0
        b_start = 0.0
    end

    if !doseasonal
        gamma = 0.0
        s_start .= ifelse(seasonal === :additive, 0, 1)
    end

    lastlevel = copy(l_start)
    level0 = copy(l_start)
    lasttrend = copy(b_start)
    trend0 = copy(b_start)
    season0 = copy(s_start)

    @inbounds for i = 1:lenx
        if i > 1
            lastlevel = level[i-1]
        end

        if i > 1
            lasttrend = trend[i-1]
        end

        if i > m
            lastseason = season[i-m]
        else
            if i <= length(season0)
                lastseason = season0[i]
            else
                lastseason = nothing
            end
        end

        if isnothing(lastseason)
            lastseason = ifelse(seasonal === :additive, 0, 1)
        end

        if seasonal === :additive
            if !exponential
                xhat = lastlevel .+ phi .* lasttrend .+ lastseason
            else
                xhat = lastlevel .* lasttrend .^ phi .+ lastseason
            end

        else
            if !exponential
                xhat = (lastlevel .+ phi .* lasttrend) .* lastseason
            else
                xhat = lastlevel .* lasttrend .^ phi .* lastseason
            end

        end

        xfit[i] = xhat[1]
        res = x[i] - xhat[1]
        residuals[i] = res
        sse_val += res^2

        if seasonal === :additive
            if !exponential
                level[i] =
                    (alpha*(x[i]-lastseason).+(1-alpha)*(lastlevel.+phi*lasttrend))[1]
            else
                level[i] =
                    (alpha*(x[i]-lastseason).+(1-alpha)*(lastlevel.*lasttrend .^ phi))[1]
            end
        else
            if !exponential
                level[i] =
                    (alpha*(x[i]/lastseason).+(1-alpha)*(lastlevel.+phi*lasttrend))[1]
            else
                level[i] =
                    (alpha.*(x[i]./lastseason).+(1-alpha).*(lastlevel.*lasttrend .^ phi))[1]
            end
        end

        if !exponential
            trend[i] = (beta.*(level[i].-lastlevel).+(1-beta).*phi.*lasttrend)[1]
        else
            trend[i] = (beta.*(level[i]./lastlevel).+(1-beta).*lasttrend .^ phi)[1]
        end

        if seasonal === :additive
            if !exponential
                season[i] =
                    (gamma*(x[i].-lastlevel.-phi*lasttrend).+(1-gamma)*lastseason)[1]
            else
                season[i] =
                    (gamma*(x[i].-lastlevel.*lasttrend .^ phi).+(1-gamma)*lastseason)[1]
            end
        else
            if !exponential
                season[i] =
                    (gamma*(x[i]./(lastlevel.+phi*lasttrend)).+(1-gamma)*lastseason)[1]
            else
                season[i] =
                    (gamma.*(x[i]./(lastlevel.*lasttrend .^ phi)).+(1-gamma).*lastseason)[1]
            end
        end
    end

    return SimpleHoltWinters(
        sse_val,
        xfit,
        residuals,
        [level0; level],
        [trend0; trend],
        [season0; season],
        phi,
    )
end

function calculate_opt_sse(
    p,
    select::AbstractArray,
    x::AbstractArray,
    lenx::Int,
    alpha::Union{Nothing,Float64,Bool},
    beta::Union{Nothing,Float64,Bool},
    gamma::Union{Nothing,Float64,Bool},
    seasonal::Symbol,
    m::Int,
    exponential::Union{Nothing,Bool},
    phi::Union{Nothing,Float64},
    l_start::Union{Nothing,AbstractArray},
    b_start::Union{Nothing,AbstractArray},
    s_start::Union{Nothing,AbstractArray},
)

    if select[1] > 0
        alpha = p[1]
    end
    if select[2] > 0
        beta = p[1+select[1]]
    end
    if select[3] > 0
        gamma = p[1+select[1]+select[2]]
    end

    dotrend = (!isa(beta, Bool) || beta)
    doseasonal = (!isa(gamma, Bool) || gamma)

    out = simple_holt_winters(
        x,
        lenx,
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        phi = phi,
        seasonal = seasonal,
        m = m,
        dotrend = dotrend,
        doseasonal = doseasonal,
        l_start = l_start,
        exponential = exponential,
        b_start = b_start,
        s_start = s_start,
    )

    return out.sse
end

function holt_winters_conventional(
    x::AbstractArray,
    m::Int;
    alpha::Union{Nothing,Float64,Bool} = nothing,
    beta::Union{Nothing,Float64,Bool} = nothing,
    gamma::Union{Nothing,Float64,Bool} = nothing,
    phi::Union{Nothing,Float64,Bool} = nothing,
    seasonal::Symbol = :additive,
    exponential::Bool = false,
    lambda::Union{Nothing,Float64} = nothing,
    biasadj::Bool = false,
    warnings::Bool = true,
    options::NelderMeadOptions
)
    if !(seasonal in (:additive, :multiplicative))
        throw(
            ArgumentError(
                "Invalid seasonal component: must be :additive or :multiplicative.",
            ),
        )
    end

    origx = copy(x)
    lenx = length(x)

    if (lambda === :auto) || (typeof(lambda) == Float64 && !isnothing(lambda))
        x, lambda = box_cox(x, m, lambda = lambda)
    end

    if isnothing(phi) || !(phi isa Number) || (phi isa Bool)
        phi = 1.0
    end

    if !isnothing(alpha) && !(alpha isa Number)
        throw(
            ArgumentError(
                "Cannot fit models without level ('alpha' must not be 0 or false).",
            ),
        )
    end


    if !all(isnothing.([alpha, beta, gamma])) &&
       any(x -> (!isnothing(x) && (x < 0 || x > 1)), [alpha, beta, gamma])
        throw(
            ArgumentError(
                "'alpha', 'beta', and 'gamma' must be within the unit interval (0, 1).",
            ),
        )
    end

    if (isnothing(gamma) || gamma > 0)
        if seasonal === :multiplicative && any(x .<= 0)
            throw(ArgumentError("Data must be positive for multiplicative Holt-Winters."))
        end
    end

    if m <= 1
        gamma = false
    end

    # Initialize l0, b0, s0
    if !isnothing(gamma) && gamma isa Bool && !gamma
        seasonal = :none
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
        if seasonal === :additive
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

    if seasonal === :none
        seasontype = "N"
    elseif seasonal === :multiplicative
        seasontype = "M"
    else
        seasontype = "A"
    end

    optim_start = initparam(
        alpha,
        beta,
        gamma,
        1.0,
        trendtype,
        seasontype,
        false,
        lower,
        upper,
        m,
        :usual,
    )

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
            push!(starting_points, optim_start.alpha)
        end

        if select[2] > 0
            push!(starting_points, optim_start.beta)
        end

        if select[3] > 0
            push!(starting_points, optim_start.gamma)
        end

        parscale = max.(abs.(starting_points), 0.1)
        
        cal_opt_sse_closure =
            p -> calculate_opt_sse(
                descaler(p, parscale),
                select,
                x,
                lenx,
                alpha,
                beta,
                gamma,
                seasonal,
                m,
                exponential,
                phi,
                l_start,
                b_start,
                s_start,
            )

        sol = nelder_mead(cal_opt_sse_closure, scaler(starting_points, parscale), options)

        is_convergence = sol.fail == 0
        minimizers = descaler(sol.x_opt, parscale)

        if (!is_convergence || any((minimizers .< 0) .| (minimizers .> 1))) && warnings
            if sol.fail in [1, 10]
                @warn "Optimization difficulties: convergence code $(sol.fail)"
            elseif sol.fail ∉ [0, 1, 10]
                @warn "Optimization failure: convergence code $(sol.fail), using best parameters found"
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

    final_fit = simple_holt_winters(
        x,
        lenx,
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        phi = phi,
        seasonal = seasonal,
        m = m,
        dotrend = (!isa(beta, Bool) || beta),
        doseasonal = (!isa(gamma, Bool) || gamma),
        l_start = l_start,
        exponential = exponential,
        b_start = b_start,
        s_start = s_start,
    )

    res = final_fit.residuals

    fitted = final_fit.fitted
    if !isnothing(lambda)
        fitted = inv_box_cox(fitted, lambda = lambda, biasadj = biasadj, fvar = var(res))
    end

    states, state_names, initstates = construct_states(
        final_fit.level,
        final_fit.trend,
        final_fit.season,
        trendtype,
        seasontype,
        m,
    )

    damped = phi < 1.0

    if seasonal === :additive
        components = ["A", trendtype, seasontype, string(damped)]
    elseif seasonal === :multiplicative
        components = ["M", trendtype, seasontype, string(damped)]
    elseif seasonal === :none && exponential
        components = ["M", trendtype, seasontype, string(damped)]
    else
        components = ["A", trendtype, seasontype, string(damped)]
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

    method_parts = []
    if exponential && trendtype != "N"
        push!(method_parts, "Holt's method with exponential trend")
    elseif trendtype != "N"
        push!(method_parts, "Holt's method")
    else
        push!(method_parts, "Simple exponential smoothing")
    end

    if damped
        push!(method_parts, "Damped")
    end

    if seasonal === :additive
        push!(method_parts, "additive seasonality")
    elseif seasonal === :multiplicative
        push!(method_parts, "multiplicative seasonality")
    end

    method = join(method_parts, " with ")

    out = HoltWintersConventional(
        fitted,
        res,
        components,
        origx,
        merge(param, Dict("initstate" => initstates)),
        initstates,
        states,
        state_names,
        final_fit.sse,
        sigma2,
        m,
        lambda,
        biasadj,
        method,
    )

    return (out)
end
