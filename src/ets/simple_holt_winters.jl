function simple_holt_winters(x::AbstractArray, lenx::Int;
    alpha::Union{Nothing,Float64,Bool}=nothing,
    beta::Union{Nothing,Float64,Bool}=nothing,
    gamma::Union{Nothing,Float64,Bool}=nothing,
    phi::Union{Nothing,Float64,Bool}=nothing,
    seasonal::String="additive",
    m::Int,
    dotrend::Bool=false,
    doseasonal::Bool=false,
    exponential::Union{Nothing,Bool}=nothing,
    l_start::Union{Nothing,AbstractArray}=nothing,
    b_start::Union{Nothing,AbstractArray}=nothing,
    s_start::Union{Nothing,AbstractArray}=nothing)

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
    SSE = 0.0

    if !dotrend
        beta = 0.0
        b_start = 0.0
    end

    if !doseasonal
        gamma = 0.0
        s_start .= ifelse(seasonal == "additive", 0, 1)
    end

    lastlevel = copy(l_start)
    level0 = copy(l_start)
    lasttrend = copy(b_start)
    trend0 = copy(b_start)
    season0 = copy(s_start)

    for i in 1:lenx
        if i > 1
            lastlevel = level[i-1]
        end
        # Define b(t-1)
        if i > 1
            lasttrend = trend[i-1]
        end
        # Define s(t-m)
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
            lastseason = ifelse(seasonal == "additive", 0, 1)
        end

        if seasonal == "additive"
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
        SSE += res^2

        if seasonal == "additive"
            if !exponential
                level[i] = (alpha*(x[i]-lastseason).+(1-alpha)*(lastlevel.+phi*lasttrend))[1]
            else
                level[i] = (alpha*(x[i]-lastseason).+(1-alpha)*(lastlevel.*lasttrend .^ phi))[1]
            end
        else
            if !exponential
                level[i] = (alpha*(x[i]/lastseason).+(1-alpha)*(lastlevel.+phi*lasttrend))[1]
            else
                level[i] = (alpha.*(x[i]./lastseason).+(1-alpha).*(lastlevel.*lasttrend .^ phi))[1]
            end
        end

        if !exponential
            trend[i] = (beta.*(level[i].-lastlevel).+(1-beta).*phi.*lasttrend)[1]
        else
            trend[i] = (beta.*(level[i]./lastlevel).+(1-beta).*lasttrend .^ phi)[1]
        end

        if seasonal == "additive"
            if !exponential
                season[i] = (gamma*(x[i].-lastlevel.-phi*lasttrend).+(1-gamma)*lastseason)[1]
            else
                season[i] = (gamma*(x[i].-lastlevel.*lasttrend .^ phi).+(1-gamma)*lastseason)[1]
            end
        else
            if !exponential
                season[i] = (gamma*(x[i]./(lastlevel.+phi*lasttrend)).+(1-gamma)*lastseason)[1]
            else
                season[i] = (gamma.*(x[i]./(lastlevel.*lasttrend .^ phi)).+(1-gamma).*lastseason)[1]
            end
        end
    end

    return SimpleHoltWinters(SSE,
        xfit,
        residuals,
        [level0; level],
        [trend0; trend],
        [season0; season],
        phi
    )
end