
function crost_cost(params, x, cost_metric, method, fixed_weights, num_params, optimize_weights, 
    initial_values, optimize_inits, lower_bounds, upper_bounds)
    forecast_input = nothing

    if optimize_weights
        weights = params[1:num_params]
        inits = optimize_inits ? params[(num_params+1):(num_params+2)] : initial_values
    else
        weights = fixed_weights
        inits = params
    end

    forecast_result = crost_not_optimized(x, 0, collect(weights), inits, method, false)
    forecast_input = forecast_result["frc_in"]

    error = evaluation_metrics(x, forecast_input)[cost_metric]

    total_param_count = num_params * optimize_weights + 2 * optimize_inits
    for i in 1:total_param_count
        if params[i] < lower_bounds[i] || params[i] > upper_bounds[i]
            return 9e99
        end
    end

    return error
end

function crost_opt(x, method, cost, w, nop, init, init_opt)

    nzd = findall(xi -> xi != 0, x)
    k = length(nzd)
    intervals = [nzd[1], nzd[2:k] .- nzd[1:(k-1)]...]

    if isnothing(w) && !init_opt
        p0 = fill(0.05, nop)
        lbound = fill(0, nop)
        ubound = fill(1, nop)

        if nop != 1
            wopt = optimize(p -> crost_cost(p, x, cost, method, w,
                nop, true, init, init_opt, lbound, ubound),
                p0, NelderMead(), Options(iterations = 2000)).minimizer
        else
            wopt = optimize(p -> crost_cost([p], x, cost, method, w,
                nop, true, init, init_opt, lbound, ubound),
                lbound[1], ubound[1], Brent()).minimizer
        end

        wopt = [wopt..., init...]

    elseif isnothing(w) && init_opt
        p0 = [fill(0.05, nop)..., init[1], init[2]]
        lbound = [fill(0, nop)..., 0, 1]
        ubound = [fill(1, nop)..., maximum(x), minimum(intervals)]

        wopt = optimize(p -> crost_cost(p, x, cost, method, w,
            nop, true, init, true, lbound, ubound),
            p0, NelderMead(), Options(iterations = 2000)).minimizer

    elseif !isnothing(w) && init_opt
        nop = length(w)
        p0 = [init[1], init[2]]
        lbound = [0, 1]
        ubound = [maximum(x), maximum(intervals)]

        wopt = optimize(p -> crost_cost(p, x, cost, method, w,
            nop, false, init, true, lbound, ubound),
            p0, NelderMead(), Options(iterations = 2000)).minimizer

        wopt = [wopt..., init...]
    end

    return Dict("w" => wopt[1:nop], "init" => wopt[(nop+1):(nop+2)])
end

function crost_optimized(x, h, init, nop, method, cost, init_opt, na_rm)
    
    if !(nop in [1, 2])
        @warn "nop can be either 1 or 2. Overriden to 2."
        nop = 2
    end

    if na_rm
        x = filter(!ismissing, x)
    end

    @assert sum(x .!= 0) >= 2 "I need at least two non-zero values to model your time series."
    nzd = findall(xi -> xi != 0, x)
    z = x[nzd]
    k = length(nzd)
    intervals = [nzd[1], nzd[2:k] .- nzd[1:(k-1)]...]

    if !(isa(init, Array))
        init = init == "mean" ? [z[1], mean(intervals)] : [z[1], intervals[1]]
    end

    wopt = crost_opt(x, method, cost, nothing, nop, init, init_opt)
    w = wopt["w"]
    init = wopt["init"]

    if na_rm 
        na_rm = false
    end

    out = crost_not_optimized(x, h, w, init, method, na_rm)
    return(out)
end

function crost_not_optimized(x,h,w,init,method,na_rm)

    if na_rm
        x = filter(!ismissing, x)
    end
    n = length(x)

    @assert sum(x .!= 0) >= 2 "I need at least two non-zero values to model your time series."
    nzd = findall(xi -> xi != 0, x)
    k = length(nzd)
    z = x[nzd]
    intervals = [nzd[1], nzd[2:k] .- nzd[1:(k-1)]...]

    if !(isa(init, Array))
        init = init == "mean" ? [z[1], mean(intervals)] : [z[1], intervals[1]]
    end

    w = convert(Vector{Float64}, w)
    init = convert(Vector{Float64}, init)

    zfit = zeros(k)
    xfit = zeros(k)
    zfit[1] = init[1]
    xfit[1] = init[2]

    a_demand, a_interval = length(w) == 1 ? (w[1], w[1]) : (w[1], w[2])

    coeff = method == "sba" ? 1 - (a_interval / 2) :
            method == "sbj" ? 1 - a_interval / (2 - a_interval) :
            1

    for i in 2:k
        zfit[i] = zfit[i-1] + a_demand * (z[i] - zfit[i-1])
        xfit[i] = xfit[i-1] + a_interval * (intervals[i] - xfit[i-1])
    end

    cc = coeff .* zfit ./ xfit
    frc_in = zeros(n)
    x_in = zeros(n)
    z_in = zeros(n)
    tv = [nzd .+ 1..., n]

    for i in 1:k
        idx_range = tv[i]:min(tv[i+1], n)
        frc_in[idx_range] .= cc[i]
        x_in[idx_range] .= xfit[i]
        z_in[idx_range] .= zfit[i]
    end

    if h > 0
        frc_out = fill(cc[k], h)
        x_out = fill(xfit[k], h)
        z_out = fill(zfit[k], h)
    else
        frc_out = nothing
        x_out = nothing
        z_out = nothing
    end

    w = length(w) == 1 ? [w, w] : w

    c_in = Dict("Demand" => z_in, "Interval" => x_in)
    c_out = h > 0 ? Dict("Demand" => z_out, "Interval" => x_out) : NaN

    comp = Dict("c_in" => c_in, "c_out" => c_out, "coeff" => coeff)
    initial = [zfit[1], xfit[1]]

    return Dict(
        "data" => x,
        "method" => method,
        "frc_in" => frc_in,
        "frc_out" => frc_out,
        "weights" => w,
        "initial" => initial,
        "components" => comp
    )
end

function crost(x, h, w,init,nop,method,cost,init_opt,na_rm)

    method = match_arg(method, ["croston", "sba", "sbj"])
    init = match_arg(init, ["naive", "mean"])
    cost = match_arg(cost, ["mar", "msr", "mae", "mse"])

    out = if isnothing(w) || init_opt
        crost_optimized(x, h, init, nop, method, cost, init_opt, na_rm)
    else
        crost_not_optimized(x, h, w, init, method, na_rm)
    end

    res = out["data"] - out["frc_in"]

    return Dict("x" => out["data"], "fitted" => out["frc_in"],
     "residuals" => res, "weights" => out["weights"], 
     "initial" => out["initial"],  "components" => out["components"]), "mean" => out["frc_out"]
end