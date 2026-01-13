function analyze_series(x::AbstractVector)
    miss = ismissing.(x)
    first = findfirst(!, miss)
    last = findlast(!, miss)

    if isnothing(first)
        return (firstnonmiss = nothing, serieslength = 0, x_trim = x)
    end

    serieslength = count(!, @view miss[first:last])
    x_trim = x[first:end]   # trim leading missings only

    (firstnonmiss = first, serieslength = serieslength, x_trim = x_trim)
end

function compute_approx_offset(;
    approximation::Bool,
    x::AbstractVector,
    d::Int,
    D::Int,
    m::Int = 1,
    xreg::Union{NamedMatrix,Nothing} = nothing,              # matrix/vector or nothing
    truncate::Union{Int,Nothing} = nothing,
    kwargs... 
    )
    # no approximation -> zero offset
    if !approximation
        #return (offset = 0.0, fit = nothing)
        return 0.0
    end


    xx = x
    Xreg = xreg
    N0 = length(xx)

    # truncate tail of x (and xreg if it aligns with x)
    if !isnothing(truncate) && N0 > truncate
        start_idx = N0 - truncate + 1
        xx = collect(@view xx[start_idx:end])

        if !isnothing(Xreg)
            # row count of xreg
            nrows = size(Xreg.data, 1)
            if nrows == N0
                Xreg = select_rows(Xreg, start_idx:nrows)
            end
        end
    end

    serieslength = length(xx)

    # quick ARIMA fit and offset
    try
        fit = if D == 0
            arima(xx, m; order = PDQ(0, d, 0), seasonal = PDQ(0, 0, 0), xreg = Xreg, kwargs...)
        else
            arima(xx, m; order = PDQ(0, d, 0), seasonal = PDQ(0, D, 0), xreg = Xreg, kwargs...)
        end

        loglik = fit.loglik
        sigma2 = fit.sigma2
        offset = -2 * loglik - serieslength * log(sigma2)
        #return (offset=offset, fit=fit)
        return offset
    catch
        # mirrors try-error fallback: offset <- 0
        #return (offset=0.0, fit=nothing)
        return 0.0
    end
end


function newmodel(p::Int, d::Int, q::Int, P::Int, D::Int, Q::Int, constant::Bool, results::Matrix, k::Int)
    c_int = constant ? 1 : 0
    @inbounds for i in 1:k
        # Quick check: if any component is missing, skip
        if ismissing(results[i, 1])
            continue
        end
        # Direct comparison without allocating Tuple or array
        if results[i, 1] == p && results[i, 2] == d && results[i, 3] == q &&
           results[i, 4] == P && results[i, 5] == D && results[i, 6] == Q &&
           results[i, 7] == c_int
            return false
        end
    end
    return true
end

get_pdq(x) = hasproperty(x, :p) ? (x.p, x.d, x.q) : (x[1], x[2], x[3])
get_sum(x) = hasproperty(x, :p) ? x.p + x.d + x.q : sum(x)

function arima_trace_str(order::PDQ, seasonal::PDQ, m::Int, constant::Bool; ic_value::Real = Inf)
    (p, d, q) = get_pdq(order)
    (P, D, Q) = get_pdq(seasonal)
    seasonal_part = get_sum(seasonal) > 0 && m > 0 ? "($P,$D,$Q)[$m]" : ""
    if constant && (d + D == 0)
        mean_str = " with non-zero mean"
    elseif constant && (d + D == 1)
        mean_str = " with drift        "
    elseif !constant && (d + D == 0)
        mean_str = " with zero mean    "
    else
        mean_str = "                   "
    end
    ic_str = if isnan(ic_value)
        "NaN"
    elseif isfinite(ic_value)
        string(round(ic_value, digits = 3))
    else
        "Inf"
    end
    # Combine all parts, pad to fixed width
    s = " ARIMA($(p),$(d),$(q))$(seasonal_part)$(mean_str) : $(ic_str)"

    return s
end

function fit_custom_arima(
    x,
    m;
    order = PDQ(0, 0, 0),
    seasonal = PDQ(0, 0, 0),
    constant::Bool = true,
    ic::AbstractString = "aic",
    trace::Bool = false,
    approximation::Bool = false,
    offset::Float64 = 0.0,
    xreg::Union{Nothing,NamedMatrix} = nothing,
    method::Union{Nothing,AbstractString} = nothing,
    nstar::Union{Nothing,Int} = nothing,  # Pre-computed series length
    kwargs...,
)

    # Use pre-computed nstar if available, otherwise compute
    if isnothing(nstar)
        first = findfirst(xi -> !(ismissing(xi) || isnan(xi)), x)
        last = findlast(xi -> !(ismissing(xi) || isnan(xi)), x)

        if isnothing(first) || isnothing(last)
            n = 0
        else
            n = count(xi -> !(ismissing(xi) || isnan(xi)), @view x[first:last])
        end
    else
        n = nstar + order.d + seasonal.d * m
    end

    use_season = (seasonal.p + seasonal.d + seasonal.q) > 0 && m > 0
    diffs = order.d + seasonal.d

    if isnothing(method)
        if approximation
            method = "CSS"
        else
            method = "CSS-ML"
        end
    end

    drift_case = (diffs == 1) && constant

    if drift_case
        drift = collect(1:length(x))
        if isnothing(xreg)
            xreg = NamedMatrix(reshape(drift, :, 1), ["drift"])
        else
            xreg = add_drift_term(xreg, drift, "drift")
        end

        fit = try
            if use_season
                arima(
                    x,
                    m;
                    order = order,
                    seasonal = seasonal,
                    xreg = xreg,
                    method = method,
                    kwargs...,
                )
            else
                arima(
                    x,
                    m;
                    order = order,
                    seasonal = PDQ(0, 0, 0),
                    xreg = xreg,
                    method = method,
                    kwargs...,
                )
            end
        catch err
            err  # Store error, handle below
        end
    else
        fit = try
            if use_season
                arima(
                    x,
                    m;
                    order = order,
                    seasonal = seasonal,
                    xreg = xreg,
                    method = method,
                    include_mean = constant,
                    kwargs...,
                )
            else
                arima(
                    x,
                    m;
                    order = order,
                    seasonal = PDQ(0, 0, 0),
                    xreg = xreg,
                    method = method,
                    include_mean = constant,
                    kwargs...,
                )
            end
        catch err
            err  # Store error, handle below
        end
    end

    if isnothing(xreg)
        nxreg = 0
    else
        nxreg = size(xreg.data, 2)
    end

    if fit isa ArimaFit
        nstar = n - order.d - seasonal.d * m
        if drift_case
            fit.xreg = xreg
        end
        #npar = length(get_vector(fit.coef, row = 1)[fit.mask]) + 1
        npar = (sum(fit.mask) + 1)
        if method == "CSS"
            fit.aic = offset + nstar * log(fit.sigma2) + 2 * npar
        end
        if !(isnan(fit.aic))
            fit.bic = fit.aic + npar * (log(nstar) - 2)
            fit.aicc = fit.aic + 2 * npar * (npar + 1) / (nstar - npar - 1)
            # Use direct field access instead of string comparison
            fit.ic = ic == "bic" ? fit.bic : (ic == "aicc" ? fit.aicc : fit.aic)
        else
            fit.aic, fit.bic, fit.aicc, fit.ic = Inf, Inf, Inf, Inf
        end

        minroot = 2.0

        # AR roots
        if (order.p + seasonal.p) > 0
            testvec = fit.model.phi
            # Find last non-zero element more efficiently
            lastnz = 0
            @inbounds for i in length(testvec):-1:1
                if abs(testvec[i]) > 1e-8
                    lastnz = i
                    break
                end
            end
            if lastnz > 0
                proots = try
                    roots(Polynomial([1.0; -@view(testvec[1:lastnz])]))
                catch err
                    nothing  # Flag error, handled below
                end
                if !isnothing(proots) && !(proots isa Exception)
                    minroot = min(minroot, minimum(abs.(proots)))
                else
                    fit.ic = Inf
                end
            end
        end
        if (order.q + seasonal.q) > 0 && fit.ic < Inf
            testvec = fit.model.theta
            # Find last non-zero element more efficiently
            lastnz = 0
            @inbounds for i in length(testvec):-1:1
                if abs(testvec[i]) > 1e-8
                    lastnz = i
                    break
                end
            end
            if lastnz > 0
                proots = try
                    roots(Polynomial([1.0; @view(testvec[1:lastnz])]))
                catch err
                    nothing  # Flag error, handled below
                end
                if !isnothing(proots) && !(proots isa Exception)
                    minroot = min(minroot, minimum(abs.(proots)))
                else
                    fit.ic = Inf
                end
            end
        end
       
        if minroot < 1 + 1e-2
            fit.ic = Inf
        end
        fit.xreg = xreg

        if trace
            println()
            println(arima_trace_str(order, seasonal, m, constant; ic_value = fit.ic))
        end
        return fit
    else
        errtxt = sprint(showerror, fit)
        if occursin("unused argument", errtxt)
            error(first(split(errtxt, '\n')))
        end
        if trace
            println()
            println(arima_trace_str(order, seasonal, m, constant))
        end
        return ArimaFit(
            Array{Float64}(undef, 0),
            Array{Float64}(undef, 0),
            NamedMatrix(zeros(1, 1), ["z"]),
            0.0,
            zeros(0, 0),
            Bool[],
            0.0,
            nothing,
            nothing,
            nothing,
            Inf,
            Int[],
            Float64[],
            false,
            0,
            0,
            ArimaStateSpace(
                Array{Float64}(undef, 0),
                Array{Float64}(undef, 0),
                Array{Float64}(undef, 0),
                Array{Float64}(undef, 0),
                Array{Float64}(undef, 0),
                zeros(1, 1),
                zeros(1, 1),
                0.0,
                0.0,
                zeros(1, 1),
            ),
            nothing,
            "Error model",
            nothing,
            nothing,
            nothing,
        )

    end

end

function search_arima(
    x::AbstractArray,
    m::Int;
    d::Int,
    D::Int,
    max_p::Int = 5,
    max_q::Int = 5,
    max_P::Int = 2,
    max_Q::Int = 2,
    max_order::Int = 5,
    stationary::Bool = false,
    ic = "aic",
    trace::Bool = false,
    approximation::Bool = false,
    xreg = nothing,
    offset::Float64 = 0.0,
    allowdrift::Bool = true,
    allowmean::Bool = true,
    kwargs...,
)
    ic = match_arg(ic, ["aic", "aicc", "bic"])

    allowdrift = allowdrift && (d + D == 1)
    allowmean = allowmean && (d + D == 0)
    maxK = Int(allowdrift || allowmean)

    best_ic = Inf
    bestfit = nothing
    constant = nothing

    @inbounds for i = 0:max_p
        for j = 0:max_q
            for I = 0:max_P
                for J = 0:max_Q
                    if i + j + I + J <= max_order
                        for K = 0:maxK
                            fit = fit_custom_arima(
                                x,
                                m;
                                order = PDQ(i, d, j),
                                seasonal = PDQ(I, D, J),
                                constant = (K == 1),
                                ic = ic,
                                trace = trace,
                                approximation = approximation,
                                offset = offset,
                                xreg = xreg,
                                kwargs...,
                            )

                            if fit isa ArimaFit
                                if fit.ic < best_ic
                                    best_ic = fit.ic
                                    bestfit = fit
                                    constant = (K == 1)
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    if !isnothing(bestfit)

        if approximation
            if trace
                println("\n\n Now re-fitting the best model(s) without approximations...\n")
            end


            p = bestfit.arma[1]
            d_ = bestfit.arma[6]
            q = bestfit.arma[2]
            P = bestfit.arma[3]
            D_ = bestfit.arma[7]
            Q = bestfit.arma[4]

            newbestfit = fit_custom_arima(
                x,
                m;
                order = PDQ(p, d_, q),
                seasonal = PDQ(P, D_, Q),
                constant = constant,
                ic = ic,
                trace = false,
                approximation = false,
                xreg = xreg,
                kwargs...,
            )

            if newbestfit.ic == Inf
                bestfit = search_arima(
                    x,
                    m;
                    d = d,
                    D = D,
                    max_p = max_p,
                    max_q = max_q,
                    max_P = max_P,
                    max_Q = max_Q,
                    max_order = max_order,
                    stationary = stationary,
                    ic = ic,
                    trace = trace,
                    approximation = false,
                    xreg = xreg,
                    offset = offset,
                    allowdrift = allowdrift,
                    allowmean = allowmean,
                    kwargs...,
                )
            else
                bestfit = newbestfit
            end
        end
    else
        error("No ARIMA model able to be estimated")
    end

    return bestfit
end
