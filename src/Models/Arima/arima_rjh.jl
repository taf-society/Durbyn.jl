function _has_coef(model, name::AbstractString)
    if hasproperty(model, :coefnames) && model.coefnames !== nothing
        return any(==(name), String.(model.coefnames))
    end
    if hasproperty(model, :coef)
        c = getfield(model, :coef)
        if c isa NamedTuple
            return haskey(c, Symbol(name))
        elseif c isa Dict
            return haskey(c, name)
        elseif c isa NamedMatrix
            return any(==(name), c.colnames)
        elseif c isa AbstractVector && hasproperty(model, :coeflabels) && model.coeflabels !== nothing
            return any(==(name), String.(model.coeflabels))
        end
    end
    return false
end

_namedmatrix_has_col(nm::NamedMatrix, name::AbstractString) = any(==(name), nm.colnames)

time_index(n::Int, m::Int; start::Float64=1.0) = start .+ (0:n-1) ./ max(m, 1)

function _npar(fit) :: Int
    if hasproperty(fit, :mask) && fit.mask !== nothing
        return count(identity, fit.mask) + 1
    elseif hasproperty(fit, :coef) && fit.coef !== nothing
        return (fit.coef isa NamedMatrix ? size(fit.coef.data, 2) : length(fit.coef)) + 1
    else
        return 1
    end
end


_arma_from(order::PDQ, seasonal::PDQ, m::Int) = [order.p, order.q, seasonal.p, seasonal.q, m, order.d, seasonal.d]


function _residual_window(r::AbstractVector)
    r_m = Vector{Union{Missing,Float64}}(undef, length(r))
    @inbounds for i in eachindex(r)
        v = r[i]
        r_m[i] = (v === missing || (v isa Real && isnan(v))) ? missing : Float64(v)
    end
    first = findfirst(!ismissing, r_m)
    last  = findlast(!ismissing, r_m)
    return r_m, first, last
end

function _prepend_drift_col(xreg::Union{Nothing,NamedMatrix}, drift::AbstractVector{<:Real})
    driftcol = reshape(Float64.(drift), :, 1)
    if xreg === nothing
        return NamedMatrix(driftcol, ["drift"])
    else
        
        return add_dift_term(xreg, driftcol, "drift")
    end
end

function prepare_drift(model, x, xreg::Union{Nothing,NamedMatrix})
    n_train = length(model.x)
    m_train = hasproperty(model, :arma) ? Int(model.arma[5]) : 1
    start_train = hasproperty(model, :tstart) ? float(model.tstart) : 1.0
    t_train = time_index(n_train, m_train; start=start_train)

    (hasproperty(model, :xreg) && model.xreg !== nothing && _namedmatrix_has_col(model.xreg, "drift")) ||
        error("Refit requested but original model has no 'drift' regressor recorded.")
    drift_vec = get_vector(model.xreg; col="drift")

    X = hcat(ones(n_train), t_train)
    fitt = ols(drift_vec, X)
    coef = coefficients(fitt)
    a, b = coef[1], coef[2]

    n_new = length(x)
    m_new = hasproperty(model, :arma) ? Int(model.arma[5]) : 1
    start_new = hasproperty(model, :tstart_new) ? float(model.tstart_new) : 1.0
    t_new = time_index(n_new, m_new; start=start_new)
    new_drift = a .+ b .* t_new

    xreg_with_drift = _prepend_drift_col(xreg, new_drift)
    xreg_aligned = align_columns(xreg_with_drift, model.xreg.colnames)
    return xreg_aligned
end


function refit_arima_model(x, m::Int, model, xreg::Union{Nothing,NamedMatrix}, method::String; kwargs...)
    p, q, P, Q, m0, d, D = model.arma
    order    = PDQ(Int(p), Int(d), Int(q))
    seasonal = PDQ(Int(P), Int(D), Int(Q))

    fit = arima(x, m;
        order        = order,
        seasonal     = seasonal,
        xreg         = xreg,
        include_mean = _has_coef(model, "intercept"),
        method       = method,
        fixed        = hasproperty(model, :coef) ? model.coef : nothing,
        kwargs...
    )

    if hasproperty(fit, :var_coef) && hasproperty(fit, :coef) && fit.coef !== nothing
        k = (fit.coef isa NamedMatrix) ? size(fit.coef.data, 2) : length(fit.coef)
        fit.var_coef = zeros(k, k)
    end
    if xreg !== nothing && hasproperty(fit, :xreg)
        fit.xreg = xreg
    end
    if hasproperty(model, :sigma2) && hasproperty(fit, :sigma2)
        fit.sigma2 = model.sigma2
    end
    return fit
end

function arima_rjh(y, m::Int;
    order::PDQ = PDQ(0,0,0),
    seasonal::PDQ = PDQ(0,0,0),
    xreg::Union{Nothing,NamedMatrix} = nothing,
    include_mean::Bool = true,
    include_drift::Bool = false,
    include_constant = nothing,
    lambda = nothing,
    biasadj::Bool = false,
    method::String = "CSS-ML",
    model = nothing,
    x = y,
    kwargs...
)
    origx = y
    method = match_arg(method, ["CSS-ML","ML","CSS"])

    if lambda !== nothing
        x = box_cox(x, m; lambda=lambda)
    end

    seasonal = (m <= 1) ? PDQ(0,0,0) : seasonal
    if (m <= 1) && (length(x) <= order.d)
        error("Not enough data to fit the model")
    elseif (m > 1) && (length(x) <= order.d + seasonal.d * m)
        error("Not enough data to fit the model")
    end

    if include_constant !== nothing
        if include_constant === true
            include_mean = true
            if (order.d + seasonal.d) == 1
                include_drift = true
            end
        else
            include_mean = false
            include_drift = false
        end
    end
    if (order.d + seasonal.d) > 1 && include_drift
        @warn "No drift term fitted as the order of difference is 2 or more."
        include_drift = false
    end

    fit = nothing
    if model !== nothing
        had_xreg = hasproperty(model, :xreg) && model.xreg !== nothing
        use_drift = had_xreg && _namedmatrix_has_col(model.xreg, "drift")
        if had_xreg
            xreg === nothing && error("No regressors provided")
        end

        xreg2 = use_drift ? prepare_drift(model, x, xreg) :
                 (had_xreg ? align_columns(xreg, model.xreg.colnames) : xreg)

        fit = refit_arima_model(x, m, model, xreg2, method; kwargs...)
        fit.lambda = hasproperty(model, :lambda) ? model.lambda : lambda

    else
        xreg2 = include_drift ? _prepend_drift_col(xreg, 1:length(x)) : xreg

        fit = arima(x, m;
            order        = order,
            seasonal     = seasonal,
            xreg         = xreg2,
            include_mean = include_mean,
            method       = method,
            kwargs...
        )
    end

    if !hasproperty(fit, :arma) || fit.arma === nothing
        fit.arma = _arma_from(order, seasonal, m)
    end

    r = fit.residuals
    r_m, first, last = _residual_window(r)
    (first === nothing || last === nothing) && error("No non-missing residuals produced by the fit")

    n     = count(!ismissing, @view r_m[first:last])
    npar  = _npar(fit)
    d     = Int(fit.arma[6])
    D     = Int(fit.arma[7])
    mm    = Int(fit.arma[5])
    nstar = n - d - D*mm

    if hasproperty(fit, :aic) && fit.aic !== nothing
        fit.aicc = fit.aic + 2npar * (nstar / (nstar - npar - 1) - 1)
        fit.bic  = fit.aic + npar * (log(nstar) - 2)
    end

    if model === nothing && hasproperty(fit, :sigma2)
        ss = zero(Float64)
        @inbounds for i in first:last
            v = r_m[i]
            if v !== missing
                ss += v*v
            end
        end
        fit.sigma2 = ss / (nstar - npar + 1)
    end

    if hasproperty(fit, :xreg)
        fit.xreg = (model === nothing) ? (include_drift ? xreg2 : xreg) : xreg2
    end
    if hasproperty(fit, :x)
        fit.x = origx
    end
    if hasproperty(fit, :series)
        fit.series = "x"
    end
    if lambda !== nothing && hasproperty(fit, :lambda_biasadj)
        fit.lambda_biasadj = biasadj
    end

    return fit
end