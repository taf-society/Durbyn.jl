
time_index(n::Int, m::Int; start::Float64=1.0) = start .+ (0:n-1) ./ max(m, 1)

has_coef(fit::ArimaFit, name::AbstractString) = any(==(name), fit.coef.colnames)

npar(fit::ArimaFit) = count(identity, fit.mask) + 1

function n_and_nstar(fit::ArimaFit)
    n     = fit.nobs - fit.n_cond
    d     = Int(fit.arma[6])
    D     = Int(fit.arma[7])
    m     = Int(fit.arma[5])
    nstar = n - d - D*m
    return n, nstar
end
function prepend_drift(xreg::Union{Nothing,NamedMatrix}, drift::AbstractVector{<:Real})
    driftcol = reshape(Float64.(drift), :, 1)
    return xreg === nothing ? NamedMatrix(driftcol, ["drift"]) :
                              add_dift_term(xreg, driftcol, "drift")
end

function prepare_drift(model::ArimaFit, x, xreg::Union{Nothing,NamedMatrix})
    n_train   = length(model.y)
    m_train   = Int(model.arma[5])
    t_train   = time_index(n_train, m_train)
    @assert model.xreg isa NamedMatrix "Original model has no xreg for drift reconstruction"
    @assert any(==("drift"), model.xreg.colnames) "Original model has no 'drift' column"

    drift_vec = get_vector(model.xreg; col="drift")

    # OLS: drift_vec ~ a + b * t_train
    X = hcat(ones(n_train), t_train)
    fitt = ols(drift_vec, X)
    coef = coefficients(fitt)
    a, b = coef[1], coef[2]

    n_new  = length(x)
    m_new  = Int(model.arma[5])
    t_new  = time_index(n_new, m_new)
    newdr  = a .+ b .* t_new
    xreg_with_drift = prepend_drift(xreg, newdr)
    return align_columns(xreg_with_drift, model.xreg.colnames)
end

function refit_arima_model(x, m::Int, model::ArimaFit, 
    xreg::Union{Nothing,NamedMatrix}, method::String; kwargs...)
    p, q, P, Q, _m, d, D = model.arma
    order    = PDQ(p, d, q)
    seasonal = PDQ(P, D, Q)

    fit = arima(x, m;
        order        = order,
        seasonal     = seasonal,
        xreg         = xreg,
        include_mean = has_coef(model, "intercept"),
        method       = method,
        fixed        = model.coef.data,
        kwargs...
    )

    k = size(fit.coef.data, 2)
    fit.var_coef .= 0.0
    fit.sigma2 = model.sigma2
    if xreg !== nothing
        fit.xreg = xreg
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
    model::Union{Nothing,ArimaFit} = nothing,
    kwargs...
)


    x2 = copy(y)
    method = match_arg(method, ["CSS-ML","ML","CSS"])

    if lambda !== nothing
        x2, lambda = box_cox(x2, m; lambda=lambda)
    end

    seasonal2 = (m <= 1) ? PDQ(0,0,0) : seasonal
    if (m <= 1) && (length(x2) <= order.d)
        error("Not enough data to fit the model")
    elseif (m > 1) && (length(x2) <= order.d + seasonal2.d * m)
        error("Not enough data to fit the model")
    end

    if include_constant !== nothing
        if include_constant === true
            include_mean = true
            if (order.d + seasonal2.d) == 1
                include_drift = true
            end
        else
            include_mean = false
            include_drift = false
        end
    end
    if (order.d + seasonal2.d) > 1 && include_drift
        @warn "No drift term fitted as the order of difference is 2 or more."
        include_drift = false
    end

    fit = nothing
    if model !== nothing
        had_xreg  = (model.xreg isa NamedMatrix)
        use_drift = had_xreg && any(==("drift"), model.xreg.colnames)

        if had_xreg && xreg === nothing
            error("No regressors provided")
        end

        xreg2 = use_drift ? prepare_drift(model, x2, xreg) :
                (had_xreg ? align_columns(xreg, model.xreg.colnames) : xreg)

        fit = refit_arima_model(x2, m, model, xreg2, method; kwargs...)
    else
        xreg2 = include_drift ? prepend_drift(xreg, 1:length(x2)) : xreg
        fit = arima(x2, m;
            order        = order,
            seasonal     = seasonal2,
            xreg         = xreg2,
            include_mean = include_mean,
            method       = method,
            kwargs...
        )
    end

    n, nstar = n_and_nstar(fit)
    np = npar(fit)

    if fit.aic !== nothing
        fit.aicc = fit.aic + 2*np * (nstar / (nstar - np - 1) - 1)
        fit.bic  = fit.aic + np * (log(nstar) - 2)
    end

    if model === nothing
        ss = sum(v->v*v, fit.residuals[(fit.n_cond+1):end])
        fit.sigma2 = ss / (nstar - np + 1)
    end

    if model === nothing && include_drift && xreg !== nothing
        fit.xreg = xreg2 
    end
    
    return fit
end