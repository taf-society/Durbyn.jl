"""
    ArimaFit

Backward-compatible wrapper around [`SARIMA`](@ref).

Provides the same field-access interface as the legacy `ArimaFit` mutable struct,
delegating reads and writes to the underlying `SARIMA` model and its components.
"""
struct ArimaFit
    _model::SARIMA{Float64}
    _method_string::Ref{String}
    _offset::Ref{Union{Float64,Nothing}}
end

ArimaFit(model::SARIMA{Float64}, method_str::String) =
    ArimaFit(model, Ref(method_str), Ref{Union{Float64,Nothing}}(nothing))

function Base.getproperty(fit::ArimaFit, sym::Symbol)
    sym === :_model && return getfield(fit, :_model)
    sym === :_method_string && return getfield(fit, :_method_string)
    sym === :_offset && return getfield(fit, :_offset)

    m = getfield(fit, :_model)
    r = m.results

    sym === :y && return m.y_orig
    sym === :fitted && return isnothing(r) ? Float64[] : r.fitted_values
    sym === :residuals && return isnothing(r) ? Float64[] : r.residuals

    sym === :coef && return isnothing(r) ? NamedMatrix(zeros(1, 1), ["z"]) : r.coef
    sym === :sigma2 && return isnothing(r) ? 0.0 : r.sigma2
    sym === :var_coef && return isnothing(r) ? zeros(0, 0) : r.var_coef
    sym === :mask && return m.hyperparameters.mask

    sym === :loglik && return isnothing(r) ? 0.0 : r.loglik
    sym === :aic && return isnothing(r) ? nothing : r.aic
    sym === :bic && return isnothing(r) ? nothing : r.bic
    sym === :aicc && return isnothing(r) ? nothing : r.aicc
    sym === :ic && return isnothing(r) ? Inf : r.ic

    sym === :arma && return arma_vector(m.order)

    sym === :convergence_code && return isnothing(r) ? false : r.convergence_code
    sym === :n_cond && return isnothing(r) ? 0 : r.n_cond
    sym === :nobs && return isnothing(r) ? 0 : r.nobs

    sym === :model && return m.system

    sym === :xreg && return m.xreg
    sym === :method && return getfield(fit, :_method_string)[]
    sym === :lambda && return m.lambda
    sym === :biasadj && return m.biasadj
    sym === :offset && return getfield(fit, :_offset)[]

    error("ArimaFit has no field $sym")
end

function Base.setproperty!(fit::ArimaFit, sym::Symbol, val)
    m = getfield(fit, :_model)
    r = m.results

    sym === :aic && (r.aic = val; return val)
    sym === :bic && (r.bic = val; return val)
    sym === :aicc && (r.aicc = val; return val)
    sym === :ic && (r.ic = val; return val)
    sym === :sigma2 && (r.sigma2 = val; return val)
    sym === :xreg && (m.xreg = val; return val)
    sym === :lambda && (m.lambda = val; return val)
    sym === :y && (m.y_orig = [ismissing(vi) ? NaN : Float64(vi) for vi in val]; return val)
    sym === :fitted && (r.fitted_values = val; return val)
    sym === :method && (getfield(fit, :_method_string)[] = val; return val)
    sym === :offset && (getfield(fit, :_offset)[] = val; return val)
    sym === :biasadj && (m.biasadj = val; return val)
    sym === :var_coef && (r.var_coef = val; return val)
    sym === :n_cond && (r.n_cond = val; return val)
    sym === :loglik && (r.loglik = val; return val)

    error("Cannot set ArimaFit field $sym")
end

function Base.propertynames(::ArimaFit)
    (:y, :fitted, :coef, :sigma2, :var_coef, :mask, :loglik,
     :aic, :bic, :aicc, :ic, :arma, :residuals, :convergence_code,
     :n_cond, :nobs, :model, :xreg, :method, :lambda, :biasadj, :offset)
end

function _error_arimafit()
    order = SARIMAOrder(0, 0, 0, 0, 0, 0, 0)
    hp = HyperParameters{Float64}(
        Float64[], Float64[], Bool[], String[], Float64[],
    )
    sys = SARIMASystem{Float64}(
        Float64[], Float64[], Float64[], Float64[],
        zeros(1, 1), zeros(1, 1), 0.0,
        Float64[], zeros(1, 1), zeros(1, 1),
    )
    results = SARIMAResults{Float64}(
        NamedMatrix(zeros(1, 1), ["z"]),
        0.0,
        zeros(0, 0),
        0.0,
        nothing, nothing, nothing, Inf,
        Float64[],
        Float64[],
        false,
        0, 0,
    )
    m = SARIMA{Float64}(
        Float64[], Float64[], order, hp,
        sys, nothing, results,
        nothing, nothing,
        false, false, nothing, nothing,
        :css_ml, false, :gardner1980, :bfgs, Dict(), 1e6,
        0, nothing, nothing,
    )
    return ArimaFit(m, "Error model")
end

function fitted(model::ArimaFit)
    return model.fitted
end

function residuals(model::ArimaFit)
    return model.residuals
end
