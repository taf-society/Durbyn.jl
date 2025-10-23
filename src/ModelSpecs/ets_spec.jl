"""
    EtsSpec

Specification for Exponential Smoothing (ETS) models using the forecasting grammar.

An ETS specification is defined via `@formula` with the helper terms:
- `e("A"|"M"|"Z")` for the error component
- `t("N"|"A"|"M"|"Z")` for the trend component
- `s("N"|"A"|"M"|"Z")` for the seasonal component
- Optional `drift()` to request a damped trend (`drift(false)` forbids damping,
  `drift(:auto)` lets the algorithm decide).
"""
struct EtsSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    components::NamedTuple{(:error, :trend, :seasonal), NTuple{3, String}}
    damped::Union{Bool, Nothing}
    options::Dict{Symbol, Any}

    function EtsSpec(formula::ModelFormula;
                     m::Union{Int, Nothing} = nothing,
                     kwargs...)
        parts = compile_ets_formula(formula)

        opts = Dict{Symbol, Any}(kwargs)
        damped_kw = haskey(opts, :damped) ? pop!(opts, :damped) : nothing

        damped_from_formula = parts.damped
        damped = if isnothing(damped_kw)
            damped_from_formula
        elseif damped_kw === nothing || damped_kw isa Bool
            damped_kw
        else
            throw(ArgumentError("damped kwarg must be Bool or nothing, got $(typeof(damped_kw))"))
        end

        components = (error = parts.error,
                      trend = parts.trend,
                      seasonal = parts.seasonal)

        new(formula, m, components, damped, opts)
    end
end

"""
    FittedEts

Container for ETS models fitted via `EtsSpec`.
"""
struct FittedEts <: AbstractFittedModel
    spec::EtsSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedEts(spec::EtsSpec,
                       fit,
                       target_col::Symbol,
                       data,
                       m::Int)
        schema = Dict{Symbol, Type}()
        for (k, v) in pairs(data)
            schema[k] = eltype(v)
        end
        new(spec, fit, target_col, schema, m)
    end
end

"""
    extract_metrics(model::FittedEts)

Collect standard fit statistics from an ETS model.
"""
function extract_metrics(model::FittedEts)
    metrics = Dict{Symbol, Float64}()
    for name in (:aic, :aicc, :bic, :mse, :amse, :sigma2)
        if hasproperty(model.fit, name)
            value = getproperty(model.fit, name)
            if !(value isa Bool) && !isnothing(value) && value isa Real
                metrics[name] = Float64(value)
            end
        end
    end
    return metrics
end

function Base.show(io::IO, spec::EtsSpec)
    comp = spec.components
    print(io, "EtsSpec: ")
    print(io, spec.formula.target, " ~ ETS(",
          comp.error, ",", comp.trend, ",", comp.seasonal, ")")
    if !isnothing(spec.m)
        print(io, ", m = ", spec.m)
    end
    if !isnothing(spec.damped)
        print(io, ", damped = ", spec.damped)
    end
end

function Base.show(io::IO, fitted::FittedEts)
    comp = fitted.spec.components
    print(io, "FittedEts: ETS(",
          comp.error, ",", comp.trend, ",", comp.seasonal, ")")
    if !isnothing(fitted.spec.damped)
        print(io, fitted.spec.damped ? " with damping" : " (no damping)")
    end
    if hasproperty(fitted.fit, :aic) && !isnothing(fitted.fit.aic)
        print(io, ", AIC = ", round(fitted.fit.aic, digits=2))
    end
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedEts)
    comp = fitted.spec.components
    println(io, "FittedEts")
    println(io, "  Model: ETS(", comp.error, ",", comp.trend, ",", comp.seasonal, ")")
    println(io, "  m: ", fitted.m)
    if hasproperty(fitted.fit, :method)
        println(io, "  Method: ", fitted.fit.method)
    end
    if !isnothing(fitted.spec.damped)
        println(io, "  Damped: ", fitted.spec.damped)
    end
    if hasproperty(fitted.fit, :aic) && !isnothing(fitted.fit.aic)
        println(io, "  AIC:  ", round(fitted.fit.aic, digits=4))
    end
    if hasproperty(fitted.fit, :aicc) && !isnothing(fitted.fit.aicc)
        println(io, "  AICc: ", round(fitted.fit.aicc, digits=4))
    end
    if hasproperty(fitted.fit, :bic) && !isnothing(fitted.fit.bic)
        println(io, "  BIC:  ", round(fitted.fit.bic, digits=4))
    end
    if hasproperty(fitted.fit, :mse) && !isnothing(fitted.fit.mse)
        println(io, "  MSE:  ", round(fitted.fit.mse, digits=6))
    end
    println(io, "  n:    ", length(fitted.fit.x))
end
