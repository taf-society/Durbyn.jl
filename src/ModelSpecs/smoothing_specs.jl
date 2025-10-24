import ..Grammar: SesTerm, HoltTerm, HoltWintersTerm, CrostonTerm, _extract_single_term

"""
    SesSpec

Specification wrapper for Simple Exponential Smoothing (`ses()`).
"""
struct SesSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    options::Dict{Symbol, Any}

    function SesSpec(formula::ModelFormula;
                     m::Union{Int, Nothing} = nothing,
                     kwargs...)
        _extract_single_term(formula, SesTerm)
        new(formula, m, Dict{Symbol, Any}(kwargs))
    end
end

"""
    HoltSpec

Specification wrapper for Holt's linear trend method (`holt()`).
"""
struct HoltSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    damped::Union{Bool, Nothing}
    exponential::Bool
    options::Dict{Symbol, Any}

    function HoltSpec(formula::ModelFormula;
                      m::Union{Int, Nothing} = nothing,
                      kwargs...)
        term = _extract_single_term(formula, HoltTerm)
        opts = Dict{Symbol, Any}(kwargs)
        damped = haskey(opts, :damped) ? pop!(opts, :damped) : term.damped
        if !(damped === nothing || damped isa Bool)
            throw(ArgumentError("damped must be Bool or nothing, got $(typeof(damped))"))
        end
        exponential = haskey(opts, :exponential) ? pop!(opts, :exponential) : term.exponential
        exponential isa Bool ||
            throw(ArgumentError("exponential must be Bool, got $(typeof(exponential))"))
        new(formula, m, damped, exponential, opts)
    end
end

"""
    HoltWintersSpec

Specification wrapper for Holt-Winters seasonal exponential smoothing (`hw()`/`holt_winters()`).
"""
struct HoltWintersSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    seasonal::String
    damped::Union{Bool, Nothing}
    exponential::Bool
    options::Dict{Symbol, Any}

    function HoltWintersSpec(formula::ModelFormula;
                             m::Union{Int, Nothing} = nothing,
                             kwargs...)
        term = _extract_single_term(formula, HoltWintersTerm)
        opts = Dict{Symbol, Any}(kwargs)
        seasonal = haskey(opts, :seasonal) ? pop!(opts, :seasonal) : term.seasonal
        seasonal_str = lowercase(String(seasonal))
        seasonal_str in ("additive", "multiplicative") ||
            throw(ArgumentError("seasonal must be \"additive\" or \"multiplicative\", got $(seasonal)"))
        damped = haskey(opts, :damped) ? pop!(opts, :damped) : term.damped
        if !(damped === nothing || damped isa Bool)
            throw(ArgumentError("damped must be Bool or nothing, got $(typeof(damped))"))
        end
        exponential = haskey(opts, :exponential) ? pop!(opts, :exponential) : term.exponential
        exponential isa Bool ||
            throw(ArgumentError("exponential must be Bool, got $(typeof(exponential))"))
        if exponential && seasonal_str == "additive"
            throw(ArgumentError("exponential trend cannot be combined with additive seasonality."))
        end
        new(formula, m, seasonal_str, damped, exponential, opts)
    end
end

"""
    CrostonSpec

Specification wrapper for Croston's intermittent demand method (`croston()`).
"""
struct CrostonSpec <: AbstractModelSpec
    formula::ModelFormula
    m::Union{Int, Nothing}
    options::Dict{Symbol, Any}

    function CrostonSpec(formula::ModelFormula;
                         m::Union{Int, Nothing} = nothing,
                         kwargs...)
        _extract_single_term(formula, CrostonTerm)
        new(formula, m, Dict{Symbol, Any}(kwargs))
    end
end

"""
    FittedSes
"""
struct FittedSes <: AbstractFittedModel
    spec::SesSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedSes(spec::SesSpec,
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
    FittedHolt
"""
struct FittedHolt <: AbstractFittedModel
    spec::HoltSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedHolt(spec::HoltSpec,
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
    FittedHoltWinters
"""
struct FittedHoltWinters <: AbstractFittedModel
    spec::HoltWintersSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedHoltWinters(spec::HoltWintersSpec,
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
    FittedCroston
"""
struct FittedCroston <: AbstractFittedModel
    spec::CrostonSpec
    fit::Any
    target_col::Symbol
    data_schema::Dict{Symbol, Type}
    m::Int

    function FittedCroston(spec::CrostonSpec,
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

function extract_metrics(model::FittedSes)
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

function extract_metrics(model::FittedHolt)
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

function extract_metrics(model::FittedHoltWinters)
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

function extract_metrics(::FittedCroston)
    return Dict{Symbol, Float64}()
end

function Base.show(io::IO, spec::SesSpec)
    print(io, "SesSpec: ", spec.formula.target)
    if !isnothing(spec.m)
        print(io, ", m = ", spec.m)
    end
end

function Base.show(io::IO, spec::HoltSpec)
    print(io, "HoltSpec: ", spec.formula.target)
    if !isnothing(spec.damped)
        print(io, ", damped = ", spec.damped)
    end
    if spec.exponential
        print(io, ", exponential = true")
    end
    if !isnothing(spec.m)
        print(io, ", m = ", spec.m)
    end
end

function Base.show(io::IO, spec::HoltWintersSpec)
    print(io, "HoltWintersSpec: ", spec.formula.target,
          ", seasonal = \"", spec.seasonal, "\"")
    if !isnothing(spec.damped)
        print(io, ", damped = ", spec.damped)
    end
    if spec.exponential
        print(io, ", exponential = true")
    end
    if !isnothing(spec.m)
        print(io, ", m = ", spec.m)
    end
end

function Base.show(io::IO, spec::CrostonSpec)
    print(io, "CrostonSpec: ", spec.formula.target)
    if !isnothing(spec.m)
        print(io, ", m = ", spec.m)
    end
end

function Base.show(io::IO, fitted::FittedSes)
    print(io, "FittedSes: SES")
    if hasproperty(fitted.fit, :aic) && !isnothing(fitted.fit.aic)
        print(io, ", AIC = ", round(fitted.fit.aic, digits=2))
    end
end

function Base.show(io::IO, fitted::FittedHolt)
    print(io, "FittedHolt: Holt")
    if hasproperty(fitted.fit, :method)
        print(io, " (", fitted.fit.method, ")")
    end
    if hasproperty(fitted.fit, :aic) && !isnothing(fitted.fit.aic)
        print(io, ", AIC = ", round(fitted.fit.aic, digits=2))
    end
end

function Base.show(io::IO, fitted::FittedHoltWinters)
    print(io, "FittedHoltWinters: Holt-Winters(", fitted.spec.seasonal, ")")
    if hasproperty(fitted.fit, :aic) && !isnothing(fitted.fit.aic)
        print(io, ", AIC = ", round(fitted.fit.aic, digits=2))
    end
end

function Base.show(io::IO, fitted::FittedCroston)
    print(io, "FittedCroston: Croston")
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedSes)
    println(io, "FittedSes")
    println(io, "  Target: ", fitted.target_col)
    println(io, "  m: ", fitted.m)
    if hasproperty(fitted.fit, :aic) && !isnothing(fitted.fit.aic)
        println(io, "  AIC:  ", round(fitted.fit.aic, digits=4))
    end
    if hasproperty(fitted.fit, :aicc) && !isnothing(fitted.fit.aicc)
        println(io, "  AICc: ", round(fitted.fit.aicc, digits=4))
    end
    if hasproperty(fitted.fit, :bic) && !isnothing(fitted.fit.bic)
        println(io, "  BIC:  ", round(fitted.fit.bic, digits=4))
    end
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedHolt)
    println(io, "FittedHolt")
    println(io, "  Target: ", fitted.target_col)
    println(io, "  m: ", fitted.m)
    if hasproperty(fitted.fit, :method)
        println(io, "  Method: ", fitted.fit.method)
    end
    if hasproperty(fitted.fit, :aic) && !isnothing(fitted.fit.aic)
        println(io, "  AIC:  ", round(fitted.fit.aic, digits=4))
    end
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedHoltWinters)
    println(io, "FittedHoltWinters")
    println(io, "  Target: ", fitted.target_col)
    println(io, "  m: ", fitted.m)
    println(io, "  Seasonal: ", fitted.spec.seasonal)
    if hasproperty(fitted.fit, :method)
        println(io, "  Method: ", fitted.fit.method)
    end
    if hasproperty(fitted.fit, :aic) && !isnothing(fitted.fit.aic)
        println(io, "  AIC:  ", round(fitted.fit.aic, digits=4))
    end
end

function Base.show(io::IO, ::MIME"text/plain", fitted::FittedCroston)
    println(io, "FittedCroston")
    println(io, "  Target: ", fitted.target_col)
    println(io, "  m: ", fitted.m)
    println(io, "  Type: ", fitted.fit.type)
end
