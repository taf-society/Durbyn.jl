"""
    theta(formula::ModelFormula, data; m::Int=1, kwargs...)

Fit a Theta model from a formula term in Durbyn grammar.
"""
function theta(formula::ModelFormula, data; m::Int=1, kwargs...)
    if !Tables.istable(data)
        throw(ArgumentError("Input must be a Tables.jl-compatible table"))
    end
    tbl = Tables.columntable(data)

    target = formula.target
    if !haskey(tbl, target)
        available_cols = join(keys(tbl), ", ")
        throw(ArgumentError(
            "Target variable ':$(target)' not found in data. Available columns: $(available_cols)"
        ))
    end

    y = tbl[target]
    if !(y isa AbstractVector)
        throw(ArgumentError("Target variable ':$(target)' must be a vector, got $(typeof(y))"))
    end

    theta_term = _extract_single_term(formula, ThetaTerm)
    theta_args = Dict{Symbol, Any}()

    if !isnothing(theta_term.alpha)
        theta_args[:alpha] = theta_term.alpha
    end
    if !isnothing(theta_term.theta)
        theta_args[:theta_param] = theta_term.theta
    end
    if !isnothing(theta_term.decomposition_type)
        theta_args[:decomposition_type] = theta_term.decomposition_type
    end
    if !isnothing(theta_term.nmse)
        theta_args[:nmse] = theta_term.nmse
    end

    merge!(theta_args, Dict{Symbol, Any}(kwargs))

    if isnothing(theta_term.model_type)
        return auto_theta(y, m; theta_args...)
    end

    model_enum = _symbol_to_theta_model_type(theta_term.model_type)
    if haskey(theta_args, :decomposition_type)
        return auto_theta(y, m; model = model_enum, theta_args...)
    end
    return theta(y, m; model_type = model_enum, theta_args...)
end

function _symbol_to_theta_model_type(sym::Symbol)
    if sym === :STM
        return STM
    elseif sym === :OTM
        return OTM
    elseif sym === :DSTM
        return DSTM
    elseif sym === :DOTM
        return DOTM
    else
        throw(ArgumentError("Unknown Theta model type: :$(sym). Valid types: :STM, :OTM, :DSTM, :DOTM"))
    end
end
