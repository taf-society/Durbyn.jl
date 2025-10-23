function _xreg_formula_symbols(formula)::Vector{Symbol}
    if isnothing(formula)
        return Symbol[]
    end
    rhs_syms = Symbol[]
    for term in formula.rhs
        append!(rhs_syms, term)
    end
    unique!(rhs_syms)
    return rhs_syms
end

function _build_xreg_formula_matrix(spec::ArimaSpec,
                                    tbl,
                                    n_rows::Int,
                                    Utils_mod)
    formula = spec.xreg_formula
    if isnothing(formula)
        return nothing, Symbol[]
    end

    required_cols = _xreg_formula_symbols(formula)

    if isempty(required_cols)
        source = Utils_mod.NamedMatrix(n_rows, String[]; T=Float64)
        design = Utils_mod.model_matrix(formula, source)
        return design, Symbol.(design.colnames)
    end

    columns = Vector{AbstractVector}(undef, length(required_cols))
    numeric_types = DataType[]

    for (idx, sym) in enumerate(required_cols)
        if !haskey(tbl, sym)
            available_cols = join(string.(keys(tbl)), ", ")
            throw(ArgumentError(
                "Variable ':$(sym)' referenced in xreg_formula not found in data. " *
                "Available columns: $(available_cols)"
            ))
        end
        col = tbl[sym]
        if !(col isa AbstractVector)
            throw(ArgumentError(
                "Variable ':$(sym)' referenced in xreg_formula must be a vector, got $(typeof(col))"
            ))
        end
        if length(col) != n_rows
            throw(ArgumentError(
                "Variable ':$(sym)' referenced in xreg_formula has length $(length(col)), " *
                "but expected length $(n_rows)."
            ))
        end
        if any(ismissing, col)
            throw(ArgumentError(
                "Variable ':$(sym)' referenced in xreg_formula contains missing values. " *
                "Please remove or impute missings before fitting/forecasting."
            ))
        end
        core_type = Base.nonmissingtype(eltype(col))
        if !(core_type <: Number)
            throw(ArgumentError(
                "Variable ':$(sym)' referenced in xreg_formula must be numeric, got element type $(eltype(col))."
            ))
        end
        columns[idx] = col
        push!(numeric_types, core_type)
    end

    promoted_type = promote_type(numeric_types...)
    matrix_data = Matrix{promoted_type}(undef, n_rows, length(required_cols))
    for (j, sym) in enumerate(required_cols)
        matrix_data[:, j] = convert.(promoted_type, columns[j])
    end

    source = Utils_mod.NamedMatrix(matrix_data, String[string(sym) for sym in required_cols])
    design = Utils_mod.model_matrix(formula, source)
    return design, Symbol.(design.colnames)
end

function _auto_exclusion_set(target::Symbol,
                             groupby_cols::Vector{Symbol},
                             datecol::Union{Symbol, Nothing})
    exclude = Set{Symbol}()
    push!(exclude, target)
    for col in groupby_cols
        push!(exclude, col)
    end
    if !isnothing(datecol)
        push!(exclude, datecol)
    end
    return exclude
end

function _collect_auto_xreg_columns(tbl,
                                    exclude::Set{Symbol},
                                    n_rows::Int)
    accepted = Symbol[]
    skipped = Vector{Tuple{Symbol, String}}()
    for key in keys(tbl)
        sym = Symbol(key)
        if sym in exclude
            continue
        end
        col = tbl[sym]
        if !(col isa AbstractVector)
            push!(skipped, (sym, "not a vector"))
            continue
        end
        if length(col) != n_rows
            push!(skipped, (sym, "length $(length(col)) ≠ $(n_rows)"))
            continue
        end
        if any(ismissing, col)
            push!(skipped, (sym, "contains missing values"))
            continue
        end
        core_type = Base.nonmissingtype(eltype(col))
        if !(core_type <: Number)
            push!(skipped, (sym, "non-numeric element type $(core_type)"))
            continue
        end
        push!(accepted, sym)
    end
    return accepted, skipped
end

function _warn_skipped_auto(skipped::Vector{Tuple{Symbol, String}})
    isempty(skipped) && return
    summary = join(["$(string(sym)): $(reason)" for (sym, reason) in skipped], "; ")
    @warn "Skipped $(length(skipped)) column(s) for automatic exogenous selection" summary = summary
end

function _build_auto_xreg_matrix(tbl,
                                 cols::Vector{Symbol},
                                 n_rows::Int,
                                 Utils_mod)
    isempty(cols) && return nothing
    numeric_types = DataType[]
    for sym in cols
        push!(numeric_types, Base.nonmissingtype(eltype(tbl[sym])))
    end
    promoted_type = promote_type(numeric_types...)
    matrix = Matrix{promoted_type}(undef, n_rows, length(cols))
    for (j, sym) in enumerate(cols)
        matrix[:, j] = convert.(promoted_type, tbl[sym])
    end
    return Utils_mod.NamedMatrix(matrix, String[string(sym) for sym in cols])
end
