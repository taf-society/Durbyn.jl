const _DEFAULT_NAMES_TO = :variable
const _DEFAULT_VALUES_TO = :value

struct _PivotSentinel end
const _PIVOT_SENTINEL = _PivotSentinel()

struct GroupedTable{CT, KT, VT}
    data::CT
    keycols::Vector{Symbol}
    keys::Vector{KT}
    indices::Vector{Vector{Int}}
end

Base.length(gt::GroupedTable) = length(gt.keys)

function Base.show(io::IO, gt::GroupedTable)
    n_groups = length(gt)
    keydesc = isempty(gt.keycols) ? "(none)" : join(string.(gt.keycols), ", ")
    print(io, "GroupedTable(", n_groups, " groups by ", keydesc, ")")
end

function Base.show(io::IO, ::MIME"text/plain", gt::GroupedTable)
    n_groups = length(gt)
    println(io, "GroupedTable")
    println(io, "  Groups: ", n_groups)
    println(io, "  Key columns: ", isempty(gt.keycols) ? "(none)" : join(string.(gt.keycols), ", "))

    if n_groups > 0
        sizes = [length(idx) for idx in gt.indices]
        total_rows = sum(sizes)
        min_size = minimum(sizes)
        max_size = maximum(sizes)
        avg_size = total_rows / n_groups
        println(io, "  Rows: ", total_rows, " (avg ", round(avg_size; digits=2),
                ", min ", min_size, ", max ", max_size, ")")

        sample_count = min(n_groups, 5)
        println(io, "  Sample groups (key => rows):")
        for i in 1:sample_count
            key = gt.keys[i]
            key_str = isempty(gt.keycols) ? "(all rows)" : string(key)
            println(io, "    ", key_str, " => ", sizes[i])
        end
        if n_groups > sample_count
            println(io, "    …")
        end
    else
        println(io, "  Rows: 0")
    end
end

function _to_columns(data)
    if data isa NamedTuple
        return data
    end
    Tables.istable(data) ||
        throw(ArgumentError("Input must be Tables.jl-compatible or a NamedTuple of vectors"))
    cols = Tables.columntable(data)
    names = Tuple(propertynames(cols))
    values = ntuple(i -> getproperty(cols, names[i]), length(names))
    return NamedTuple{names}(values)
end

function _column_names(ct::NamedTuple)
    return Tuple(propertynames(ct))
end

function _nrows(ct::NamedTuple)
    names = _column_names(ct)
    isempty(names) && return 0
    return length(ct[names[1]])
end

function _check_lengths(ct::NamedTuple)
    names = _column_names(ct)
    n = _nrows(ct)
    for name in names
        length(ct[name]) == n ||
            throw(ArgumentError("Column '$(name)' length mismatch (expected $(n), got $(length(ct[name])))"))
    end
    return n
end

function _subset_mask(ct::NamedTuple, mask::AbstractVector{Bool})
    names = _column_names(ct)
    vals = ntuple(i -> ct[names[i]][mask], length(names))
    return NamedTuple{names}(vals)
end

function _subset_indices(ct::NamedTuple, idxs::AbstractVector{Int})
    names = _column_names(ct)
    vals = ntuple(i -> ct[names[i]][idxs], length(names))
    return NamedTuple{names}(vals)
end

function _assemble(columns::Vector{Symbol}, data::Vector)
    tuple_names = Tuple(columns)
    values = Tuple(data)
    return NamedTuple{tuple_names}(values)
end

function _row_namedtuple(ct::NamedTuple, i::Int, names::Tuple)
    vals = ntuple(j -> ct[names[j]][i], length(names))
    return NamedTuple{names}(vals)
end

function _finalize_column(vec::Vector)
    if isempty(vec)
        return Vector{Any}()
    end
    T = promote_type(map(typeof, vec)...)
    out = Vector{T}(undef, length(vec))
    for i in eachindex(vec)
        out[i] = convert(T, vec[i])
    end
    return out
end

function select(data, specs...)
    ct = _to_columns(data)
    isempty(specs) && return ct
    available = Set(_column_names(ct))

    names_out = Symbol[]
    columns_out = Vector{Any}()

    for spec in specs
        if spec isa Pair
            new_name = Symbol(first(spec))
            old_name = Symbol(last(spec))
        else
            new_name = Symbol(spec)
            old_name = Symbol(spec)
        end
        old_name in available ||
            throw(ArgumentError("Column '$(old_name)' not found"))
        push!(names_out, new_name)
        push!(columns_out, ct[old_name])
    end

    return _assemble(names_out, columns_out)
end

function query(data, predicate::Function)
    Tables.istable(data) || throw(ArgumentError("TableOps.query expects a Tables.jl compatible source; use Base.filter for other collections."))
    ct = _to_columns(data)
    names = _column_names(ct)
    n = _check_lengths(ct)
    mask = Vector{Bool}(undef, n)
    for i in 1:n
        row = _row_namedtuple(ct, i, names)
        mask[i] = predicate(row)
    end
    return _subset_mask(ct, mask)
end

function arrange(data, cols...; rev::Bool=false)
    ct = _to_columns(data)
    n = _check_lengths(ct)
    n <= 1 && return ct
    if isempty(cols)
        perm = collect(1:n)
    else
        order_cols = Symbol[]
        descending = Bool[]
        for spec in cols
            if spec isa Pair
                col = Symbol(first(spec))
                dir = last(spec)
                desc = dir === :desc || dir === :descending || dir === :reverse || dir === false
            else
                col = Symbol(spec)
                desc = false
            end
            if !(col in propertynames(ct))
                throw(ArgumentError("Column '$(col)' not found"))
            end
            push!(order_cols, col)
            push!(descending, desc)
        end
        perm = collect(1:n)
        values = [ct[sym] for sym in order_cols]
        sort!(perm, lt = (a, b) -> begin
            for (vec, desc) in zip(values, descending)
                va = vec[a]
                vb = vec[b]
                if va == vb
                    continue
                end
                return desc ? vb < va : va < vb
            end
            return false
        end)
    end
    if rev
        perm = reverse(perm)
    end
    names = _column_names(ct)
    columns = ntuple(i -> ct[names[i]][perm], length(names))
    return NamedTuple{names}(columns)
end

function groupby(data, cols::Symbol...)
    return groupby(data, collect(cols))
end

function groupby(data, cols::AbstractVector{Symbol})
    isempty(cols) && throw(ArgumentError("groupby requires at least one grouping column"))
    ct = _to_columns(data)
    n = _check_lengths(ct)

    for col in cols
        if !(col in propertynames(ct))
            throw(ArgumentError("Grouping column '$(col)' not found"))
        end
    end

    groups = Dict{NamedTuple, Vector{Int}}()
    names = Tuple(cols)
    for i in 1:n
        key = NamedTuple{names}(ntuple(j -> ct[names[j]][i], length(names)))
        if haskey(groups, key)
            push!(groups[key], i)
        else
            groups[key] = [i]
        end
    end

    key_list = collect(keys(groups))
    order = sortperm(key_list, by = string)
    key_list = key_list[order]
    idx_list = [groups[key_list[i]] for i in 1:length(key_list)]
    return GroupedTable(ct, collect(names), key_list, idx_list)
end

function mutate(data; kwargs...)
    ct = _to_columns(data)
    names = collect(_column_names(ct))
    columns = Dict{Symbol, Any}()
    for name in names
        columns[name] = ct[name]
    end
    n = _check_lengths(ct)

    for (new_name, spec) in pairs(kwargs)
        new_sym = Symbol(new_name)
        current = NamedTuple{Tuple(names)}(Tuple(columns[s] for s in names))
        coldata = if spec isa Function
            spec(current)
        else
            spec
        end
        coldata isa AbstractVector ||
            throw(ArgumentError("mutate requires vector results; got $(typeof(coldata)) for $(new_sym)"))
        length(coldata) == n ||
            throw(ArgumentError("mutate column '$(new_sym)' has length $(length(coldata)) but expected $(n)"))
        columns[new_sym] = coldata
        new_sym in names || push!(names, new_sym)
    end

    return NamedTuple{Tuple(names)}(Tuple(columns[s] for s in names))
end

function _compute_summary(spec, subgroup::NamedTuple)
    if spec isa Function
        return spec(subgroup)
    elseif spec isa Pair{Symbol, Function}
        col = first(spec)
        func = last(spec)
        return func(subgroup[col])
    elseif spec isa Pair{Tuple, Function}
        cols = first(spec)
        func = last(spec)
        return func(map(c -> subgroup[c], cols)...)
    else
        throw(ArgumentError("Unsupported summarise specification of type $(typeof(spec))"))
    end
end

function summarise(gt::GroupedTable; kwargs...)
    m = length(gt)
    keycols = gt.keycols

    key_data = Dict{Symbol, Vector{Any}}()
    for col in keycols
        key_data[col] = Vector{Any}(undef, m)
    end

    for (i, key) in enumerate(gt.keys)
        for col in keycols
            key_data[col][i] = key[col]
        end
    end

    summary_data = Dict{Symbol, Vector{Any}}()
    for (name, spec) in pairs(kwargs)
        summary_data[Symbol(name)] = Vector{Any}(undef, m)
    end

    for (i, idxs) in enumerate(gt.indices)
        subgroup = _subset_indices(gt.data, idxs)
        for (name, spec) in pairs(kwargs)
            summary_data[Symbol(name)][i] = _compute_summary(spec, subgroup)
        end
    end

    names = Symbol[]
    cols = Vector{Any}()
    for col in keycols
        push!(names, col)
        push!(cols, _finalize_column(key_data[col]))
    end
    for (name, data) in pairs(summary_data)
        push!(names, name)
        push!(cols, _finalize_column(data))
    end

    return _assemble(names, cols)
end

summarize(gt::GroupedTable; kwargs...) = summarise(gt; kwargs...)

function pivot_longer(data;
                      id_cols::Union{Symbol, AbstractVector{Symbol}} = Symbol[],
                      value_cols::Union{Symbol, AbstractVector{Symbol}} = Symbol[],
                      names_to::Symbol = _DEFAULT_NAMES_TO,
                      values_to::Symbol = _DEFAULT_VALUES_TO)
    ct = _to_columns(data)
    cols = collect(_column_names(ct))

    ids = id_cols isa Symbol ? Symbol[id_cols] : collect(id_cols)
    vals = value_cols isa Symbol ? Symbol[value_cols] : collect(value_cols)

    if isempty(ids) && isempty(vals)
        ids = Symbol[cols[1]]
    end

    if isempty(vals)
        vals = [c for c in cols if !(c in ids)]
    elseif isempty(ids)
        ids = [c for c in cols if !(c in vals)]
    end

    isempty(vals) && throw(ArgumentError("pivot_longer requires at least one value column"))

    n = _check_lengths(ct)
    total = n * length(vals)

    id_data = Dict{Symbol, Vector{Any}}()
    for id in ids
        id in propertynames(ct) || throw(ArgumentError("ID column '$(id)' not found"))
        id_data[id] = Vector{Any}(undef, total)
    end

    for val in vals
        val in propertynames(ct) || throw(ArgumentError("Value column '$(val)' not found"))
    end

    name_col = Vector{String}(undef, total)
    value_type = promote_type(map(val -> eltype(ct[val]), vals)...)
    value_col = Vector{value_type}(undef, total)

    idx = 1
    for val in vals
        column = ct[val]
        val_name = String(val)
        for row in 1:n
            for id in ids
                id_data[id][idx] = ct[id][row]
            end
            name_col[idx] = val_name
            value_col[idx] = convert(value_type, column[row])
            idx += 1
        end
    end

    out_names = Symbol[]
    out_cols = Vector{Any}()
    for id in ids
        push!(out_names, id)
        push!(out_cols, _finalize_column(id_data[id]))
    end
    push!(out_names, names_to)
    push!(out_cols, name_col)
    push!(out_names, values_to)
    push!(out_cols, value_col)

    return _assemble(out_names, out_cols)
end

function pivot_wider(data;
                     names_from::Symbol,
                     values_from::Symbol,
                     id_cols::Union{Symbol, AbstractVector{Symbol}} = Symbol[],
                     fill = missing,
                     sort_names::Bool = false)
    ct = _to_columns(data)
    all_names = Set(propertynames(ct))

    names_from == values_from &&
        throw(ArgumentError("names_from and values_from must refer to different columns"))

    names_from in all_names ||
        throw(ArgumentError("Column '$(names_from)' not found (names_from)"))
    values_from in all_names ||
        throw(ArgumentError("Column '$(values_from)' not found (values_from)"))

    ids = id_cols isa Symbol ? Symbol[id_cols] : collect(id_cols)
    if isempty(ids)
        for name in propertynames(ct)
            if name != names_from && name != values_from
                push!(ids, name)
            end
        end
    end

    if length(ids) != length(unique(ids))
        throw(ArgumentError("Duplicate id columns specified in pivot_wider"))
    end

    for id in ids
        id in all_names || throw(ArgumentError("ID column '$(id)' not found"))
    end

    names_from in ids && throw(ArgumentError("names_from column cannot also be listed as an id column"))
    values_from in ids && throw(ArgumentError("values_from column cannot also be listed as an id column"))

    n = _check_lengths(ct)
    if n == 0
        out_names = Symbol[]
        out_cols = Vector{Any}()
        for id in ids
            T = eltype(ct[id])
            push!(out_names, id)
            push!(out_cols, Vector{T}(undef, 0))
        end
        return _assemble(out_names, out_cols)
    end

    names_values = ct[names_from]
    values_values = ct[values_from]

    id_vectors = Dict{Symbol, Vector{Any}}()
    for id in ids
        id_vectors[id] = Vector{Any}()
    end

    key_index = Dict{Tuple, Int}()
    row_values = Vector{Vector{Any}}()
    name_order = Symbol[]
    name_index = Dict{Symbol, Int}()

    for i in 1:n
        key_tuple = tuple((ct[id][i] for id in ids)...)
        row_idx = get(key_index, key_tuple, 0)
        if row_idx == 0
            row_idx = length(row_values) + 1
            key_index[key_tuple] = row_idx
            new_row = Vector{Any}(undef, length(name_order))
            fill!(new_row, _PIVOT_SENTINEL)
            push!(row_values, new_row)
            for (id_sym, value) in zip(ids, key_tuple)
                push!(id_vectors[id_sym], value)
            end
        end
        name_val = names_values[i]
        if ismissing(name_val)
            throw(ArgumentError("names_from column contains missing at row $(i)"))
        end
        name_sym = Symbol(string(name_val))
        if !haskey(name_index, name_sym)
            push!(name_order, name_sym)
            name_index[name_sym] = length(name_order)
            for row_vec in row_values
                push!(row_vec, _PIVOT_SENTINEL)
            end
        end
        col_idx = name_index[name_sym]
        current = row_values[row_idx][col_idx]
        if current !== _PIVOT_SENTINEL
            throw(ArgumentError("Duplicate entries encountered for key $(key_tuple) and name $(name_sym)"))
        end
        row_values[row_idx][col_idx] = values_values[i]
    end

    if sort_names && !isempty(name_order)
        perm = sortperm(name_order, by = string)
        name_order = name_order[perm]
        for i in 1:length(row_values)
            row_values[i] = row_values[i][perm]
        end
    end

    out_names = Symbol[]
    out_cols = Vector{Any}()

    row_count = length(row_values)

    for id in ids
        length(id_vectors[id]) == row_count ||
            error("pivot_wider internal consistency error: mismatched row counts for column $(id)")
    end

    for id in ids
        push!(out_names, id)
        push!(out_cols, _finalize_column(id_vectors[id]))
    end

    if isempty(name_order)
        return _assemble(out_names, out_cols)
    end

    for (j, name_sym) in enumerate(name_order)
        col_data = Vector{Any}(undef, row_count)
        for i in 1:row_count
            value = row_values[i][j]
            if value === _PIVOT_SENTINEL
                col_data[i] = fill
            else
                col_data[i] = value
            end
        end
        push!(out_names, name_sym)
    push!(out_cols, _finalize_column(col_data))
    end

    return _assemble(out_names, out_cols)
end

"""
    glimpse(data; maxrows=5, io=stdout)

Display a compact summary of a Tables.jl-compatible object (similar to dplyr's
`glimpse`), showing column names, types, and a few sample values.

Supports both plain column tables and `GroupedTable`.
"""
function glimpse(data; maxrows::Integer = 5, maxgroups::Integer = 3, io::IO = stdout)
    if data isa GroupedTable
        return glimpse(io, data; maxrows=maxrows, maxgroups=maxgroups)
    end
    glimpse(io, data; maxrows=maxrows)
end

function glimpse(io::IO, data; maxrows::Integer = 5)
    ct = _to_columns(data)
    names = _column_names(ct)
    nrows = _check_lengths(ct)
    ncols = length(names)

    println(io, "Table glimpse")
    println(io, "  Rows: ", nrows)
    println(io, "  Columns: ", ncols)

    preview = clamp(maxrows, 0, nrows)
    for name in names
        col = ct[name]
        T = eltype(col)
        print(io, "  ", rpad(string(name), 20), ":: ", T, "  ")
        if preview == 0
            println(io, "[]")
            continue
        end

        values = Vector{Any}(undef, preview)
        for i in 1:preview
            values[i] = col[i]
        end

        snippets = join(repr.(values), ", ")
        if preview < nrows
            snippets = snippets * ", …"
        end
        println(io, "[", snippets, "]")
    end
    return nothing
end

function glimpse(io::IO, gt::GroupedTable; maxrows::Integer = 5, maxgroups::Integer = 3)
    n_groups = length(gt)
    println(io, "GroupedTable glimpse")
    println(io, "  Groups: ", n_groups)
    println(io, "  Key columns: ", isempty(gt.keycols) ? "(none)" : join(string.(gt.keycols), ", "))

    if n_groups == 0
        println(io, "  (no groups)")
        return nothing
    end

    sizes = [length(idx) for idx in gt.indices]
    total_rows = sum(sizes)
    avg_size = total_rows / n_groups
    println(io, "  Rows: ", total_rows, " (avg ", round(avg_size; digits=2),
            ", min ", minimum(sizes), ", max ", maximum(sizes), ")")

    sample = min(n_groups, maxgroups)
    for i in 1:sample
        key = gt.keys[i]
        key_str = isempty(gt.keycols) ? "(all rows)" : string(key)
        println(io, "  Group ", i, ": ", key_str, " (", sizes[i], " rows)")
        subgroup = _subset_indices(gt.data, gt.indices[i])
        io2 = IOBuffer()
        glimpse(io2, subgroup; maxrows=maxrows)
        seekstart(io2)
        for line in eachline(io2)
            println(io, "    ", line)
        end
    end
    if n_groups > sample
        println(io, "  … and ", n_groups - sample, " more groups")
    end
    return nothing
end
