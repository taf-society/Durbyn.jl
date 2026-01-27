const _DEFAULT_NAMES_TO = :variable
const _DEFAULT_VALUES_TO = :value

struct _PivotSentinel end
const _PIVOT_SENTINEL = _PivotSentinel()

struct GroupedTable{CT, KT}
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

"""
    _finalize_column(vec::Vector; eltype_hint::Union{Type, Nothing}=nothing)

Convert a Vector{Any} to a properly typed vector.
If `eltype_hint` is provided and vec is empty, returns Vector{eltype_hint}().
"""
function _finalize_column(vec::Vector; eltype_hint::Union{Type, Nothing}=nothing)
    if isempty(vec)
        # Use eltype_hint if provided, otherwise Any
        T = eltype_hint !== nothing ? eltype_hint : Any
        return Vector{T}()
    end
    # Use promote_type for proper Union handling (e.g., Missing + Int64 → Union{Missing, Int64})
    T = reduce(promote_type, map(typeof, vec))
    out = Vector{T}(undef, length(vec))
    for i in eachindex(vec)
        out[i] = convert(T, vec[i])
    end
    return out
end

"""
    _namedtuple_lt(a::NamedTuple, b::NamedTuple)

Compare two NamedTuples lexicographically using `isless`.
Handles missing values (sorts last) and provides proper numeric/date ordering.
Throws an error for mixed types that can't be compared with `isless`.
"""
function _namedtuple_lt(a::NamedTuple, b::NamedTuple)
    for i in 1:length(a)
        va = a[i]
        vb = b[i]
        # Handle missing values - sort them last
        va_miss = ismissing(va)
        vb_miss = ismissing(vb)
        if va_miss && vb_miss
            continue
        elseif va_miss
            return false  # a (missing) comes after b
        elseif vb_miss
            return true   # a comes before b (missing)
        end
        # Use isless for proper type-aware comparison
        if isequal(va, vb)
            continue
        end
        # isless - error on incompatible types (no silent string fallback)
        try
            return isless(va, vb)
        catch e
            throw(ArgumentError(
                "Cannot compare values of types $(typeof(va)) and $(typeof(vb)). " *
                "Column contains mixed types that cannot be ordered."))
        end
    end
    return false  # equal
end

"""
    select(data, specs...)

Select and optionally rename columns from a Tables.jl-compatible data source.

# Arguments
- `data`: Any Tables.jl-compatible source (e.g., `NamedTuple`, `DataFrame`, `CSV.File`)
- `specs...`: Column specifications, either:
  - Column names as `Symbol`s to select
  - `Pair`s like `:new_name => :old_name` to rename columns

# Returns
A `NamedTuple` with the selected (and optionally renamed) columns.

# Examples
```julia
using Durbyn.TableOps

tbl = (a = [1, 2, 3], b = [4, 5, 6], c = [7, 8, 9])

# Select specific columns
select(tbl, :a, :c)
# Output: (a = [1, 2, 3], c = [7, 8, 9])

# Rename while selecting
select(tbl, :x => :a, :y => :b)
# Output: (x = [1, 2, 3], y = [4, 5, 6])
```
"""
function select(data, specs...)
    ct = _to_columns(data)
    isempty(specs) && return ct
    available = Set(_column_names(ct))

    names_out = Symbol[]
    columns_out = Vector{Any}()
    seen_source = Set{Symbol}()  # Track source columns to avoid selecting same column twice
    seen_output = Set{Symbol}()

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
        if old_name in seen_source
            throw(ArgumentError("Source column '$old_name' selected multiple times"))
        end
        if new_name in seen_output
            throw(ArgumentError("Duplicate output column name: '$new_name'"))
        end
        push!(seen_source, old_name)
        push!(seen_output, new_name)
        push!(names_out, new_name)
        push!(columns_out, ct[old_name])
    end

    return _assemble(names_out, columns_out)
end

"""
    query(data, predicate::Function)

Filter rows based on a predicate function.

# Arguments
- `data`: A Tables.jl-compatible data source
- `predicate`: A function that takes a `NamedTuple` row and returns a `Bool`

# Returns
A `NamedTuple` containing only rows where `predicate` returns `true`.

# Examples
```julia
using Durbyn.TableOps

tbl = (id = [1, 2, 3, 4], value = [10, 20, 15, 30])

# Filter rows where value > 15
query(tbl, row -> row.value > 15)
# Output: (id = [2, 4], value = [20, 30])

# Combine multiple conditions
query(tbl, row -> row.id > 1 && row.value < 25)
# Output: (id = [2, 3], value = [20, 15])
```
"""
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

"""
    arrange(data, cols...; rev::Bool=false)

Sort rows by one or more columns.

# Arguments
- `data`: A Tables.jl-compatible data source
- `cols...`: Column specifications:
  - Column names as `Symbol`s for ascending order
  - `Pair`s like `:col => :asc` for explicit ascending order
  - `Pair`s like `:col => :desc` for descending order
  - Accepted direction values: `:asc`, `:ascending` (ascending); `:desc`, `:descending`, `:reverse` (descending)
- `rev`: If `true`, reverse sort direction for all columns (missing values still sort last)

Missing values always sort last, regardless of sort direction.

# Returns
A `NamedTuple` with rows sorted according to the specified columns.

# Examples
```julia
using Durbyn.TableOps

tbl = (name = ["Alice", "Bob", "Charlie"], age = [25, 30, 20])

# Sort by age (ascending)
arrange(tbl, :age)
# Output: (name = ["Charlie", "Alice", "Bob"], age = [20, 25, 30])

# Sort by age descending
arrange(tbl, :age => :desc)
# Output: (name = ["Bob", "Alice", "Charlie"], age = [30, 25, 20])

# Multi-column sort
tbl2 = (group = ["A", "B", "A", "B"], value = [3, 1, 2, 4])
arrange(tbl2, :group, :value => :desc)
# Output: (group = ["A", "A", "B", "B"], value = [3, 2, 4, 1])
```
"""
function arrange(data, cols...; rev::Bool=false)
    ct = _to_columns(data)
    n = _check_lengths(ct)
    n <= 1 && return ct
    if isempty(cols)
        perm = collect(1:n)
        if rev
            perm = reverse(perm)
        end
    else
        order_cols = Symbol[]
        descending = Bool[]
        for spec in cols
            if spec isa Pair
                col = Symbol(first(spec))
                dir = last(spec)
                # :asc/:ascending = ascending (false), :desc/:descending/:reverse = descending (true)
                if dir === :asc || dir === :ascending
                    desc = false
                elseif dir === :desc || dir === :descending || dir === :reverse
                    desc = true
                else
                    throw(ArgumentError("Invalid sort direction '$dir'. Use :asc, :ascending, :desc, :descending, or :reverse"))
                end
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
        if rev
            descending = .!descending
        end
        perm = collect(1:n)
        values = [ct[sym] for sym in order_cols]
        # Use stable sort with missing-aware comparisons (missing always sorts last)
        sort!(perm, alg=Base.Sort.MergeSort, lt = (a, b) -> begin
            for (vec, desc) in zip(values, descending)
                va = vec[a]
                vb = vec[b]
                # Handle missing values: missing always sorts last (regardless of asc/desc)
                va_miss = ismissing(va)
                vb_miss = ismissing(vb)
                if va_miss && vb_miss
                    continue  # Both missing, equal
                elseif va_miss
                    return false  # a (missing) comes after b, so a is NOT less than b
                elseif vb_miss
                    return true   # a comes before b (missing), so a IS less than b
                end
                # Both non-missing: use isequal for proper equality, isless for ordering
                if isequal(va, vb)
                    continue
                end
                # isless - error on incompatible types (no silent string fallback)
                try
                    return desc ? isless(vb, va) : isless(va, vb)
                catch e
                    throw(ArgumentError(
                        "Cannot compare values of types $(typeof(va)) and $(typeof(vb)). " *
                        "Column contains mixed types that cannot be ordered."))
                end
            end
            return false  # All equal, maintain original order (stable)
        end)
    end
    names = _column_names(ct)
    columns = ntuple(i -> ct[names[i]][perm], length(names))
    return NamedTuple{names}(columns)
end

"""
    groupby(data, cols...)

Group rows by unique combinations of values in specified columns.

# Arguments
- `data`: A Tables.jl-compatible data source
- `cols...`: One or more column names (as `Symbol`s) to group by

# Returns
A `GroupedTable` object containing:
- Groups organized by unique key combinations
- Metadata about grouping columns
- Indices for accessing each group's rows

# Examples
```julia
using Durbyn.TableOps

tbl = (category = ["A", "B", "A", "B", "A"],
       value = [1, 2, 3, 4, 5])

# Group by category
gt = groupby(tbl, :category)
# Output: GroupedTable(2 groups by category)

# Use with summarise to compute group statistics
summarise(gt, mean_value = :value => mean, count = data -> length(data.value))
```

See also: [`summarise`](@ref), [`summarize`](@ref)
"""
function groupby(data, cols::Symbol...)
    return groupby(data, collect(cols))
end

function groupby(data, cols::AbstractVector{Symbol})
    isempty(cols) && throw(ArgumentError("groupby requires at least one grouping column"))

    # Check for duplicate grouping columns
    if length(cols) != length(Set(cols))
        throw(ArgumentError("Duplicate grouping columns: $(cols)"))
    end

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
    # Sort using proper comparison (not string-based) for correct numeric/date ordering
    order = sortperm(key_list, lt = _namedtuple_lt)
    key_list = key_list[order]
    idx_list = [groups[key_list[i]] for i in 1:length(key_list)]
    return GroupedTable(ct, collect(names), key_list, idx_list)
end

"""
    mutate(data; kwargs...)

Add new columns or modify existing columns.

# Arguments
- `data`: A Tables.jl-compatible data source
- `kwargs...`: Named arguments where:
  - Name is the new/modified column name
  - Value is either:
    - A function taking the current table and returning a vector
    - A vector of values (must match row count)

# Returns
A `NamedTuple` with the original columns plus the new/modified columns.

# Examples
```julia
using Durbyn.TableOps

tbl = (a = [1, 2, 3], b = [4, 5, 6])

# Add a new column computed from existing columns
mutate(tbl, c = data -> data.a .+ data.b)
# Output: (a = [1, 2, 3], b = [4, 5, 6], c = [5, 7, 9])

# Add multiple columns
mutate(tbl,
    sum = data -> data.a .+ data.b,
    product = data -> data.a .* data.b)

# Modify existing column
mutate(tbl, a = data -> data.a .* 2)
# Output: (a = [2, 4, 6], b = [4, 5, 6])
```
"""
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
    elseif spec isa Pair
        col = first(spec)
        func = last(spec)
        if col isa Symbol
            return func(subgroup[col])
        elseif col isa Tuple
            return func(map(c -> subgroup[c], col)...)
        else
            throw(ArgumentError("First element of Pair must be Symbol or Tuple, got $(typeof(col))"))
        end
    else
        throw(ArgumentError("Unsupported summarise specification of type $(typeof(spec))"))
    end
end

"""
    _infer_summary_type(spec, data::NamedTuple) -> Type

Attempt to infer the return type of a summary specification on empty data.
Falls back to Any if type cannot be determined.
Throws KeyError for invalid column names (validation errors should propagate).
"""
function _infer_summary_type(spec, data::NamedTuple)
    if spec isa Pair
        col = first(spec)
        func = last(spec)
        if col isa Symbol
            # Validate column exists (throws KeyError if not)
            col in propertynames(data) || throw(KeyError(col))
            # Try calling func on empty typed vector
            col_eltype = eltype(data[col])
            empty_col = Vector{col_eltype}()
            try
                result = func(empty_col)
                return typeof(result)
            catch
                # Function call failed (e.g., mean on empty), fall back to Any
                return Any
            end
        elseif col isa Tuple
            # Validate all columns exist
            for c in col
                c in propertynames(data) || throw(KeyError(c))
            end
            return Any  # Complex multi-column specs fall back to Any
        end
    end
    # For functions on full group, fall back to Any
    return Any
end

"""
    summarise(gt::GroupedTable; kwargs...)
    summarize(gt::GroupedTable; kwargs...)

Compute summary statistics for each group in a `GroupedTable`.

# Arguments
- `gt`: A `GroupedTable` created by `groupby`
- `kwargs...`: Named summary specifications where each value can be:
  - A function taking the group's data (NamedTuple) and returning a scalar
  - A `Pair` of `:column => function` to apply function to a specific column
  - A `Pair` of `(cols...) => function` to apply function to multiple columns

# Returns
A `NamedTuple` with:
- Key columns from the original grouping
- Summary columns specified in `kwargs`

# Examples
```julia
using Durbyn.TableOps
using Statistics

tbl = (category = ["A", "B", "A", "B", "A"],
       value = [10, 20, 30, 40, 50])

gt = groupby(tbl, :category)

# Compute mean for each group
summarise(gt, mean_value = :value => mean)
# Output: (category = ["A", "B"], mean_value = [30.0, 30.0])

# Multiple summaries
summarise(gt,
    mean_val = :value => mean,
    count = data -> length(data.value),
    sum_val = :value => sum)

# Custom function on entire group
summarise(gt, range = data -> maximum(data.value) - minimum(data.value))
```

Note: `summarize` is an alias for `summarise`.

See also: [`groupby`](@ref)
"""
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

    # Preserve kwargs order by storing names in order
    summary_names = Symbol[]
    summary_data = Dict{Symbol, Vector{Any}}()
    for (name, _) in pairs(kwargs)
        sym = Symbol(name)
        push!(summary_names, sym)
        summary_data[sym] = Vector{Any}(undef, m)
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
        if m == 0
            # Empty result: preserve eltype from original data
            original_eltype = eltype(gt.data[col])
            push!(cols, Vector{original_eltype}())
        else
            push!(cols, _finalize_column(key_data[col]))
        end
    end
    # Iterate in preserved order, not Dict order
    for (name, spec) in zip(summary_names, [kwargs[Symbol(n)] for n in summary_names])
        push!(names, name)
        if m == 0
            # Try to infer return type for empty data
            result_type = _infer_summary_type(spec, gt.data)
            push!(cols, Vector{result_type}())
        else
            push!(cols, _finalize_column(summary_data[name]))
        end
    end

    return _assemble(names, cols)
end

"""
    summarize(gt::GroupedTable; kwargs...)

American English spelling alias for `summarise`. See [`summarise`](@ref) for details.
"""
summarize(gt::GroupedTable; kwargs...) = summarise(gt; kwargs...)

"""
    pivot_longer(data; id_cols=Symbol[], value_cols=Symbol[],
                 names_to=:variable, values_to=:value)

Transform data from wide format to long format by pivoting columns into rows.

# Arguments
- `data`: A Tables.jl-compatible data source
- `id_cols`: Column(s) to keep as identifiers (not pivoted)
- `value_cols`: Column(s) to pivot into rows (if empty, all non-id columns are used)
- `names_to`: Name for the new column containing original column names (default: `:variable`)
- `values_to`: Name for the new column containing values (default: `:value`)

When both `id_cols` and `value_cols` are empty, all columns are pivoted into values (no id columns).
`id_cols` and `value_cols` must not overlap.

# Returns
A `NamedTuple` in long format with:
- All `id_cols` repeated for each pivoted column
- A `names_to` column with original column names
- A `values_to` column with corresponding values

# Examples
```julia
using Durbyn.TableOps

# Wide format data
wide = (date = ["2024-01", "2024-02"],
        A = [100, 110],
        B = [200, 220],
        C = [300, 330])

# Pivot to long format
long = pivot_longer(wide, id_cols=:date, names_to=:series, values_to=:value)
# Output: (date = ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02"],
#          series = ["A", "B", "C", "A", "B", "C"],
#          value = [100, 200, 300, 110, 220, 330])

# Specify which columns to pivot
pivot_longer(wide, id_cols=:date, value_cols=[:A, :B])
```

See also: [`pivot_wider`](@ref)
"""
function pivot_longer(data;
                      id_cols::Union{Symbol, AbstractVector{Symbol}} = Symbol[],
                      value_cols::Union{Symbol, AbstractVector{Symbol}} = Symbol[],
                      names_to::Symbol = _DEFAULT_NAMES_TO,
                      values_to::Symbol = _DEFAULT_VALUES_TO)
    ct = _to_columns(data)
    cols = collect(_column_names(ct))

    ids = id_cols isa Symbol ? Symbol[id_cols] : collect(id_cols)
    vals = value_cols isa Symbol ? Symbol[value_cols] : collect(value_cols)

    # Check for duplicate id_cols
    if length(ids) != length(unique(ids))
        seen = Set{Symbol}()
        for id in ids
            if id in seen
                throw(ArgumentError("Duplicate id column: '$id'"))
            end
            push!(seen, id)
        end
    end

    # Check for duplicate value_cols
    if length(vals) != length(unique(vals))
        seen = Set{Symbol}()
        for v in vals
            if v in seen
                throw(ArgumentError("Duplicate value column: '$v'"))
            end
            push!(seen, v)
        end
    end

    # Validate names_to/values_to collision first (applies to all cases)
    if names_to == values_to
        throw(ArgumentError("names_to and values_to cannot be the same: '$names_to'"))
    end

    # Handle empty table case (after basic validation)
    if isempty(cols)
        # Validate that specified id_cols/value_cols are empty (can't reference non-existent columns)
        if !isempty(ids)
            throw(ArgumentError("ID column '$(ids[1])' not found in empty table"))
        end
        if !isempty(vals)
            throw(ArgumentError("Value column '$(vals[1])' not found in empty table"))
        end
        # Return empty table with correct schema
        return NamedTuple{(names_to, values_to)}((String[], Any[]))
    end

    # Check for overlap between id_cols and value_cols before defaults
    if !isempty(ids) && !isempty(vals)
        overlap = intersect(Set(ids), Set(vals))
        if !isempty(overlap)
            throw(ArgumentError("Columns cannot be both id and value columns: $(collect(overlap))"))
        end
    end

    # When both are empty, treat all columns as value columns (no id columns)
    # This matches the docstring: "all non-id columns become value columns"
    if isempty(ids) && isempty(vals)
        vals = copy(cols)  # All columns become values
    elseif isempty(vals)
        vals = [c for c in cols if !(c in ids)]
    elseif isempty(ids)
        ids = [c for c in cols if !(c in vals)]
    end

    isempty(vals) && throw(ArgumentError("pivot_longer requires at least one value column"))

    # Check for name collisions between names_to/values_to and id_cols
    if names_to in ids
        throw(ArgumentError("names_to='$names_to' conflicts with id column of same name"))
    end
    if values_to in ids
        throw(ArgumentError("values_to='$values_to' conflicts with id column of same name"))
    end

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

    # Row-major order: for each row, output all value columns
    # This matches tidyr's pivot_longer behavior
    idx = 1
    for row in 1:n
        for val in vals
            for id in ids
                id_data[id][idx] = ct[id][row]
            end
            name_col[idx] = String(val)
            value_col[idx] = convert(value_type, ct[val][row])
            idx += 1
        end
    end

    out_names = Symbol[]
    out_cols = Vector{Any}()
    for id in ids
        push!(out_names, id)
        # Preserve eltype from original id column
        id_eltype = eltype(ct[id])
        push!(out_cols, _finalize_column(id_data[id]; eltype_hint=id_eltype))
    end
    push!(out_names, names_to)
    push!(out_cols, name_col)
    push!(out_names, values_to)
    push!(out_cols, value_col)

    return _assemble(out_names, out_cols)
end

"""
    pivot_wider(data; names_from::Symbol, values_from::Symbol,
                id_cols=Symbol[], fill=missing, sort_names=false)

Transform data from long format to wide format by spreading rows into columns.

# Arguments
- `data`: A Tables.jl-compatible data source in long format
- `names_from`: Column containing values to become new column names
- `values_from`: Column containing values to populate the new columns
- `id_cols`: Column(s) that uniquely identify each row (if empty, uses all other columns)
- `fill`: Value to use for missing combinations (default: `missing`)
- `sort_names`: If `true`, sort new column names alphabetically (default: `false`)

# Returns
A `NamedTuple` in wide format with:
- All `id_cols` preserved
- New columns created from unique values in `names_from`
- Values from `values_from` distributed into the new columns

# Examples
```julia
using Durbyn.TableOps

# Long format data
long = (date = ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02"],
        series = ["A", "B", "C", "A", "B", "C"],
        value = [100, 200, 300, 110, 220, 330])

# Pivot to wide format
wide = pivot_wider(long, names_from=:series, values_from=:value, id_cols=:date)
# Output: (date = ["2024-01", "2024-02"],
#          A = [100, 110],
#          B = [200, 220],
#          C = [300, 330])

# Sort column names alphabetically
pivot_wider(long, names_from=:series, values_from=:value,
            id_cols=:date, sort_names=true)

# Handle missing values
incomplete = (id = [1, 1, 2], category = ["A", "B", "A"], val = [10, 20, 30])
pivot_wider(incomplete, names_from=:category, values_from=:val, fill=0)
# Output: (id = [1, 2], A = [10, 30], B = [20, 0])
```

See also: [`pivot_longer`](@ref)
"""
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
    name_original = Dict{Symbol, Any}()  # Track original value for each Symbol

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
            name_original[name_sym] = name_val
            for row_vec in row_values
                push!(row_vec, _PIVOT_SENTINEL)
            end
        else
            # Check if distinct values stringify to the same Symbol
            orig = name_original[name_sym]
            if !isequal(orig, name_val) && typeof(orig) != typeof(name_val)
                throw(ArgumentError("Distinct values $(repr(orig)) ($(typeof(orig))) and $(repr(name_val)) ($(typeof(name_val))) in names_from column both produce column name ':$(name_sym)'. Ensure names_from values are of consistent type."))
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

    # Check for collisions between names_from values and id_cols
    id_set = Set(ids)
    for name_sym in name_order
        if name_sym in id_set
            throw(ArgumentError("pivot_wider: generated column name '$name_sym' from names_from conflicts with id column"))
        end
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

function glimpse(data::ForecastModelCollection; maxrows::Integer = 5, include_failures::Bool = false, io::IO = stdout)
    tbl = as_table(data; include_failures=include_failures)
    return glimpse(io, tbl; maxrows=maxrows)
end

function glimpse(io::IO, data::ForecastModelCollection; maxrows::Integer = 5, include_failures::Bool = false)
    tbl = as_table(data; include_failures=include_failures)
    return glimpse(io, tbl; maxrows=maxrows)
end

function glimpse(io::IO, panel::PanelData; maxrows::Integer = 5)
    println(io, "PanelData glimpse")
    println(io, "  Groups: ", isempty(panel.groups) ? "(none)" : join(string.(panel.groups), ", "))
    println(io, "  Date: ", isnothing(panel.date) ? "(none)" : string(panel.date))

    # Show frequency and m
    if panel.frequency !== nothing
        println(io, "  Frequency: ", panel.frequency)
    end
    if panel.m !== nothing
        if panel.m isa Vector
            println(io, "  Seasonal period(s) m: [", join(panel.m, ", "), "]")
        else
            println(io, "  Seasonal period m: ", panel.m)
        end
    else
        println(io, "  Seasonal period m: (not set)")
    end

    # Show target if set
    if panel.target !== nothing
        println(io, "  Target: ", panel.target)
    end

    # Show preprocessing metadata if any
    has_preprocessing = panel.time_fill_meta !== nothing ||
                        panel.target_meta !== nothing ||
                        !isempty(panel.xreg_meta)

    if has_preprocessing
        println(io, "  Preprocessing:")
        if panel.time_fill_meta !== nothing
            meta = panel.time_fill_meta
            println(io, "    Time grid: ", meta.n_added, " rows added (step: ", meta.step, ")")
        end
        if panel.target_meta !== nothing
            meta = panel.target_meta
            pct = round(100 * meta.n_imputed / meta.n_total, digits=1)
            println(io, "    Target (:", panel.target, "): ", meta.n_imputed, " imputed (", pct, "%) via :", meta.strategy)
        end
        for (col, meta) in panel.xreg_meta
            pct = round(100 * meta.n_imputed / meta.n_total, digits=1)
            println(io, "    Exog (:", col, "): ", meta.n_imputed, " imputed (", pct, "%) via :", meta.strategy)
        end
    end

    data = Tables.columntable(panel.data)
    glimpse(io, data; maxrows=maxrows)
end

glimpse(panel::PanelData; kwargs...) = glimpse(stdout, panel; kwargs...)

# =============================================================================
# Additional dplyr-style functions
# =============================================================================

"""
    rename(data, specs...)

Rename columns in a table.

# Arguments
- `data`: A Tables.jl-compatible data source
- `specs...`: Rename specifications as `Pair`s: `:new_name => :old_name`

# Returns
A `NamedTuple` with renamed columns. Columns not mentioned in `specs` are kept unchanged.

# Examples
```julia
using Durbyn.TableOps

tbl = (a = [1, 2, 3], b = [4, 5, 6], c = [7, 8, 9])

# Rename single column
rename(tbl, :x => :a)
# Output: (x = [1, 2, 3], b = [4, 5, 6], c = [7, 8, 9])

# Rename multiple columns
rename(tbl, :x => :a, :y => :b)
# Output: (x = [1, 2, 3], y = [4, 5, 6], c = [7, 8, 9])
```

See also: [`select`](@ref)
"""
function rename(data, specs::Pair{Symbol, Symbol}...)
    ct = _to_columns(data)
    isempty(specs) && return ct
    available = Set(_column_names(ct))

    rename_map = Dict{Symbol, Symbol}()
    new_names_in_specs = Set{Symbol}()
    old_names_in_specs = Set{Symbol}()
    for (new_name, old_name) in specs
        old_name in available ||
            throw(ArgumentError("Column '$(old_name)' not found"))
        if new_name in new_names_in_specs
            throw(ArgumentError("Duplicate output column name in rename: '$new_name'"))
        end
        if old_name in old_names_in_specs
            throw(ArgumentError("Duplicate source column in rename: '$old_name'"))
        end
        push!(new_names_in_specs, new_name)
        push!(old_names_in_specs, old_name)
        rename_map[old_name] = new_name
    end

    names_out = Symbol[]
    columns_out = Vector{Any}()
    seen_output = Set{Symbol}()

    for name in _column_names(ct)
        new_name = get(rename_map, name, name)
        if new_name in seen_output
            throw(ArgumentError("Duplicate output column name: '$new_name' (column '$name' would conflict with an earlier column)"))
        end
        push!(seen_output, new_name)
        push!(names_out, new_name)
        push!(columns_out, ct[name])
    end

    return _assemble(names_out, columns_out)
end

"""
    distinct(data, cols...; keep_all::Bool=false)

Remove duplicate rows based on specified columns.

# Arguments
- `data`: A Tables.jl-compatible data source
- `cols...`: Column names to consider for uniqueness (if empty, uses all columns)
- `keep_all`: If `true`, keep all columns in output; if `false`, only keep `cols` (default: `false`)

# Returns
A `NamedTuple` with duplicate rows removed. When duplicates exist, the first occurrence is kept.

# Examples
```julia
using Durbyn.TableOps

tbl = (a = [1, 1, 2, 2, 3], b = [1, 1, 2, 3, 3], c = [10, 20, 30, 40, 50])

# Distinct by all columns
distinct(tbl)
# Output: (a = [1, 1, 2, 2, 3], b = [1, 1, 2, 3, 3], c = [10, 20, 30, 40, 50])

# Distinct by specific column
distinct(tbl, :a)
# Output: (a = [1, 2, 3],)

# Distinct by specific column but keep all columns
distinct(tbl, :a; keep_all=true)
# Output: (a = [1, 2, 3], b = [1, 2, 3], c = [10, 30, 50])
```
"""
function distinct(data, cols::Symbol...; keep_all::Bool=false)
    ct = _to_columns(data)
    all_names = _column_names(ct)
    n = _check_lengths(ct)

    key_cols = isempty(cols) ? collect(all_names) : collect(cols)

    for col in key_cols
        col in all_names ||
            throw(ArgumentError("Column '$(col)' not found"))
    end

    # Julia's Set uses isequal for hashing, so it handles missing/NaN correctly
    seen = Set{Any}()
    keep_indices = Int[]

    for i in 1:n
        key = if length(key_cols) == 1
            ct[key_cols[1]][i]
        else
            tuple((ct[c][i] for c in key_cols)...)
        end

        if !(key in seen)
            push!(seen, key)
            push!(keep_indices, i)
        end
    end

    output_cols = keep_all || isempty(cols) ? all_names : Tuple(key_cols)
    names_out = Symbol[]
    columns_out = Vector{Any}()

    for name in output_cols
        push!(names_out, name)
        push!(columns_out, ct[name][keep_indices])
    end

    return _assemble(names_out, columns_out)
end

"""
    bind_rows(tables...)

Combine multiple tables vertically (row-wise).

# Arguments
- `tables...`: Two or more Tables.jl-compatible data sources

# Returns
A `NamedTuple` with all rows from all input tables stacked vertically.
Tables may have different columns; missing columns will be filled with `missing`.

# Examples
```julia
using Durbyn.TableOps

tbl1 = (a = [1, 2], b = [3, 4])
tbl2 = (a = [5, 6], b = [7, 8])

bind_rows(tbl1, tbl2)
# Output: (a = [1, 2, 5, 6], b = [3, 4, 7, 8])

# With mismatched columns (fills with missing)
tbl3 = (a = [1, 2], b = [3, 4])
tbl4 = (a = [5, 6], c = [7, 8])
bind_rows(tbl3, tbl4)
# Output: (a = [1, 2, 5, 6], b = Union{Missing, Int64}[3, 4, missing, missing],
#          c = Union{Missing, Int64}[missing, missing, 7, 8])
```
"""
function bind_rows(tables...)
    isempty(tables) && return NamedTuple()
    length(tables) == 1 && return _to_columns(tables[1])

    cts = [_to_columns(t) for t in tables]

    # Validate each input table has consistent column lengths
    for (i, ct) in enumerate(cts)
        try
            _check_lengths(ct)
        catch e
            throw(ArgumentError("Table $i has mismatched column lengths: $(e.msg)"))
        end
    end

    all_cols = Symbol[]
    col_set = Set{Symbol}()
    col_eltypes = Dict{Symbol, Type}()

    # Pre-compute column name sets for all tables (avoid repeated Set creation)
    ct_col_sets = [Set(_column_names(ct)) for ct in cts]

    for (ct, ct_cols) in zip(cts, ct_col_sets)
        for name in _column_names(ct)
            if !(name in col_set)
                push!(all_cols, name)
                push!(col_set, name)
            end
            # Track eltype from input columns
            col_eltype = eltype(ct[name])
            if haskey(col_eltypes, name)
                # Promote types when same column appears in multiple tables
                col_eltypes[name] = promote_type(col_eltypes[name], col_eltype)
            else
                col_eltypes[name] = col_eltype
            end
        end
    end

    # Columns that don't appear in all tables need Union{..., Missing}
    for col in all_cols
        appears_in_all = all(col in ct_cols for ct_cols in ct_col_sets)
        if !appears_in_all
            col_eltypes[col] = Union{col_eltypes[col], Missing}
        end
    end

    total_rows = sum(_nrows(ct) for ct in cts)

    # Handle empty result - return typed empty vectors
    if total_rows == 0
        names_out = all_cols
        columns_out = [Vector{col_eltypes[col]}() for col in all_cols]
        return _assemble(names_out, columns_out)
    end

    # Use computed eltypes for result columns (not Vector{Any})
    result_cols = Dict{Symbol, Vector}()
    for col in all_cols
        result_cols[col] = Vector{col_eltypes[col]}(undef, total_rows)
    end

    row_offset = 0
    for (ct, ct_names) in zip(cts, ct_col_sets)
        n = _nrows(ct)
        for col in all_cols
            if col in ct_names
                for i in 1:n
                    result_cols[col][row_offset + i] = ct[col][i]
                end
            else
                for i in 1:n
                    result_cols[col][row_offset + i] = missing
                end
            end
        end
        row_offset += n
    end

    names_out = all_cols
    columns_out = Vector{Vector}(undef, length(all_cols))
    for (i, col) in enumerate(all_cols)
        columns_out[i] = result_cols[col]
    end

    return _assemble(names_out, columns_out)
end

"""
    ungroup(gt::GroupedTable)

Remove grouping from a `GroupedTable`, returning the underlying data.

# Arguments
- `gt`: A `GroupedTable` created by `groupby`

# Returns
A `NamedTuple` containing all the original data without grouping.

# Examples
```julia
using Durbyn.TableOps

tbl = (category = ["A", "B", "A", "B"], value = [1, 2, 3, 4])
gt = groupby(tbl, :category)

# Remove grouping
ungroup(gt)
# Output: (category = ["A", "B", "A", "B"], value = [1, 2, 3, 4])
```

See also: [`groupby`](@ref)
"""
ungroup(gt::GroupedTable) = gt.data

# =============================================================================
# Column selection helpers
# =============================================================================

"""
    ColumnSelector

Abstract type for column selection specifications used with `select` and `across`.
"""
abstract type ColumnSelector end

"""
    AllOf <: ColumnSelector

Select columns by a vector of names.

# Examples
```julia
using Durbyn.TableOps

tbl = (a = [1, 2], b = [3, 4], c = [5, 6])
select(tbl, all_of([:a, :c]))
# Output: (a = [1, 2], c = [5, 6])
```
"""
struct AllOf <: ColumnSelector
    cols::Vector{Symbol}
end

"""
    all_of(cols)

Create a column selector that selects columns by name from a vector.

# Arguments
- `cols`: A vector of column names (as `Symbol`s or `String`s)

# Returns
An `AllOf` selector for use with `select`.

# Examples
```julia
using Durbyn.TableOps

tbl = (a = [1, 2], b = [3, 4], c = [5, 6])
cols_to_select = [:a, :c]
select(tbl, all_of(cols_to_select))
# Output: (a = [1, 2], c = [5, 6])
```
"""
all_of(cols::AbstractVector) = AllOf(Symbol.(cols))
all_of(cols::Symbol...) = AllOf(collect(cols))

"""
    Everything <: ColumnSelector

Select all columns.

# Examples
```julia
using Durbyn.TableOps

tbl = (a = [1, 2], b = [3, 4])
select(tbl, everything())
# Output: (a = [1, 2], b = [3, 4])
```
"""
struct Everything <: ColumnSelector end

"""
    everything()

Create a column selector that selects all columns.

# Returns
An `Everything` selector for use with `select`.

# Examples
```julia
using Durbyn.TableOps

tbl = (a = [1, 2], b = [3, 4])

# Useful when combined with other selections to reorder
select(tbl, :b, everything())  # puts :b first, then all others
```
"""
everything() = Everything()

function _expand_selector(selector::AllOf, available::Tuple)
    avail_set = Set(available)
    for col in selector.cols
        col in avail_set ||
            throw(ArgumentError("Column '$(col)' not found"))
    end
    return selector.cols
end

function _expand_selector(::Everything, available::Tuple)
    return collect(available)
end

function _expand_selector(spec::Symbol, available::Tuple)
    spec in available ||
        throw(ArgumentError("Column '$(spec)' not found"))
    return [spec]
end

function _expand_selector(spec::Pair, available::Tuple)
    old_name = Symbol(last(spec))
    old_name in available ||
        throw(ArgumentError("Column '$(old_name)' not found"))
    return [spec]
end

# Enhanced select to support column selectors
function select(data, specs::Union{Symbol, Pair, ColumnSelector}...)
    ct = _to_columns(data)
    isempty(specs) && return ct
    available = _column_names(ct)

    names_out = Symbol[]
    columns_out = Vector{Any}()
    seen_source = Set{Symbol}()  # Track source columns to avoid selecting same column twice
    seen_output = Set{Symbol}()  # Track output names to prevent duplicates

    for spec in specs
        # Determine if this is an auto-expanding selector (skip duplicates) or explicit spec (error on duplicates)
        is_selector = spec isa ColumnSelector
        expanded = _expand_selector(spec, available)
        for item in expanded
            if item isa Pair
                new_name = Symbol(first(item))
                old_name = Symbol(last(item))
            else
                new_name = Symbol(item)
                old_name = Symbol(item)
            end
            if old_name in seen_source
                if is_selector
                    # Selectors like everything() silently skip already-selected columns
                    continue
                else
                    # Explicit specs error on duplicate source columns
                    throw(ArgumentError("Source column '$old_name' selected multiple times"))
                end
            end
            if new_name in seen_output
                throw(ArgumentError("Duplicate output column name: '$new_name'"))
            end
            push!(seen_source, old_name)
            push!(seen_output, new_name)
            push!(names_out, new_name)
            push!(columns_out, ct[old_name])
        end
    end

    return _assemble(names_out, columns_out)
end

"""
    Across

Specification for applying functions across multiple columns in `mutate` or `summarise`.
"""
struct Across
    cols::Union{Vector{Symbol}, ColumnSelector}
    fns::Vector{Pair{Symbol, Function}}
end

"""
    across(cols, fns...)

Apply functions across multiple columns.

# Arguments
- `cols`: Column specification (vector of symbols, `all_of(...)`, or `everything()`)
- `fns...`: One or more `Pair`s of `:name => function` to apply to each column

# Returns
An `Across` specification for use with `mutate` or `summarise`.

# Examples
```julia
using Durbyn.TableOps
using Statistics

tbl = (a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0], group = ["x", "x", "y"])

# Apply mean to multiple columns in summarise
gt = groupby(tbl, :group)
summarise(gt, across([:a, :b], :mean => mean))
# Output: (group = ["x", "y"], a_mean = [1.5, 3.0], b_mean = [4.5, 6.0])

# Multiple functions
summarise(gt, across([:a, :b], :mean => mean, :sum => sum))

# With everything() selector (excludes grouping columns automatically)
summarise(gt, across(everything(), :mean => mean))
```
"""
function across(cols::Union{AbstractVector, ColumnSelector}, fns::Pair{Symbol, <:Function}...)
    col_spec = cols isa AbstractVector ? Symbol.(cols) : cols
    return Across(col_spec, collect(fns))
end

across(cols::Symbol, fns::Pair{Symbol, <:Function}...) = across([cols], fns...)

function _expand_across_cols(ac::Across, available::Tuple)
    if ac.cols isa Vector{Symbol}
        avail_set = Set(available)
        for col in ac.cols
            col in avail_set ||
                throw(ArgumentError("Column '$(col)' not found"))
        end
        return ac.cols
    else
        return _expand_selector(ac.cols, available)
    end
end

# Extended summarise to support Across
function summarise(gt::GroupedTable, ac::Across)
    m = length(gt)
    keycols = gt.keycols
    keycol_set = Set(keycols)

    available = _column_names(gt.data)
    target_cols = [c for c in _expand_across_cols(ac, available) if !(c in keycol_set)]

    key_data = Dict{Symbol, Vector{Any}}()
    for col in keycols
        key_data[col] = Vector{Any}(undef, m)
    end

    for (i, key) in enumerate(gt.keys)
        for col in keycols
            key_data[col][i] = key[col]
        end
    end

    # Check for output name collisions
    seen_names = Set{Symbol}(keycols)
    for col in target_cols
        for (fn_name, _) in ac.fns
            out_name = Symbol("$(col)_$(fn_name)")
            if out_name in seen_names
                throw(ArgumentError("Across produces duplicate output column name '$out_name'. Consider renaming columns or functions to avoid collision."))
            end
            push!(seen_names, out_name)
        end
    end

    summary_data = Dict{Symbol, Vector{Any}}()
    for col in target_cols
        for (fn_name, _) in ac.fns
            out_name = Symbol("$(col)_$(fn_name)")
            summary_data[out_name] = Vector{Any}(undef, m)
        end
    end

    for (i, idxs) in enumerate(gt.indices)
        subgroup = _subset_indices(gt.data, idxs)
        for col in target_cols
            for (fn_name, fn) in ac.fns
                out_name = Symbol("$(col)_$(fn_name)")
                summary_data[out_name][i] = fn(subgroup[col])
            end
        end
    end

    names_out = Symbol[]
    cols_out = Vector{Any}()

    for col in keycols
        push!(names_out, col)
        # Preserve eltype from original data for key columns
        key_eltype = eltype(gt.data[col])
        push!(cols_out, _finalize_column(key_data[col]; eltype_hint=key_eltype))
    end

    for col in target_cols
        for (fn_name, fn) in ac.fns
            out_name = Symbol("$(col)_$(fn_name)")
            push!(names_out, out_name)
            # Try to infer summary eltype from function on empty typed vector
            col_eltype = eltype(gt.data[col])
            summary_eltype = try
                typeof(fn(Vector{col_eltype}()))
            catch
                Any
            end
            push!(cols_out, _finalize_column(summary_data[out_name]; eltype_hint=summary_eltype))
        end
    end

    return _assemble(names_out, cols_out)
end

# Extended mutate to support Across
function mutate(data, ac::Across)
    ct = _to_columns(data)
    available = _column_names(ct)
    target_cols = _expand_across_cols(ac, available)
    n = _check_lengths(ct)

    names_out = collect(available)
    columns = Dict{Symbol, Any}()
    for name in available
        columns[name] = ct[name]
    end

    # Check for output name collisions before computing
    seen_names = Set{Symbol}(available)
    for col in target_cols
        for (fn_name, _) in ac.fns
            out_name = Symbol("$(col)_$(fn_name)")
            if out_name in seen_names
                throw(ArgumentError("Across produces duplicate output column name '$out_name'. Consider renaming columns or functions to avoid collision."))
            end
            push!(seen_names, out_name)
        end
    end

    for col in target_cols
        for (fn_name, fn) in ac.fns
            out_name = Symbol("$(col)_$(fn_name)")
            result = fn(ct[col])
            if !(result isa AbstractVector)
                result = fill(result, n)
            elseif length(result) != n
                throw(DimensionMismatch(
                    "across function '$fn_name' on column '$col' returned vector of length $(length(result)), expected $n"))
            end
            columns[out_name] = result
            push!(names_out, out_name)
        end
    end

    return NamedTuple{Tuple(names_out)}(Tuple(columns[s] for s in names_out))
end

# =============================================================================
# tidyr-style functions
# =============================================================================

"""
    separate(data, col::Symbol; into::Vector{Symbol}, sep::Union{String, Regex, Char}=" ", remove::Bool=true, convert::Bool=false)

Separate a character column into multiple columns.

# Arguments
- `data`: A Tables.jl-compatible data source
- `col`: Column name to separate
- `into`: Names for the new columns
- `sep`: Separator pattern (default: `" "`)
- `remove`: If `true`, remove the input column (default: `true`)
- `convert`: If `true`, attempt to convert to numeric types (default: `false`)

# Returns
A `NamedTuple` with the original column split into multiple new columns.

# Examples
```julia
using Durbyn.TableOps

tbl = (id = [1, 2, 3], name = ["John Doe", "Jane Smith", "Bob Wilson"])

# Split name into first and last
separate(tbl, :name; into=[:first, :last], sep=" ")
# Output: (id = [1, 2, 3], first = ["John", "Jane", "Bob"], last = ["Doe", "Smith", "Wilson"])

# Keep original column
separate(tbl, :name; into=[:first, :last], sep=" ", remove=false)

# With numeric conversion
tbl2 = (id = [1, 2], coords = ["10,20", "30,40"])
separate(tbl2, :coords; into=[:x, :y], sep=",", convert=true)
# Output: (id = [1, 2], x = [10, 30], y = [20, 40])
```

See also: [`unite`](@ref)
"""
function separate(data, col::Symbol;
                  into::Vector{Symbol},
                  sep::Union{String, Regex, Char}=" ",
                  remove::Bool=true,
                  convert::Bool=false)
    ct = _to_columns(data)
    available = _column_names(ct)

    col in available || throw(ArgumentError("Column '$(col)' not found"))

    # Check for duplicate names in 'into'
    if length(into) != length(unique(into))
        seen = Set{Symbol}()
        for name in into
            if name in seen
                throw(ArgumentError("Duplicate column name in 'into': '$name'"))
            end
            push!(seen, name)
        end
    end

    # Check for collision with existing columns (excluding the column being separated if remove=true)
    existing_cols = remove ? Set(c for c in available if c != col) : Set(available)
    for name in into
        if name in existing_cols
            throw(ArgumentError("Output column '$name' conflicts with existing column"))
        end
    end

    n = _check_lengths(ct)

    new_cols = Dict{Symbol, Vector{Any}}()
    for name in into
        new_cols[name] = Vector{Any}(undef, n)
    end

    source_col = ct[col]
    for i in 1:n
        val = source_col[i]
        if ismissing(val)
            for name in into
                new_cols[name][i] = missing
            end
        else
            parts = split(string(val), sep)
            for (j, name) in enumerate(into)
                if j <= length(parts)
                    new_cols[name][i] = parts[j]
                else
                    new_cols[name][i] = missing
                end
            end
        end
    end

    if convert
        for name in into
            vec = new_cols[name]
            # Try to convert to numbers
            all_numeric = true
            for v in vec
                if !ismissing(v)
                    try
                        parse(Float64, v)
                    catch
                        all_numeric = false
                        break
                    end
                end
            end
            if all_numeric
                new_cols[name] = [ismissing(v) ? missing : parse(Float64, v) for v in vec]
            end
        end
    end

    names_out = Symbol[]
    columns_out = Vector{Any}()

    for name in available
        if name == col
            if !remove
                push!(names_out, name)
                push!(columns_out, ct[name])
            end
            for new_name in into
                push!(names_out, new_name)
                push!(columns_out, _finalize_column(new_cols[new_name]))
            end
        else
            push!(names_out, name)
            push!(columns_out, ct[name])
        end
    end

    return _assemble(names_out, columns_out)
end

"""
    unite(data, new_col::Symbol, cols::Symbol...; sep::String="_", remove::Bool=true)

Combine multiple columns into a single character column.

# Arguments
- `data`: A Tables.jl-compatible data source
- `new_col`: Name for the new combined column
- `cols...`: Columns to combine
- `sep`: Separator to use between values (default: `"_"`)
- `remove`: If `true`, remove the input columns (default: `true`)

# Returns
A `NamedTuple` with the specified columns combined into one.

# Examples
```julia
using Durbyn.TableOps

tbl = (id = [1, 2, 3], year = [2020, 2021, 2022], month = [1, 6, 12])

# Combine year and month
unite(tbl, :date, :year, :month; sep="-")
# Output: (id = [1, 2, 3], date = ["2020-1", "2021-6", "2022-12"])

# Keep original columns
unite(tbl, :date, :year, :month; sep="-", remove=false)
# Output: (id = [1, 2, 3], year = [2020, 2021, 2022], month = [1, 6, 12], date = ["2020-1", "2021-6", "2022-12"])
```

See also: [`separate`](@ref)
"""
function unite(data, new_col::Symbol, cols::Symbol...; sep::String="_", remove::Bool=true)
    ct = _to_columns(data)
    available = _column_names(ct)
    n = _check_lengths(ct)

    cols_list = collect(cols)
    for col in cols_list
        col in available || throw(ArgumentError("Column '$(col)' not found"))
    end

    # Check for collision with existing columns
    # When remove=true, new_col can match a column being removed
    # When remove=false, new_col cannot match any existing column
    cols_set = Set(cols_list)
    conflicting_cols = remove ? Set(c for c in available if c ∉ cols_set) : Set(available)
    if new_col in conflicting_cols
        throw(ArgumentError("Output column '$new_col' conflicts with existing column"))
    end

    combined = Vector{Any}(undef, n)
    for i in 1:n
        parts = String[]
        has_missing = false
        for col in cols_list
            val = ct[col][i]
            if ismissing(val)
                has_missing = true
                break
            end
            push!(parts, string(val))
        end
        combined[i] = has_missing ? missing : join(parts, sep)
    end

    names_out = Symbol[]
    columns_out = Vector{Any}()

    inserted = false
    for name in available
        if name in cols_set
            if !inserted
                push!(names_out, new_col)
                push!(columns_out, _finalize_column(combined))
                inserted = true
            end
            if !remove
                push!(names_out, name)
                push!(columns_out, ct[name])
            end
        else
            push!(names_out, name)
            push!(columns_out, ct[name])
        end
    end

    if !inserted
        push!(names_out, new_col)
        push!(columns_out, _finalize_column(combined))
    end

    return _assemble(names_out, columns_out)
end

"""
    fill_missing(data, cols::Symbol...; direction::Symbol=:down)

Fill missing values in columns using the previous or next non-missing value.

# Arguments
- `data`: A Tables.jl-compatible data source
- `cols...`: Columns to fill (if empty, fills all columns)
- `direction`: Fill direction - `:down` (default), `:up`, `:downup`, or `:updown`

# Returns
A `NamedTuple` with missing values filled.

# Examples
```julia
using Durbyn.TableOps

tbl = (id = [1, 2, 3, 4, 5],
       value = [10, missing, missing, 40, missing])

# Fill down (forward fill)
fill_missing(tbl, :value)
# Output: (id = [1, 2, 3, 4, 5], value = [10, 10, 10, 40, 40])

# Fill up (backward fill)
fill_missing(tbl, :value; direction=:up)
# Output: (id = [1, 2, 3, 4, 5], value = [10, 40, 40, 40, missing])

# Fill both directions (down first, then up)
fill_missing(tbl, :value; direction=:downup)
# Output: (id = [1, 2, 3, 4, 5], value = [10, 10, 10, 40, 40])
```
"""
function fill_missing(data, cols::Symbol...; direction::Symbol=:down)
    ct = _to_columns(data)
    available = _column_names(ct)
    _check_lengths(ct)

    direction in (:down, :up, :downup, :updown) ||
        throw(ArgumentError("direction must be :down, :up, :downup, or :updown"))

    cols_to_fill = isempty(cols) ? collect(available) : collect(cols)
    cols_set = Set(cols_to_fill)

    for col in cols_to_fill
        col in available || throw(ArgumentError("Column '$(col)' not found"))
    end

    function fill_down!(vec)
        isempty(vec) && return  # Handle empty vectors
        has_valid = false
        local last_valid  # Declare without initialization
        for i in 1:length(vec)
            if !ismissing(vec[i])
                last_valid = vec[i]
                has_valid = true
            elseif has_valid
                vec[i] = last_valid
            end
        end
    end

    function fill_up!(vec)
        isempty(vec) && return  # Handle empty vectors
        has_valid = false
        local next_valid  # Declare without initialization
        for i in length(vec):-1:1
            if !ismissing(vec[i])
                next_valid = vec[i]
                has_valid = true
            elseif has_valid
                vec[i] = next_valid
            end
        end
    end

    names_out = Symbol[]
    columns_out = Vector{Any}()

    for name in available
        if name in cols_set
            vec = collect(ct[name])
            if direction == :down
                fill_down!(vec)
            elseif direction == :up
                fill_up!(vec)
            elseif direction == :downup
                fill_down!(vec)
                fill_up!(vec)
            elseif direction == :updown
                fill_up!(vec)
                fill_down!(vec)
            end
            push!(names_out, name)
            push!(columns_out, _finalize_column(vec))
        else
            push!(names_out, name)
            push!(columns_out, ct[name])
        end
    end

    return _assemble(names_out, columns_out)
end

"""
    complete(data, cols::Symbol...; fill_value=missing)

Complete a data table with all combinations of specified columns.

# Arguments
- `data`: A Tables.jl-compatible data source
- `cols...`: Columns to expand (creates all unique combinations)
- `fill_value`: Value to use for new rows (default: `missing`)

# Returns
A `NamedTuple` expanded to include all combinations of values in the specified columns.

# Examples
```julia
using Durbyn.TableOps

tbl = (year = [2020, 2020, 2021],
       quarter = [1, 2, 1],
       value = [100, 200, 150])

# Complete all year-quarter combinations
complete(tbl, :year, :quarter)
# Output: (year = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
#          quarter = [1, 2, 1, 2, 1, 2, 1, 2],
#          value = [100, 200, missing, missing, 150, missing, missing, missing])

# With custom fill value
complete(tbl, :year, :quarter; fill_value=0)
```

See also: [`fill`](@ref)
"""
function complete(data, cols::Symbol...; fill_value=missing)
    ct = _to_columns(data)
    available = _column_names(ct)
    n = _check_lengths(ct)

    cols_list = collect(cols)
    isempty(cols_list) && return ct

    for col in cols_list
        col in available || throw(ArgumentError("Column '$(col)' not found"))
    end

    unique_values = Dict{Symbol, Vector{Any}}()
    for col in cols_list
        unique_values[col] = unique(ct[col])
    end

    # Use Tuple for cartesian product results (immutable, value-based hashing)
    function cartesian_product(arrays)
        if length(arrays) == 1
            # Use (v,) to create single-element tuple; Tuple(v) would try to iterate v
            return [(v,) for v in arrays[1]]
        end
        result = Vector{Tuple}()
        rest = cartesian_product(arrays[2:end])
        for v in arrays[1]
            for r in rest
                push!(result, (v, r...))
            end
        end
        return result
    end

    all_combos = cartesian_product([unique_values[c] for c in cols_list])

    # Julia's Set uses isequal for hashing, so it handles missing/NaN correctly
    existing_keys = Set{Tuple}()
    for i in 1:n
        key = Tuple(ct[c][i] for c in cols_list)
        push!(existing_keys, key)
    end

    # Filter to combos not in existing keys
    new_rows = [combo for combo in all_combos if !(combo in existing_keys)]

    total_rows = n + length(new_rows)

    result_cols = Dict{Symbol, Vector{Any}}()
    for name in available
        result_cols[name] = Vector{Any}(undef, total_rows)
        for i in 1:n
            result_cols[name][i] = ct[name][i]
        end
    end

    cols_set = Set(cols_list)
    for (j, combo) in enumerate(new_rows)
        row_idx = n + j
        for (k, col) in enumerate(cols_list)
            result_cols[col][row_idx] = combo[k]
        end
        for name in available
            if !(name in cols_set)
                result_cols[name][row_idx] = fill_value
            end
        end
    end

    names_out = collect(available)
    # Preserve eltypes from original data (only widen to Union{Missing} if rows were actually added)
    columns_out = Vector{Vector}(undef, length(names_out))
    rows_added = !isempty(new_rows)
    for (i, name) in enumerate(names_out)
        original_eltype = eltype(ct[name])
        # Only widen non-key columns to Union{fill_value type} if new rows were actually added
        hint_eltype = if name in cols_set
            original_eltype
        elseif rows_added
            Union{original_eltype, typeof(fill_value)}
        else
            original_eltype  # No new rows, keep original type
        end
        columns_out[i] = _finalize_column(result_cols[name]; eltype_hint=hint_eltype)
    end

    return _assemble(names_out, columns_out)
end

# =============================================================================
# Join functions
# =============================================================================

"""
    _resolve_join_keys(left, right, by)

Internal helper to resolve join keys from the `by` specification.
Returns (left_keys, right_keys) as vectors of Symbols.

Special cases:
- `by=nothing`: Auto-detect common columns
- `by=Symbol[]` or empty vector: Cross join (returns empty key vectors)
"""
function _resolve_join_keys(left::NamedTuple, right::NamedTuple, by)
    left_cols = _column_names(left)
    left_names = Set(left_cols)
    right_names = Set(_column_names(right))

    if by === nothing
        # Auto-detect common columns, preserving left table column order for determinism
        common = [c for c in left_cols if c in right_names]
        isempty(common) && throw(ArgumentError("No common columns found for join. Specify `by` explicitly."))
        return common, common
    elseif by isa Symbol
        by in left_names || throw(ArgumentError("Column '$(by)' not found in left table"))
        by in right_names || throw(ArgumentError("Column '$(by)' not found in right table"))
        return [by], [by]
    elseif by isa AbstractVector{Symbol}
        # Empty vector = cross join (dplyr behavior)
        if isempty(by)
            return Symbol[], Symbol[]
        end
        # Check for duplicate keys
        if length(by) != length(Set(by))
            throw(ArgumentError("Duplicate key columns in `by` specification: $(by)"))
        end
        for col in by
            col in left_names || throw(ArgumentError("Column '$(col)' not found in left table"))
            col in right_names || throw(ArgumentError("Column '$(col)' not found in right table"))
        end
        return collect(by), collect(by)
    elseif by isa AbstractVector && isempty(by)
        # Handle empty vectors of any type (e.g., Any[])
        return Symbol[], Symbol[]
    elseif by isa Pair
        # Normalize to Symbol
        left_key = Symbol(first(by))
        right_key = Symbol(last(by))
        left_key in left_names || throw(ArgumentError("Column '$(left_key)' not found in left table"))
        right_key in right_names || throw(ArgumentError("Column '$(right_key)' not found in right table"))
        return [left_key], [right_key]
    elseif by isa AbstractVector{<:Pair}
        left_keys = Symbol[]
        right_keys = Symbol[]
        for p in by
            # Normalize to Symbol
            lk, rk = Symbol(first(p)), Symbol(last(p))
            lk in left_names || throw(ArgumentError("Column '$(lk)' not found in left table"))
            rk in right_names || throw(ArgumentError("Column '$(rk)' not found in right table"))
            push!(left_keys, lk)
            push!(right_keys, rk)
        end
        # Check for duplicate keys
        if length(left_keys) != length(Set(left_keys))
            throw(ArgumentError("Duplicate left key columns in `by` specification"))
        end
        if length(right_keys) != length(Set(right_keys))
            throw(ArgumentError("Duplicate right key columns in `by` specification"))
        end
        return left_keys, right_keys
    else
        throw(ArgumentError("Invalid `by` specification. Use Symbol, Vector{Symbol}, Pair, or Vector{Pair}."))
    end
end

"""
    _build_key(ct::NamedTuple, keys::Vector{Symbol}, i::Int)

Build a key tuple from row i using the specified key columns.
"""
function _build_key(ct::NamedTuple, keys::Vector{Symbol}, i::Int)
    if length(keys) == 1
        return ct[keys[1]][i]
    else
        return tuple((ct[k][i] for k in keys)...)
    end
end

"""
    _build_right_index(right::NamedTuple, right_keys::Vector{Symbol})

Build a dictionary mapping key values to row indices in the right table.
"""
function _build_right_index(right::NamedTuple, right_keys::Vector{Symbol})
    n_right = _nrows(right)
    index = Dict{Any, Vector{Int}}()
    for i in 1:n_right
        key = _build_key(right, right_keys, i)
        if haskey(index, key)
            push!(index[key], i)
        else
            index[key] = [i]
        end
    end
    return index
end

"""
    _validate_join_output_names(out_names::Vector{Symbol})

Check for duplicate output column names after join suffix resolution.
Throws ArgumentError if duplicates are found.
"""
function _validate_join_output_names(out_names::Vector{Symbol})
    if length(out_names) != length(unique(out_names))
        seen = Set{Symbol}()
        for name in out_names
            if name in seen
                throw(ArgumentError("Join produces duplicate column name '$name'. Consider using different suffixes or renaming columns before joining."))
            end
            push!(seen, name)
        end
    end
end

"""
    inner_join(left, right; by=nothing, suffix=("_x", "_y"))

Join two tables, keeping only rows with matching keys in both tables.

# Arguments
- `left`: Left table (Tables.jl-compatible)
- `right`: Right table (Tables.jl-compatible)
- `by`: Join specification:
  - `nothing` (default): Join on all common column names
  - `Symbol`: Single column name present in both tables
  - `Vector{Symbol}`: Multiple column names present in both tables
  - `Pair{Symbol,Symbol}`: Different column names (`:left_col => :right_col`)
  - `Vector{Pair}`: Multiple pairs for different column names
- `suffix`: Tuple of suffixes for duplicate non-key columns (default: `("_x", "_y")`)

# Returns
A `NamedTuple` containing only rows where the key exists in both tables.

# Note
Uses Julia's `isequal` semantics for key matching: `missing` matches `missing` and
`NaN` matches `NaN`. This differs from SQL where `NULL` never equals `NULL`.

# Examples
```julia
using Durbyn.TableOps

left = (id = [1, 2, 3], x = [10, 20, 30])
right = (id = [2, 3, 4], y = [200, 300, 400])

inner_join(left, right, by=:id)
# Output: (id = [2, 3], x = [20, 30], y = [200, 300])

# Join on multiple columns
left2 = (a = [1, 1, 2], b = ["x", "y", "x"], val = [10, 20, 30])
right2 = (a = [1, 2], b = ["x", "x"], val2 = [100, 200])

inner_join(left2, right2, by=[:a, :b])
# Output: (a = [1, 2], b = ["x", "x"], val = [10, 30], val2 = [100, 200])

# Join with different column names
left3 = (id = [1, 2, 3], x = [10, 20, 30])
right3 = (key = [2, 3, 4], y = [200, 300, 400])

inner_join(left3, right3, by=:id => :key)
# Output: (id = [2, 3], x = [20, 30], y = [200, 300])
```

See also: [`left_join`](@ref), [`right_join`](@ref), [`full_join`](@ref)
"""
function inner_join(left, right; by=nothing, suffix::Tuple{String,String}=("_x", "_y"))
    ct_left = _to_columns(left)
    ct_right = _to_columns(right)

    # Validate column lengths
    _check_lengths(ct_left)
    _check_lengths(ct_right)

    left_keys, right_keys = _resolve_join_keys(ct_left, ct_right, by)
    is_cross_join = isempty(left_keys)

    n_left = _nrows(ct_left)
    n_right = _nrows(ct_right)
    left_names = _column_names(ct_left)
    right_names = _column_names(ct_right)
    right_key_set = Set(right_keys)

    # Determine output columns
    out_names = Symbol[]
    out_sources = Tuple{Symbol, Int, Symbol}[]  # (name, table (1=left, 2=right), original_name)

    for name in left_names
        push!(out_names, name)
        push!(out_sources, (name, 1, name))
    end

    left_key_set = Set(left_keys)
    for name in right_names
        if name in right_key_set
            continue  # Skip key columns from right (use left's)
        end
        out_name = name
        if name in left_names
            out_name = Symbol(string(name) * suffix[2])
            # Only rename left column if it's NOT a key column
            # Key columns should keep their original names
            if !(name in left_key_set)
                idx = findfirst(==(name), out_names)
                if idx !== nothing
                    renamed_left = Symbol(string(name) * suffix[1])
                    out_names[idx] = renamed_left
                    # Update the source entry for the left column
                    out_sources[idx] = (renamed_left, 1, name)
                end
            end
        end
        push!(out_names, out_name)
        push!(out_sources, (out_name, 2, name))
    end

    # Check for duplicate output names
    _validate_join_output_names(out_names)

    # Build result
    result_cols = Dict{Symbol, Vector{Any}}()
    for name in out_names
        result_cols[name] = Any[]
    end

    if is_cross_join
        # Cross join: every left row pairs with every right row
        for i in 1:n_left
            for j in 1:n_right
                for (out_name, table, orig_name) in out_sources
                    if table == 1
                        push!(result_cols[out_name], ct_left[orig_name][i])
                    else
                        push!(result_cols[out_name], ct_right[orig_name][j])
                    end
                end
            end
        end
    else
        right_index = _build_right_index(ct_right, right_keys)
        for i in 1:n_left
            left_key = _build_key(ct_left, left_keys, i)
            if haskey(right_index, left_key)
                for j in right_index[left_key]
                    for (out_name, table, orig_name) in out_sources
                        if table == 1
                            push!(result_cols[out_name], ct_left[orig_name][i])
                        else
                            push!(result_cols[out_name], ct_right[orig_name][j])
                        end
                    end
                end
            end
        end
    end

    columns_out = [_finalize_column(result_cols[name]) for name in out_names]
    return _assemble(collect(out_names), columns_out)
end

"""
    left_join(left, right; by=nothing, suffix=("_x", "_y"))

Join two tables, keeping all rows from the left table.

# Arguments
- `left`: Left table (Tables.jl-compatible)
- `right`: Right table (Tables.jl-compatible)
- `by`: Join specification (see `inner_join` for details)
- `suffix`: Tuple of suffixes for duplicate non-key columns (default: `("_x", "_y")`)

# Returns
A `NamedTuple` containing all rows from `left`, with matching data from `right`.
Non-matching rows have `missing` for columns from `right`.

# Note
Uses Julia's `isequal` semantics for key matching: `missing` matches `missing` and
`NaN` matches `NaN`. This differs from SQL where `NULL` never equals `NULL`.

# Examples
```julia
using Durbyn.TableOps

left = (id = [1, 2, 3], x = [10, 20, 30])
right = (id = [2, 3, 4], y = [200, 300, 400])

left_join(left, right, by=:id)
# Output: (id = [1, 2, 3], x = [10, 20, 30], y = [missing, 200, 300])
```

See also: [`inner_join`](@ref), [`right_join`](@ref), [`full_join`](@ref)
"""
function left_join(left, right; by=nothing, suffix::Tuple{String,String}=("_x", "_y"))
    ct_left = _to_columns(left)
    ct_right = _to_columns(right)

    # Validate column lengths
    _check_lengths(ct_left)
    _check_lengths(ct_right)

    left_keys, right_keys = _resolve_join_keys(ct_left, ct_right, by)
    is_cross_join = isempty(left_keys)

    n_left = _nrows(ct_left)
    n_right = _nrows(ct_right)
    left_names = _column_names(ct_left)
    right_names = _column_names(ct_right)
    right_key_set = Set(right_keys)

    # Determine output columns
    out_names = Symbol[]
    out_sources = Tuple{Symbol, Int, Symbol}[]

    for name in left_names
        push!(out_names, name)
        push!(out_sources, (name, 1, name))
    end

    left_key_set = Set(left_keys)
    right_only_cols = Symbol[]
    for name in right_names
        if name in right_key_set
            continue
        end
        out_name = name
        if name in left_names
            out_name = Symbol(string(name) * suffix[2])
            # Only rename left column if it's NOT a key column
            if !(name in left_key_set)
                idx = findfirst(==(name), out_names)
                if idx !== nothing
                    renamed_left = Symbol(string(name) * suffix[1])
                    out_names[idx] = renamed_left
                    out_sources[idx] = (renamed_left, 1, name)
                end
            end
        end
        push!(out_names, out_name)
        push!(out_sources, (out_name, 2, name))
        push!(right_only_cols, out_name)
    end

    # Check for duplicate output names
    _validate_join_output_names(out_names)

    # Build result
    result_cols = Dict{Symbol, Vector{Any}}()
    for name in out_names
        result_cols[name] = Any[]
    end

    if is_cross_join
        # Cross join: every left row pairs with every right row
        for i in 1:n_left
            for j in 1:n_right
                for (out_name, table, orig_name) in out_sources
                    if table == 1
                        push!(result_cols[out_name], ct_left[orig_name][i])
                    else
                        push!(result_cols[out_name], ct_right[orig_name][j])
                    end
                end
            end
        end
    else
        right_index = _build_right_index(ct_right, right_keys)
        for i in 1:n_left
            left_key = _build_key(ct_left, left_keys, i)
            if haskey(right_index, left_key)
                for j in right_index[left_key]
                    for (out_name, table, orig_name) in out_sources
                        if table == 1
                            push!(result_cols[out_name], ct_left[orig_name][i])
                        else
                            push!(result_cols[out_name], ct_right[orig_name][j])
                        end
                    end
                end
            else
                # No match - add left row with missing for right columns
                for (out_name, table, orig_name) in out_sources
                    if table == 1
                        push!(result_cols[out_name], ct_left[orig_name][i])
                    else
                        push!(result_cols[out_name], missing)
                    end
                end
            end
        end
    end

    columns_out = [_finalize_column(result_cols[name]) for name in out_names]
    return _assemble(collect(out_names), columns_out)
end

"""
    right_join(left, right; by=nothing, suffix=("_x", "_y"))

Join two tables, keeping all rows from the right table.

# Arguments
- `left`: Left table (Tables.jl-compatible)
- `right`: Right table (Tables.jl-compatible)
- `by`: Join specification (see `inner_join` for details)
- `suffix`: Tuple of suffixes for duplicate non-key columns (default: `("_x", "_y")`)

# Returns
A `NamedTuple` containing all rows from `right`, with matching data from `left`.
Non-matching rows have `missing` for columns from `left`.

# Note
Uses Julia's `isequal` semantics for key matching: `missing` matches `missing` and
`NaN` matches `NaN`. This differs from SQL where `NULL` never equals `NULL`.

# Examples
```julia
using Durbyn.TableOps

left = (id = [1, 2, 3], x = [10, 20, 30])
right = (id = [2, 3, 4], y = [200, 300, 400])

right_join(left, right, by=:id)
# Output: (id = [2, 3, 4], x = [20, 30, missing], y = [200, 300, 400])
```

See also: [`inner_join`](@ref), [`left_join`](@ref), [`full_join`](@ref)
"""
function right_join(left, right; by=nothing, suffix::Tuple{String,String}=("_x", "_y"))
    # right_join is just left_join with tables swapped
    # But we need to handle the key column naming carefully
    ct_left = _to_columns(left)
    ct_right = _to_columns(right)

    # Validate column lengths
    _check_lengths(ct_left)
    _check_lengths(ct_right)

    left_keys, right_keys = _resolve_join_keys(ct_left, ct_right, by)
    is_cross_join = isempty(left_keys)

    n_left = _nrows(ct_left)

    n_right = _nrows(ct_right)
    left_names = _column_names(ct_left)
    right_names = _column_names(ct_right)
    left_key_set = Set(left_keys)

    # Determine output columns - right columns first (preserve order), then left non-keys
    out_names = Symbol[]
    out_sources = Tuple{Symbol, Int, Symbol}[]

    right_key_set = Set(right_keys)
    key_map = Dict{Symbol, Symbol}()
    for (lk, rk) in zip(left_keys, right_keys)
        key_map[rk] = lk
    end

    # Add right columns in original order (keys renamed to left key names)
    # Track output names to detect collisions
    seen_out_names = Set{Symbol}()
    for name in right_names
        if name in right_key_set
            out_name = key_map[name]
        else
            out_name = name
            # Check if this non-key column collides with an already-added key name
            if out_name in seen_out_names
                out_name = Symbol(string(name) * suffix[2])
            end
        end
        push!(seen_out_names, out_name)
        push!(out_names, out_name)
        push!(out_sources, (out_name, 2, name))
    end

    right_output_set = Set(out_names)

    # Add left non-key columns, suffixing if they collide with right outputs
    for name in left_names
        if name in left_key_set
            continue
        end
        out_name = name
        if out_name in right_output_set
            out_name = Symbol(string(name) * suffix[1])
        end
        push!(out_names, out_name)
        push!(out_sources, (out_name, 1, name))
    end

    # Check for duplicate output names
    _validate_join_output_names(out_names)

    # Build result
    result_cols = Dict{Symbol, Vector{Any}}()
    for name in out_names
        result_cols[name] = Any[]
    end

    if is_cross_join
        # Cross join: every right row pairs with every left row
        for j in 1:n_right
            for i in 1:n_left
                for (out_name, table, orig_name) in out_sources
                    if table == 1
                        push!(result_cols[out_name], ct_left[orig_name][i])
                    else
                        push!(result_cols[out_name], ct_right[orig_name][j])
                    end
                end
            end
        end
    else
        left_index = _build_right_index(ct_left, left_keys)
        for j in 1:n_right
            right_key = _build_key(ct_right, right_keys, j)
            if haskey(left_index, right_key)
                for i in left_index[right_key]
                    for (out_name, table, orig_name) in out_sources
                        if table == 1
                            push!(result_cols[out_name], ct_left[orig_name][i])
                        else
                            push!(result_cols[out_name], ct_right[orig_name][j])
                        end
                    end
                end
            else
                # No match - add right row with missing for left columns
                for (out_name, table, orig_name) in out_sources
                    if table == 1
                        push!(result_cols[out_name], missing)
                    else
                        push!(result_cols[out_name], ct_right[orig_name][j])
                    end
                end
            end
        end
    end

    columns_out = [_finalize_column(result_cols[name]) for name in out_names]
    return _assemble(collect(out_names), columns_out)
end

"""
    full_join(left, right; by=nothing, suffix=("_x", "_y"))

Join two tables, keeping all rows from both tables.

# Arguments
- `left`: Left table (Tables.jl-compatible)
- `right`: Right table (Tables.jl-compatible)
- `by`: Join specification (see `inner_join` for details)
- `suffix`: Tuple of suffixes for duplicate non-key columns (default: `("_x", "_y")`)

# Returns
A `NamedTuple` containing all rows from both tables.
Non-matching rows have `missing` for columns from the other table.

# Note
Uses Julia's `isequal` semantics for key matching: `missing` matches `missing` and
`NaN` matches `NaN`. This differs from SQL where `NULL` never equals `NULL`.

# Examples
```julia
using Durbyn.TableOps

left = (id = [1, 2, 3], x = [10, 20, 30])
right = (id = [2, 3, 4], y = [200, 300, 400])

full_join(left, right, by=:id)
# Output: (id = [1, 2, 3, 4], x = [10, 20, 30, missing], y = [missing, 200, 300, 400])
```

See also: [`inner_join`](@ref), [`left_join`](@ref), [`right_join`](@ref)
"""
function full_join(left, right; by=nothing, suffix::Tuple{String,String}=("_x", "_y"))
    ct_left = _to_columns(left)
    ct_right = _to_columns(right)

    # Validate column lengths
    _check_lengths(ct_left)
    _check_lengths(ct_right)

    left_keys, right_keys = _resolve_join_keys(ct_left, ct_right, by)
    is_cross_join = isempty(left_keys)

    n_left = _nrows(ct_left)
    n_right = _nrows(ct_right)
    left_names = _column_names(ct_left)
    right_names = _column_names(ct_right)
    right_key_set = Set(right_keys)

    # Determine output columns
    out_names = Symbol[]
    out_sources = Tuple{Symbol, Int, Symbol}[]

    for name in left_names
        push!(out_names, name)
        push!(out_sources, (name, 1, name))
    end

    left_key_set = Set(left_keys)
    for name in right_names
        if name in right_key_set
            continue
        end
        out_name = name
        if name in left_names
            out_name = Symbol(string(name) * suffix[2])
            # Only rename left column if it's NOT a key column
            if !(name in left_key_set)
                idx = findfirst(==(name), out_names)
                if idx !== nothing
                    renamed_left = Symbol(string(name) * suffix[1])
                    out_names[idx] = renamed_left
                    out_sources[idx] = (renamed_left, 1, name)
                end
            end
        end
        push!(out_names, out_name)
        push!(out_sources, (out_name, 2, name))
    end

    # Check for duplicate output names
    _validate_join_output_names(out_names)

    # Build result
    result_cols = Dict{Symbol, Vector{Any}}()
    for name in out_names
        result_cols[name] = Any[]
    end

    if is_cross_join
        # Cross join: every left row pairs with every right row
        for i in 1:n_left
            for j in 1:n_right
                for (out_name, table, orig_name) in out_sources
                    if table == 1
                        push!(result_cols[out_name], ct_left[orig_name][i])
                    else
                        push!(result_cols[out_name], ct_right[orig_name][j])
                    end
                end
            end
        end
    else
        right_index = _build_right_index(ct_right, right_keys)

        # Track which right rows have been matched
        matched_right = Set{Int}()

        # First pass: all left rows
        for i in 1:n_left
            left_key = _build_key(ct_left, left_keys, i)
            if haskey(right_index, left_key)
                for j in right_index[left_key]
                    push!(matched_right, j)
                    for (out_name, table, orig_name) in out_sources
                        if table == 1
                            push!(result_cols[out_name], ct_left[orig_name][i])
                        else
                            push!(result_cols[out_name], ct_right[orig_name][j])
                        end
                    end
                end
            else
                # No match - add left row with missing for right columns
                for (out_name, table, orig_name) in out_sources
                    if table == 1
                        push!(result_cols[out_name], ct_left[orig_name][i])
                    else
                        push!(result_cols[out_name], missing)
                    end
                end
            end
        end

        # Second pass: unmatched right rows
        left_key_set = Set(left_keys)
        for j in 1:n_right
            if j in matched_right
                continue
            end
            for (out_name, table, orig_name) in out_sources
                if table == 1
                    # For key columns from left, use right's key value
                    if orig_name in left_key_set
                        key_idx = findfirst(==(orig_name), left_keys)
                        push!(result_cols[out_name], ct_right[right_keys[key_idx]][j])
                    else
                        push!(result_cols[out_name], missing)
                    end
                else
                    push!(result_cols[out_name], ct_right[orig_name][j])
                end
            end
        end
    end

    columns_out = [_finalize_column(result_cols[name]) for name in out_names]
    return _assemble(collect(out_names), columns_out)
end

"""
    semi_join(left, right; by=nothing)

Return rows from `left` that have matching keys in `right`.

Unlike `inner_join`, this does not add columns from `right` - it only filters `left`.

# Arguments
- `left`: Left table (Tables.jl-compatible)
- `right`: Right table (Tables.jl-compatible)
- `by`: Join specification (see `inner_join` for details)

# Returns
A `NamedTuple` containing rows from `left` where the key exists in `right`.
Only columns from `left` are returned.

# Examples
```julia
using Durbyn.TableOps

left = (id = [1, 2, 3, 4], x = [10, 20, 30, 40])
right = (id = [2, 4], y = [200, 400])

semi_join(left, right, by=:id)
# Output: (id = [2, 4], x = [20, 40])
```

See also: [`anti_join`](@ref), [`inner_join`](@ref)
"""
function semi_join(left, right; by=nothing)
    ct_left = _to_columns(left)
    ct_right = _to_columns(right)

    left_keys, right_keys = _resolve_join_keys(ct_left, ct_right, by)

    # Build set of keys present in right
    n_right = _nrows(ct_right)
    right_key_set = Set{Any}()
    for j in 1:n_right
        key = _build_key(ct_right, right_keys, j)
        push!(right_key_set, key)
    end

    # Filter left rows
    n_left = _nrows(ct_left)
    keep_indices = Int[]
    for i in 1:n_left
        left_key = _build_key(ct_left, left_keys, i)
        if left_key in right_key_set
            push!(keep_indices, i)
        end
    end

    return _subset_indices(ct_left, keep_indices)
end

"""
    anti_join(left, right; by=nothing)

Return rows from `left` that do NOT have matching keys in `right`.

This is the complement of `semi_join`.

# Arguments
- `left`: Left table (Tables.jl-compatible)
- `right`: Right table (Tables.jl-compatible)
- `by`: Join specification (see `inner_join` for details)

# Returns
A `NamedTuple` containing rows from `left` where the key does NOT exist in `right`.
Only columns from `left` are returned.

# Examples
```julia
using Durbyn.TableOps

left = (id = [1, 2, 3, 4], x = [10, 20, 30, 40])
right = (id = [2, 4], y = [200, 400])

anti_join(left, right, by=:id)
# Output: (id = [1, 3], x = [10, 30])
```

See also: [`semi_join`](@ref), [`left_join`](@ref)
"""
function anti_join(left, right; by=nothing)
    ct_left = _to_columns(left)
    ct_right = _to_columns(right)

    left_keys, right_keys = _resolve_join_keys(ct_left, ct_right, by)

    # Build set of keys present in right
    n_right = _nrows(ct_right)
    right_key_set = Set{Any}()
    for j in 1:n_right
        key = _build_key(ct_right, right_keys, j)
        push!(right_key_set, key)
    end

    # Filter left rows - keep those NOT in right
    n_left = _nrows(ct_left)
    keep_indices = Int[]
    for i in 1:n_left
        left_key = _build_key(ct_left, left_keys, i)
        if !(left_key in right_key_set)
            push!(keep_indices, i)
        end
    end

    return _subset_indices(ct_left, keep_indices)
end

# =============================================================================
# PanelData dispatches - apply operations by group
# =============================================================================

"""
    _apply_by_group(panel::PanelData, op::Function)

Internal helper to apply an operation to each group in a PanelData object.
Returns a new NamedTuple with results from all groups combined.

Note: PanelData stores data sorted by groups and date, so operations maintain
grouped order (not original input order). This is intentional for time series
operations that require data to be organized by group and time.
"""
function _apply_by_group(panel::PanelData, op::Function)
    ct = _to_columns(panel.data)
    groups = panel.groups

    if isempty(groups)
        # No grouping - apply to entire dataset
        return op(ct)
    end

    gt = groupby(ct, groups...)
    results = NamedTuple[]

    for idxs in gt.indices
        subgroup = _subset_indices(ct, idxs)
        result = op(subgroup)
        push!(results, result)
    end

    # Combine all results
    if isempty(results)
        # Apply op to empty data to get the correct output schema
        # (e.g., mutate should add new columns even on empty panels)
        empty_ct = _subset_indices(ct, Int[])
        return op(empty_ct)
    end

    return bind_rows(results...)
end

"""
    _rebuild_panel(panel::PanelData, new_data; groups=panel.groups, date=panel.date)

Create a new PanelData with new data but preserving metadata from the original panel.
"""
function _rebuild_panel(panel::PanelData, new_data;
                        groups::Vector{Symbol}=panel.groups,
                        date::Union{Symbol, Nothing}=panel.date)
    return PanelData(new_data, groups, date, panel.m,
                     panel.frequency, panel.target,
                     panel.time_fill_meta, panel.target_meta, panel.xreg_meta)
end

"""
    _apply_by_group_to_panel(panel::PanelData, op::Function)

Apply an operation by group and return a new PanelData with same metadata.
"""
function _apply_by_group_to_panel(panel::PanelData, op::Function)
    result_data = _apply_by_group(panel, op)
    return _rebuild_panel(panel, result_data)
end

"""
    query(panel::PanelData, predicate::Function)

Filter rows in a PanelData by group, applying the predicate within each group.

# Arguments
- `panel`: A PanelData object with grouping columns
- `predicate`: A function that takes a row (NamedTuple) and returns `Bool`

# Returns
A new `PanelData` with filtered rows, preserving grouping metadata.

# Examples
```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

data = (series = ["A", "A", "A", "B", "B", "B"],
        date = [1, 2, 3, 1, 2, 3],
        value = [10, 20, 30, 100, 200, 300])

panel = PanelData(data; groupby=:series, date=:date)

# Filter rows where value > 15 (applied per group)
filtered = query(panel, row -> row.value > 15)
```
"""
function query(panel::PanelData, predicate::Function)
    _apply_by_group_to_panel(panel, ct -> query(ct, predicate))
end

"""
    mutate(panel::PanelData; kwargs...)

Add or modify columns in a PanelData, applying transformations within each group.

When using functions that reference group data, the function receives only
the current group's data, enabling group-relative computations.

# Arguments
- `panel`: A PanelData object with grouping columns
- `kwargs...`: Column specifications (same as regular `mutate`)

# Returns
A new `PanelData` with transformed columns, preserving grouping metadata.

# Examples
```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs
using Statistics

data = (series = ["A", "A", "A", "B", "B", "B"],
        date = [1, 2, 3, 1, 2, 3],
        value = [10, 20, 30, 100, 200, 300])

panel = PanelData(data; groupby=:series, date=:date)

# Add group-relative columns
result = mutate(panel,
    group_mean = d -> fill(mean(d.value), length(d.value)),
    centered = d -> d.value .- mean(d.value))
```
"""
function mutate(panel::PanelData; kwargs...)
    _apply_by_group_to_panel(panel, ct -> mutate(ct; kwargs...))
end

"""
    arrange(panel::PanelData, cols...; rev::Bool=false)

Sort rows within each group of a PanelData.

# Arguments
- `panel`: A PanelData object with grouping columns
- `cols...`: Sort specifications (same as regular `arrange`)
- `rev`: Reverse sort order (default: false)

# Returns
A new `PanelData` with rows sorted within each group, preserving grouping metadata.

# Examples
```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

data = (series = ["A", "A", "A", "B", "B", "B"],
        date = [3, 1, 2, 2, 3, 1],
        value = [30, 10, 20, 200, 300, 100])

panel = PanelData(data; groupby=:series, date=:date)

# Sort by date within each series
sorted = arrange(panel, :date)
```
"""
function arrange(panel::PanelData, cols...; rev::Bool=false)
    _apply_by_group_to_panel(panel, ct -> arrange(ct, cols...; rev=rev))
end

"""
    select(panel::PanelData, specs...)

Select columns from a PanelData. Grouping columns are always preserved.

# Arguments
- `panel`: A PanelData object with grouping columns
- `specs...`: Column specifications (same as regular `select`)

# Returns
A new `PanelData` with selected columns, preserving grouping metadata.
Grouping columns are automatically included even if not specified.

# Examples
```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

data = (series = ["A", "A", "B", "B"],
        date = [1, 2, 1, 2],
        value = [10, 20, 100, 200],
        extra = [1, 2, 3, 4])

panel = PanelData(data; groupby=:series, date=:date)

# Select value column (series is automatically kept as grouping column)
result = select(panel, :value)
```
"""
function select(panel::PanelData, spec1::Union{Symbol, Pair, ColumnSelector}, specs::Union{Symbol, Pair, ColumnSelector}...)
    ct = _to_columns(panel.data)

    # Ensure grouping columns are included
    all_specs = [spec1, specs...]
    group_set = Set(panel.groups)

    # Check for renaming specs on structural columns (groups and date)
    # tsibble philosophy: structural columns require explicit rename()
    structural_cols = copy(group_set)
    if panel.date !== nothing
        push!(structural_cols, panel.date)
    end

    for spec in all_specs
        if spec isa Pair
            # Pair can be :new => :old or :new => transform
            # For renaming, it's :new => :old where old is a Symbol
            old_col = last(spec)
            new_col = first(spec)
            if old_col isa Symbol && new_col isa Symbol && old_col != new_col
                if old_col in structural_cols
                    col_type = old_col in group_set ? "grouping" : "date"
                    throw(ArgumentError(
                        "Cannot rename $col_type column :$old_col via select. " *
                        "Use rename(panel, :$new_col => :$old_col) to rename structural columns."))
                end
            end
        end
    end

    # Add grouping columns first if not already in specs
    final_specs = Any[]
    for g in panel.groups
        push!(final_specs, g)
    end

    for spec in all_specs
        col_name = spec isa Pair ? Symbol(last(spec)) : (spec isa ColumnSelector ? nothing : Symbol(spec))
        if col_name === nothing || !(col_name in group_set)
            push!(final_specs, spec)
        end
    end

    result = select(ct, final_specs...)

    # Check if date column is in the output; if not, set date to nothing
    result_names = Set(_column_names(result))
    new_date = (panel.date !== nothing && panel.date in result_names) ? panel.date : nothing

    return _rebuild_panel(panel, result; date=new_date)
end

# Zero-argument select for PanelData - returns just grouping columns
function select(panel::PanelData)
    if isempty(panel.groups)
        throw(ArgumentError(
            "Cannot use zero-argument select on ungrouped PanelData. " *
            "Use select(panel, cols...) to specify columns explicitly."))
    end
    ct = _to_columns(panel.data)
    result = select(ct, panel.groups...)
    # Date column is not included, so set to nothing
    return _rebuild_panel(panel, result; date=nothing)
end

"""
    summarise(panel::PanelData; kwargs...)
    summarize(panel::PanelData; kwargs...)

Compute summary statistics for each group in a PanelData.

This is equivalent to `groupby(data, groups...) |> gt -> summarise(gt, ...)`,
but preserves the PanelData structure.

# Arguments
- `panel`: A PanelData object with grouping columns
- `kwargs...`: Summary specifications (same as regular `summarise`)

# Returns
A `NamedTuple` (not PanelData) with one row per group and summary columns.

# Examples
```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs
using Statistics

data = (series = ["A", "A", "A", "B", "B", "B"],
        date = [1, 2, 3, 1, 2, 3],
        value = [10, 20, 30, 100, 200, 300])

panel = PanelData(data; groupby=:series, date=:date)

# Summarise per group
result = summarise(panel, mean_val = :value => mean, n = d -> length(d.value))
# Output: (series = ["A", "B"], mean_val = [20.0, 200.0], n = [3, 3])
```
"""
function summarise(panel::PanelData; kwargs...)
    ct = _to_columns(panel.data)
    groups = panel.groups

    if isempty(groups)
        throw(ArgumentError("Cannot summarise PanelData without grouping columns"))
    end

    gt = groupby(ct, groups...)
    return summarise(gt; kwargs...)
end

summarize(panel::PanelData; kwargs...) = summarise(panel; kwargs...)

"""
    summarise(panel::PanelData, ac::Across)

Apply `across` summary to each group in a PanelData.

# Examples
```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs
using Statistics

data = (series = ["A", "A", "B", "B"],
        x = [1.0, 2.0, 3.0, 4.0],
        y = [10.0, 20.0, 30.0, 40.0])

panel = PanelData(data; groupby=:series)

summarise(panel, across([:x, :y], :mean => mean))
# Output: (series = ["A", "B"], x_mean = [1.5, 3.5], y_mean = [15.0, 35.0])
```
"""
function summarise(panel::PanelData, ac::Across)
    ct = _to_columns(panel.data)
    groups = panel.groups

    if isempty(groups)
        throw(ArgumentError("Cannot summarise PanelData without grouping columns"))
    end

    gt = groupby(ct, groups...)
    return summarise(gt, ac)
end

"""
    distinct(panel::PanelData, cols...; keep_all::Bool=false)

Remove duplicate rows within each group of a PanelData.

# Arguments
- `panel`: A PanelData object with grouping columns
- `cols...`: Columns to consider for uniqueness
- `keep_all`: Keep all columns (default: false)

# Returns
A new `PanelData` with duplicates removed within each group.
"""
function distinct(panel::PanelData, cols::Symbol...; keep_all::Bool=false)
    _apply_by_group_to_panel(panel, ct -> distinct(ct, cols...; keep_all=keep_all))
end

"""
    fill_missing(panel::PanelData, cols::Symbol...; direction::Symbol=:down)

Fill missing values within each group of a PanelData.

This is especially useful for time series panel data where you want to
forward-fill or backward-fill missing values within each series independently.

# Arguments
- `panel`: A PanelData object with grouping columns
- `cols...`: Columns to fill
- `direction`: Fill direction (`:down`, `:up`, `:downup`, `:updown`)

# Returns
A new `PanelData` with missing values filled within each group.

# Examples
```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

data = (series = ["A", "A", "A", "B", "B", "B"],
        date = [1, 2, 3, 1, 2, 3],
        value = [10, missing, 30, missing, 200, missing])

panel = PanelData(data; groupby=:series, date=:date)

# Forward fill within each series
filled = fill_missing(panel, :value; direction=:down)
# Series A: [10, 10, 30], Series B: [missing, 200, 200]
```
"""
function fill_missing(panel::PanelData, cols::Symbol...; direction::Symbol=:down)
    _apply_by_group_to_panel(panel, ct -> fill_missing(ct, cols...; direction=direction))
end

"""
    rename(panel::PanelData, specs::Pair{Symbol,Symbol}...)

Rename columns in a PanelData. Updates grouping column references if renamed.

# Arguments
- `panel`: A PanelData object
- `specs...`: Rename specifications as `Pair`s: `:new_name => :old_name`

# Returns
A new `PanelData` with renamed columns and updated metadata.
"""
function rename(panel::PanelData, specs::Pair{Symbol,Symbol}...)
    ct = _to_columns(panel.data)
    result = rename(ct, specs...)

    # Update group column names if they were renamed
    rename_map = Dict(last(p) => first(p) for p in specs)
    new_groups = Symbol[get(rename_map, g, g) for g in panel.groups]
    new_date = panel.date !== nothing ? get(rename_map, panel.date, panel.date) : nothing

    return _rebuild_panel(panel, result; groups=new_groups, date=new_date)
end

"""
    pivot_longer(panel::PanelData; id_cols=Symbol[], value_cols=Symbol[],
                 names_to=:variable, values_to=:value)

Pivot a PanelData from wide to long format, preserving panel metadata.

Grouping columns are automatically added to `id_cols`.

# Returns
A new `PanelData` in long format with updated metadata.
"""
function pivot_longer(panel::PanelData;
                      id_cols::Union{Symbol, AbstractVector{Symbol}} = Symbol[],
                      value_cols::Union{Symbol, AbstractVector{Symbol}} = Symbol[],
                      names_to::Symbol = _DEFAULT_NAMES_TO,
                      values_to::Symbol = _DEFAULT_VALUES_TO)
    ct = _to_columns(panel.data)

    # Ensure grouping columns are in id_cols, preserving group order
    ids = id_cols isa Symbol ? Symbol[id_cols] : collect(id_cols)
    # Prepend groups not already in ids, maintaining original group order
    groups_to_add = [g for g in panel.groups if !(g in ids)]
    ids = vcat(groups_to_add, ids)

    result = pivot_longer(ct; id_cols=ids, value_cols=value_cols,
                          names_to=names_to, values_to=values_to)

    return _rebuild_panel(panel, result)
end

"""
    pivot_wider(panel::PanelData; names_from::Symbol, values_from::Symbol,
                id_cols=Symbol[], fill=missing, sort_names::Bool=false)

Pivot a PanelData from long to wide format, preserving panel metadata.

Grouping columns are automatically added to `id_cols`.

# Returns
A new `PanelData` in wide format with updated metadata.
"""
function pivot_wider(panel::PanelData;
                     names_from::Symbol,
                     values_from::Symbol,
                     id_cols::Union{Symbol, AbstractVector{Symbol}} = Symbol[],
                     fill = missing,
                     sort_names::Bool = false)
    ct = _to_columns(panel.data)

    # Ensure grouping columns are in id_cols, preserving group order
    ids = id_cols isa Symbol ? Symbol[id_cols] : collect(id_cols)
    # Prepend groups not already in ids, maintaining original group order
    groups_to_add = [g for g in panel.groups if !(g in ids)]
    ids = vcat(groups_to_add, ids)

    result = pivot_wider(ct; names_from=names_from, values_from=values_from,
                         id_cols=ids, fill=fill, sort_names=sort_names)

    return _rebuild_panel(panel, result)
end

# Note: glimpse(panel::PanelData) is defined in glimpse_extensions.jl to avoid circular dependencies
