"""
    Formula(lhs::Union{Symbol,Nothing}, rhs::Vector{Vector{Symbol}}, intercept::Bool)
    Formula(str::AbstractString)

Lightweight, parsed representation of a regression formula, a small Domain-Specific Language (DSL).

You normally construct it from a string with `Formula("y ~ x1 * x2")` or
`parse_formula("~ x1 + x2")`. The `Formula` stores only structure (symbols and
intercept flag); it does **not** know about any particular dataset.

# Fields
- `lhs` — Response variable as a `Symbol`, or `nothing` when absent (e.g. `"~ x1 + x2"`).
- `rhs` — Expanded RHS terms. Each inner vector represents a main effect
  (`[:x1]`) or an interaction (`[:a, :b]` for `a:b`).
- `intercept` — `true` if an intercept column should be included.

# Supported syntax (subset of R)
- `lhs ~ rhs` or `~ rhs` (no response).
- Intercept on by default; disable with `-1` or `+0` anywhere on the RHS.
- `+` adds terms.
- `:` is an interaction as written (e.g. `a:b`).
- `*` expands to all non-empty interactions of the factors (`a*b*c` ⇒ `a`,
  `b`, `c`, `a:b`, `a:c`, `b:c`, `a:b:c`).

!!! note
    This is intentionally minimal: no general subtraction of terms (other than
    `-1`), no `I()`, `^` degrees, offsets, or factor/contrast handling.

# Examples
```julia
julia> f = Formula("y ~ x1 * x2")
Formula(Symbol("y"), Vector{Vector{Symbol}}[[Symbol("x1")], [Symbol("x2")], [Symbol("x1"), Symbol("x2")]], true)

julia> Formula("~ x1 + x2 - 1").intercept
false
````

"""
struct Formula
    lhs::Union{Symbol,Nothing}
    rhs::Vector{Vector{Symbol}}
    intercept::Bool
end


struct CompiledFormula
    intercept::Bool
    yidx::Union{Int,Nothing}
    rhs_idxs::Vector{Vector{Int}}
    labels::Vector{String}
end

_formula_all_predictors(colnames::Vector{String}; intercept::Bool=true)::Formula =
    Formula(nothing, [[Symbol(c)] for c in colnames], intercept)


pivoted_qr(X::StridedMatrix) =
    @static isdefined(LinearAlgebra, :ColumnNorm) ? qr(X, ColumnNorm()) : qr(X, Val(true))


function _getcol(nm::NamedMatrix{T}, name::AbstractString) where {T}
    idx = findfirst(==(String(name)), nm.colnames)
    isnothing(idx) && error("Column '$name' not found in NamedMatrix.")
    return nm.data[:, idx]
end


function _interaction(cols::AbstractVector{<:AbstractVector{T}}) where {T}
    n = length(cols[1])
    out = copy(cols[1])
    @inbounds for j in 2:length(cols)
        out .= out .* cols[j]
    end
    return out
end


"""
    parse_formula(str::AbstractString) -> Formula

Parse a formula string into a [`Formula`](@ref).

# Grammar (subset)
- `lhs ~ rhs` or `~ rhs`
- Intercept is **on** by default; disable with `-1` or `+0`
- `+` to add terms
- `:` for interactions exactly as written
- `*` expands to all non-empty interactions of the factors

# Errors
- Any `-` other than `-1` is rejected.
- Pure parsing; it does not validate that variables exist in your data.

# Examples
```julia
julia> parse_formula("y ~ x1 + x2")
Formula(:y, Vector{Vector{Symbol}}[[ :x1 ], [ :x2 ]], true)

julia> parse_formula("~ a*b - 1")
Formula(nothing, Vector{Vector{Symbol}}[[ :a ], [ :b ], [ :a, :b ]], false)
````

"""
function parse_formula(formula::AbstractString)::Formula
    s = replace(String(formula), r"\s+" => "")
    lhs, rhs = if occursin("~", s)
        parts = split(s, "~"; limit=2)
        (isempty(parts[1]) ? nothing : Symbol(parts[1]), parts[2])
    else
        (nothing, s)
    end

    rhs = replace(rhs, r"-\s*1" => "+0")
    occursin('-', rhs) && error("Only '-1' is supported (to remove intercept). Found other '-' in RHS: $rhs")

    tokens = [t for t in split(rhs, '+') if !isempty(t)]
    intercept = true
    raw_terms = String[]
    for t in tokens
        if t == "0"
            intercept = false
        elseif t == "1"
        else
            push!(raw_terms, t)
        end
    end

    _subsets(v::AbstractVector{<:AbstractString}) = begin
        vs = Symbol.(v)
        n = length(vs)
        res = Vector{Vector{Symbol}}()
        for mask in 1:(2^n-1)
            sub = Symbol[]
            for i in 1:n
                if ((mask >> (i - 1)) & 1) == 1
                    push!(sub, vs[i])
                end
            end
            push!(res, sub)
        end
        res
    end

    terms = Vector{Vector{Symbol}}()
    seen = Set{String}()
    for term in raw_terms
        expanded = if occursin('*', term)
            _subsets(split(term, '*'))
        elseif occursin(':', term)
            [Symbol.(split(term, ':'))]
        else
            [[Symbol(term)]]
        end
        for trm in expanded
            key = join(string.(sort(trm, by=string)), ":")
            if !(key in seen)
                push!(terms, trm)
                push!(seen, key)
            end
        end
    end

    return Formula(lhs, terms, intercept)
end

Formula(s::AbstractString) = parse_formula(s)

"""
    compile(f::Formula, colnames::Vector{String}) -> CompiledFormula

Bind a [`Formula`](@ref) to a specific column schema for fast re-use.

Resolves each symbol in `f` to a concrete column index from `colnames`, and
returns a `CompiledFormula` that can be applied repeatedly to any
`NamedMatrix` that uses the same column layout.

# Arguments
- `f` — Parsed formula.
- `colnames` — Column names of your design source (e.g. `nm.colnames`).

# Returns
- `CompiledFormula` with:
  - `yidx::Union{Int,Nothing}` — column index of the response, if present
  - `rhs_idxs::Vector{Vector{Int}}` — indices for each RHS term
  - `intercept::Bool` and printable `labels::Vector{String}` for RHS terms

# Errors
- `ArgumentError` if any term (including the response) is not found in
  `colnames`.

# Why compile?
- Avoids repeated name lookups and parsing when building many design matrices
  (e.g., iterative fitting / cross-validation).

# Examples
```julia
julia> f  = Formula("y ~ x1 * x2");
julia> cf = compile(f, ["y","x1","x2"]);

julia> cf.yidx

julia> cf.rhs_idxs
````
"""
function compile(f::Formula, colnames::Vector{String})::CompiledFormula
    ix = Dict{Symbol,Int}(Symbol(c) => i for (i, c) in pairs(colnames))

    yidx = isnothing(f.lhs) ? nothing :
           isnothing(get(ix, f.lhs::Symbol, nothing)) ?
           throw(ArgumentError("Response $(f.lhs) not found in data columns")) :
           ix[f.lhs::Symbol]

    rhs_idxs = Vector{Vector{Int}}()
    labels = String[]
    for trm in f.rhs
        idxs = Vector{Int}(undef, length(trm))
        for (k, s) in enumerate(trm)
            haskey(ix, s) || throw(ArgumentError("Term $(s) not found in data columns"))
            idxs[k] = ix[s]
        end
        push!(rhs_idxs, idxs)
        push!(labels, join(string.(trm), ":"))
    end

    return CompiledFormula(f.intercept, yidx, rhs_idxs, labels)
end

"""
    model_matrix(f::Formula, nm::NamedMatrix{T};
                 dropcollinear::Bool=false, intercept_name::String="Intercept") -> NamedMatrix{T}
    model_matrix(cf::CompiledFormula, nm::NamedMatrix{T};
                 dropcollinear::Bool=false, intercept_name::String="Intercept") -> NamedMatrix{T}
    model_matrix(spec::AbstractString, nm::NamedMatrix{T};
                 dropcollinear::Bool=false, intercept_name::String="Intercept") -> NamedMatrix{T}
    model_matrix(y::AbstractVector, X::NamedMatrix{T};
                 intercept::Bool=true, dropcollinear::Bool=false, intercept_name::String="Intercept")
        -> NamedMatrix{T}
where {T<:Number}

Build the RHS design matrix (like R's `model.matrix`) from a formula or from
an existing predictor matrix `X`. The result is a `NamedMatrix{T}` whose
`colnames` are the term labels (e.g. `"Intercept"`, `"x1"`, `"x1:x2"`).

# Behavior
- Intercept is included unless suppressed in the formula (`-1` or `+0`).
- `:` creates elementwise interaction columns.
- `*` expands into main effects and all lower-order interactions.
- Column order follows the expanded term order plus (optional) intercept first.
- `dropcollinear=true` removes linearly dependent columns using a pivoted QR
  while preserving the original order of the retained columns.
- **`y, X` overload:** uses *all* columns in `X` as predictors.
  - `intercept=true`  ⇒ like `y ~ x1 + x2 + …`
  - `intercept=false` ⇒ like `y ~ x1 + x2 + … - 1`
  - `y` is only length-checked here; it is not returned (use `model_frame` for that).

# Keyword arguments
- `dropcollinear` — remove linearly dependent columns (default `false`).
- `intercept_name` — label for the intercept column (default `"Intercept"`).

# Errors
- Missing variables referenced by the formula cause an error.
- For `y, X`, `length(y)` must equal `nrows(X)`.

# Examples
```julia
julia> Xdata = [1.0 2.0 3.0 4.0;
                2.0 5.0 1.0 0.5;
                3.0 7.0 6.0 1.5;
                4.0 9.0 2.0 2.0];

julia> nm = NamedMatrix{Float64}(Xdata, ["r1","r2","r3","r4"], ["y","x1","x2","x3"]);

julia> f  = Formula("y ~ x1 * x2");
julia> cf = compile(f, nm.colnames);

julia> X1 = model_matrix(f,  nm);
julia> X2 = model_matrix(cf, nm);
julia> X3 = model_matrix("y ~ x1 * x2", nm);

# y, X overload (all predictors; optional intercept)
julia> y  = nm.data[:, 1];
julia> X4 = model_matrix(y, NamedMatrix{Float64}(nm.data[:, 2:end], nm.rownames, nm.colnames[2:end]));
julia> X5 = model_matrix(y, NamedMatrix{Float64}(nm.data[:, 2:end], nm.rownames, nm.colnames[2:end]);
                         intercept=false);
````

"""
function model_matrix(cf::CompiledFormula, nm::NamedMatrix{T};
    dropcollinear::Bool=false, intercept_name::String="Intercept") where {T<:Number}
    n = size(nm.data, 1)

    cols = Vector{Vector{T}}()
    names = String[]

    if cf.intercept
        push!(cols, ones(T, n))
        push!(names, intercept_name)
    end

    for (idxs, lab) in zip(cf.rhs_idxs, cf.labels)
        if length(idxs) == 1
            v = nm.data[:, idxs[1]]
            push!(cols, Vector{T}(v))
        else
            vs = [Vector{T}(nm.data[:, j]) for j in idxs]
            push!(cols, _interaction(vs))
        end
        push!(names, lab)
    end

    m = length(cols)
    X = m == 0 ? Array{T}(undef, n, 0) : Array{T}(undef, n, m)
    for j in 1:m
        X[:, j] = cols[j]
    end

    if dropcollinear && m > 0
        F = pivoted_qr(X)
        r = rank(X)
        keep = sort(F.p[1:r])
        X = X[:, keep]
        names = names[keep]
    end

    return NamedMatrix{T}(X, nm.rownames, names)
end


function model_matrix(f::Formula, nm::NamedMatrix{T}; kwargs...) where {T<:Number}
    cf = compile(f, nm.colnames)
    return model_matrix(cf, nm; kwargs...)
end
model_matrix(s::AbstractString, nm::NamedMatrix{T}; kwargs...) where {T<:Number} =
    model_matrix(parse_formula(s), nm; kwargs...)

function model_matrix(y::AbstractVector, X::NamedMatrix{T};
                      intercept::Bool=true, kwargs...) where {T<:Number}
    length(y) == size(X.data, 1) ||
        throw(ArgumentError("length(y) ($(length(y))) must equal nrows(X) ($(size(X.data,1)))."))
    f = _formula_all_predictors(X.colnames; intercept=intercept)
    return model_matrix(f, X; kwargs...)   # call the Formula overload
end

"""
    model_frame(f::Formula, nm::NamedMatrix{T};
                dropcollinear::Bool=false, intercept_name::String="Intercept")
        -> (y = Union{Vector{T},Nothing}, X = NamedMatrix{T})
    model_frame(cf::CompiledFormula, nm::NamedMatrix{T};
                dropcollinear::Bool=false, intercept_name::String="Intercept")
        -> (y, X)
    model_frame(spec::AbstractString, nm::NamedMatrix{T};
                dropcollinear::Bool=false, intercept_name::String="Intercept")
        -> (y, X)
    model_frame(y::AbstractVector, X::NamedMatrix{T};
                intercept::Bool=true, dropcollinear::Bool=false, intercept_name::String="Intercept")
        -> (y = Vector{T}, X = NamedMatrix{T})
where {T<:Number}

Build a response/design pair (like R’s `model.frame`). Returns a `NamedTuple`:

- `y` — the response vector if the formula has a LHS; otherwise `nothing`
- `X` — the RHS design as a `NamedMatrix{T}`

# Notes
- Same formula syntax/behavior as [`model_matrix`](@ref) apply to the RHS.
- Row names from `nm` propagate to `X`.
- **`y, X` overload:** uses *all* columns in `X` as predictors.
  - `intercept=true`  ⇒ like `y ~ x1 + x2 + …`
  - `intercept=false` ⇒ like `y ~ x1 + x2 + … - 1`
  - Returns `y` converted to `Vector{T}` where `T == eltype(X.data)`.

# Examples
```julia
julia> mf1 = model_frame(f, nm);
julia> mf2 = model_frame(cf, nm);
julia> mf3 = model_frame("y ~ x1 * x2", nm);

# y, X overload
julia> y  = nm.data[:, 1];
julia> Xs = NamedMatrix{Float64}(nm.data[:, 2:end], nm.rownames, nm.colnames[2:end]);

julia> mf4 = model_frame(y, Xs);                  # with intercept
julia> mf5 = model_frame(y, Xs; intercept=false); # no intercept
````
"""
function model_frame(f::Formula, nm::NamedMatrix{T}; kwargs...) where {T<:Number}
    cf = compile(f, nm.colnames)
    y = isnothing(cf.yidx) ? nothing : Vector{T}(nm.data[:, cf.yidx])
    X = model_matrix(cf, nm; kwargs...)
    return (y=y, X=X)
end

model_frame(s::AbstractString, nm::NamedMatrix{T}; kwargs...) where {T<:Number} =
    model_frame(parse_formula(s), nm; kwargs...)


model_frame(cf::CompiledFormula, nm::NamedMatrix{T}; kwargs...) where {T<:Number} =
    (y = isnothing(cf.yidx) ? nothing : Vector{T}(nm.data[:, cf.yidx]);
    (y=y, X=model_matrix(cf, nm; kwargs...)))

function model_frame(y::AbstractVector, X::NamedMatrix{T};
                     intercept::Bool=true, kwargs...) where {T<:Number}
    length(y) == size(X.data, 1) ||
        throw(ArgumentError("length(y) ($(length(y))) must equal nrows(X) ($(size(X.data,1)))."))
    yT = Vector{T}(y)
    f  = _formula_all_predictors(X.colnames; intercept=intercept)
    Xmm = model_matrix(f, X; kwargs...)    # avoid recursion; use Formula overload
    return (y = yT, X = Xmm)
end