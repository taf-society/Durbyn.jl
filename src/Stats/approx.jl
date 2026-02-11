function _apply_ties(ys, ties)
    ties isa Function || throw(ArgumentError("'ties' must be a function when used for collapsing"))
    return ties(ys)
end

function regularize_values(x, y; ties::Union{Function,Symbol}=mean, warn_collapsing::Bool=false, na_rm::Bool=true)
    x = collect(x)
    y = collect(y)
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length"))

    keptNA = false
    na_x = isna.(x)
    na_y = isna.(y)
    any_na = any(na_x .| na_y)

    notNA = nothing
    nx = Int64(0)
    if any_na
        ok = .!(na_x .| na_y)
        if na_rm
            x = x[ok]; y = y[ok]
            nx = length(x)
        else
            keptNA = true
            nx = sum(ok)
            notNA = ok
        end
    else
        nx = length(x)
    end

    if !(ties == :ordered || ties isa Function)
        throw(ArgumentError("'ties' must be :ordered or a function"))
    end

    ordered = (ties == :ordered)
    if !ordered && !issorted(x)
        p = sortperm(x)
        x = x[p]; y = y[p]
    elseif ordered && !issorted(x)
        throw(ArgumentError("ties=:ordered requires x to be nondecreasing"))
    end

    # collapse duplicates if not ordered
    if !ordered
        ux = unique(x)
        if length(ux) < nx
            if warn_collapsing
                @warn "collapsing to unique x values"
            end
            newx = eltype(x)[]
            newy = Float64[]
            i = 1
            while i <= length(x)
                xi = x[i]
                j = searchsortedlast(x, xi)
                push!(newx, xi)
                ys = @view y[i:j]
                push!(newy, _apply_ties(ys, ties))
                i = j + 1
            end
            x = newx
            y = newy
            if keptNA
                notNA = .!isna.(x)
            end
        end
    end

    return (x=x, y=y, keptNA=keptNA, notNA=notNA)
end

function _approxtest(x::Vector{Float64}, y::Vector{Float64}, method::Symbol, f::Float64; na_rm::Bool=true)
    (method === :linear || method === :constant) ||
        throw(ArgumentError("approx(): invalid interpolation method (use :linear or :constant)"))
    if method === :constant
        (!isfinite(f) || f < 0.0 || f > 1.0) && throw(ArgumentError("approx(): invalid f value"))
    end

    if na_rm
        @inbounds for i in eachindex(x,y)
            (isnan(x[i]) || isnan(y[i])) && error("approx(): attempted to interpolate missing values")
        end
    else
        @inbounds for i in eachindex(x)
            isnan(x[i]) && error("approx(x,y, .., na.rm=false): missing values in x are not allowed")
        end
    end

    length(x) == length(y) || throw(ArgumentError("approx(): length(x) and length(y) must match"))
    length(x) > 0 || error("approx(): zero non-NA points")
    if method === :linear
        length(x) > 1 || error("approx(): need at least two non-NA values to interpolate")
    end
    return nothing
end

function _approx1(v::Float64, x::Vector{Float64}, y::Vector{Float64}, method::Symbol,
                  yleft::Float64, yright::Float64, f::Float64)
    if v < x[1]
        return yleft
    elseif v > x[end]
        return yright
    end

    i = searchsortedlast(x, v)
    if x[i] == v
        return y[i]
    end
    j = min(i+1, length(x))

    if method === :linear
        dx = x[j] - x[i]
        return dx == 0.0 ? y[i] : (y[i] + (v - x[i]) * (y[j] - y[i]) / dx)
    else
        f1 = (f == 0.0) ? 0.0 : 1.0 - f
        f2 = (f == 1.0) ? 0.0 : f
        return f1*y[i] + f2*y[j]
    end
end

"""
    approx(
    x::AbstractVector,
    y::AbstractVector;
    xout::Union{AbstractVector,Nothing} = nothing,
    method::Symbol = :linear,
    n::Integer = 50,
    yleft::Union{Float64,Nothing} = nothing,
    yright::Union{Float64,Nothing} = nothing,
    rule::Tuple = (1, 1),
    f::Real = 0.0,
    ties::Union{Function, Symbol} = mean,
    na_rm::Bool = true,)

    approx(;
    x::AbstractVector,
    y::AbstractVector,
    xout::Union{AbstractVector,Nothing} = nothing,
    method::String = "linear",
    n::Integer = 50,
    yleft::Union{Float64,Nothing} = nothing,
    yright::Union{Float64,Nothing} = nothing,
    rule::Tuple = (1, 1),
    f::Real = 0.0,
    ties::Union{Function,String} = mean,
    na_rm::Bool = true,)

Interpolation Functions

Return a pair of vectors `(x = xout_vec, y = yout_vec)` which linearly or
constant-step interpolate the given data points.

# Arguments
- `x, y`: Numeric vectors giving the coordinates of the points to be interpolated.
  Both must have the same length.
- `xout`: Optional numeric values specifying where interpolation is to take place.
  If omitted, interpolation occurs at `n` equally spaced points spanning
  `[minimum(x), maximum(x)]`.
- `method`: Interpolation method. Use `:linear` or `:constant`.
- `n`: Number of points when `xout` is not specified.
- `yleft`: Value returned when query points are `< minimum(x)`. By default this is
  determined by `rule` (see below).
- `yright`: Value returned when query points are `> maximum(x)`. By default this is
  determined by `rule`.
- `rule`: Tuple of one or two integers describing extrapolation outside
  `[minimum(x), maximum(x)]`. If the (left/right) entry is `1`, the
  corresponding out-of-range values are `missing`. If it is `2`, the value at the
  closest data extreme is used. You can supply different left/right rules, e.g.
  `rule = (2, 1)`.
- `f`: For `method = :constant`, a number in `[0, 1]` indicating a compromise
  between left- and right-continuous step functions. If `y₀` and `y₁` are the
  values to the left and right of the point, the result is
  `y₀` if `f == 0`, `y₁` if `f == 1`, and `y₀*(1 - f) + y₁*f` for intermediate
  values. In this way the result is right-continuous for `f == 0` and
  left-continuous for `f == 1`, even for non-finite `y` values.
- `ties`: Handling of tied `x` values. Either `:ordered` or a function taking a
  vector of `y` values and returning a single number (e.g. `mean`, `min`, `max`).
  See *Details*.
- `na_rm`: Controls how `missing` values are handled. If `true` (default), any
  rows with `missing` in `x` or `y` are dropped. If `false`, `missing` in `x`
  is invalid and throws, while `missing` in `y` may propagate to outputs
  depending on `rule`.

# Details
- At least two complete `(x, y)` pairs are required for `method = :linear`
  (only one is required for `:constant`).
- If there are duplicated `x` values and `ties` is a function, it is applied to the
  `y` values for each distinct `x` to produce unique `x` with aggregated `y`.
  Useful choices include `mean`, `min`, and `max`.
- If `ties == :ordered`, `x` is assumed to be already sorted (and may contain
  duplicates, which are kept); no collapsing is done. This is fastest for large inputs.
- Supplying `ties = (:ordered, f)` in R slightly changes performance; in this
  Julia API, use `ties = :ordered` when already sorted, otherwise pass an
  aggregator function (e.g. `ties = mean`).
- If unspecified, `yleft` defaults to `missing` if `rule[1] == 1`, else `y[1]`;
  `yright` defaults to `missing` if `rule[2] == 1`, else `y[end]`.
- The first `y` value is used for extrapolation to the left (when `rule[1] == 2`)
  and the last for extrapolation to the right (when `rule[2] == 2`).

# Returns
A named tuple `(x = xout_vec, y = yout_vec)` containing the interpolation grid
and corresponding interpolated values, according to the chosen `method` and `rule`.

# References
Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988) The New S Language. Wadsworth & Brooks/Cole.

# Examples
```julia
x = 1:10
y = randn(10)

# Linear interpolation at default n=50 points
res = approx(x, y)

# Linear interpolation at specific points
xq = 0:0.5:11
res2 = approx(x, y; xout=xq)

# Constant (step) interpolation, left-continuous (f=1)
res_step = approx(x, y; xout=xq, method=:constant, f=1.0)

# Different extrapolation on left (constant) and right (NA/missing)
res_lr = approx(x, y; xout=xq, rule=(2, 1))

# Handling ties via aggregation
xt = [2, 2:4... , 4, 4, 5, 5, 7, 7, 7]
yt = [1:6... , 5, 4, 3:-1:1...]
res_mean = approx(xt, yt; xout=xt, ties=mean)       # collapse ties by mean
res_min  = approx(xt, yt; xout=xt, ties=min)        # collapse ties by min
res_ord  = approx(xt, yt; xout=xt, ties=:ordered)   # assume already ordered
```
"""
function approx(x::AbstractVector, y::AbstractVector;
                xout::Union{AbstractVector,Nothing}=nothing,
                method::Symbol=:linear,
                n::Integer=50,
                yleft::Union{Float64,Nothing}=nothing,
                yright::Union{Float64,Nothing}=nothing,
                rule::Union{Integer,Tuple{<:Integer,<:Integer}}=(1,1),
                f::Real=0.0,
                ties::Union{Function,Symbol}=mean,
                na_rm::Bool=true)

    rule_t = rule isa Integer ? (rule, rule) : rule
    length(rule_t) == 2 || throw(ArgumentError("rule must have 1 or 2 integers"))

    r = regularize_values(x, y; ties=ties, na_rm=na_rm)
    x = r.x; y = r.y

    noNA = na_rm || !r.keptNA
    nx = noNA ? length(x) : sum(r.notNA)
    isnan(float(nx)) && error("invalid length(x)")

    if isnothing(yleft)
        yleft = (rule_t[1] == 1) ? NaN : float(y[1])
    end
    if isnothing(yright)
        yright = (rule_t[2] == 1) ? NaN : float(y[end])
    end

    method = method === :linear ? :linear : method === :constant ? :constant :
        throw(ArgumentError("invalid interpolation method"))
    f = float(f)

    xv = collect(Float64[ismissing(v) ? NaN : Float64(v) for v in x])
    yv = collect(Float64[ismissing(v) ? NaN : Float64(v) for v in y])
    _approxtest(xv, yv, method, f; na_rm=na_rm)

    if isnothing(xout)
        n > 0 || error("approx requires n >= 1")
        if noNA
            xout = range(xv[1], xv[end], length=n)
        else
            xnn = xv[r.notNA]
            xout = range(xnn[1], xnn[end], length=n)
        end
    end

    xoutv = collect(float.(xout))
    yout = Vector{Float64}(undef, length(xoutv))
    @inbounds for i in eachindex(xoutv)
        yi = ismissing(xoutv[i]) ? NaN : _approx1(xoutv[i], xv, yv, method, yleft, yright, f)
        yout[i] = yi
    end
    return (x = xoutv, y = yout)
end

function approx(;
    x::AbstractVector,
    y::AbstractVector,
    xout::Union{AbstractVector,Nothing} = nothing,
    method::String = "linear",
    n::Integer = 50,
    yleft::Union{Float64,Nothing} = nothing,
    yright::Union{Float64,Nothing} = nothing,
    rule::Tuple = (1, 1),
    f::Real = 0.0,
    ties::Union{Function,String} = mean,
    na_rm::Bool = true,)
    method = Symbol(method)

    if ties isa String
        ties = Symbol(ties)
    end

    out = approx(
        x,
        y,
        xout = xout,
        method = method,
        n = n,
        yleft = yleft,
        yright = yright,
        rule = rule,
        f = f,
        ties = ties,
        na_rm = na_rm,
    )
    return out
end

"""

    approxfun(
    x::AbstractVector,
    y::AbstractVector;
    method::Symbol = :linear,
    yleft::Union{Float64, Nothing} = nothing,
    yright::Union{Float64, Nothing} = nothing,
    rule::Tuple = (1, 1),
    f::Real = 0.0,
    ties::Union{Function, Symbol} = mean,
    na_rm::Bool = true,)k

    approxfun(
    ;x::AbstractVector,
    y::AbstractVector,
    method::String = "linear",
    yleft::Union{Float64, Nothing} = nothing,
    yright::Union{Float64, Nothing} = nothing,
    rule::Tuple = (1, 1),
    f::Real = 0.0,
    ties::Union{Function, String} = mean,
    na_rm::Bool = true,)

Return a function performing linear or constant interpolation of the given data points.

The returned callable `g` closes over the (processed) data and, when called with
a scalar or vector of `x` values, returns the corresponding interpolated values.

# Arguments
- `x, y`: Numeric vectors giving the coordinates of the points to be interpolated.
- `method`: `:linear` or `:constant`.
- `yleft`, `yright`: Extrapolation values to the left/right; default determined by `rule`.
- `rule`: Tuple `(left_rule, right_rule)` with entries `1` or `2`:
  `1` → return `missing` outside the data range; `2` → use boundary value.
- `f`: For `:constant`, a number in `[0, 1]` controlling step continuity
  (`f = 0` right-continuous; `f = 1` left-continuous; intermediate values give
  `(1-f)*y_left + f*y_right`).
- `ties`: Handling of tied `x` values. Either `:ordered` (assume sorted, keep ties)
  or a function (e.g. `mean`, `min`, `max`) to aggregate `y` values for each
  distinct `x`.
- `na_rm`: If `true` (default), drop any rows with `missing` in `x` or `y`.
  If `false`, `missing` in `x` is not allowed; `missing` in `y` may propagate
  depending on `rule`.

# Details
- Inputs are normalized similarly to [`approx`](@ref). For `:linear`, at least two
  complete `(x, y)` pairs are required; for `:constant`, at least one.
- The function validates once at construction; subsequent calls do not re-validate.

!!! warning
    The returned function closes over internal arrays. It’s safe to use within the
    same Julia session, but if you serialize and reload it elsewhere, captured arrays
    and code versions must remain compatible.

# Returns
A callable `g(v)` where `v` may be a scalar or a vector of query points. The
result matches the behavior of `approx` for the same arguments.


# References
Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988) The New S Language. Wadsworth & Brooks/Cole.

# Examples
```julia
x = 1:10
y = randn(10)

g  = approxfun(x, y)                     # linear
gc = approxfun(x, y; method=:constant)   # step (right-continuous by default, f=0)

g(5.5)           # scalar query
g.(0:0.25:11)    # broadcasting over a range

# Different left/right extrapolation
h = approxfun(x, y; rule=(2, 1))
h.(0:0.5:11)

# Step function, left-continuous
gl = approxfun(x, y; method=:constant, f=1.0)
gl.(0:0.5:11)

# Ties handling
xt = [2, 2:4..., 4, 4, 5, 5, 7, 7, 7]
yt = [1:6..., 5, 4, 3:-1:1...]
gmean = approxfun(xt, yt; ties=mean)     # collapse ties
gord  = approxfun(xt, yt; ties=:ordered) # assume already ordered
gmean(xt)
```

"""
function approxfun(x::AbstractVector, y::AbstractVector;
                   method::Symbol=:linear,
                   yleft::Union{Float64,Nothing}=nothing,
                   yright::Union{Float64,Nothing}=nothing,
                   rule::Union{Integer,Tuple{<:Integer,<:Integer}}=(1,1),
                   f::Real=0.0,
                   ties::Union{Function,Symbol}=mean,
                   na_rm::Bool=true)

    rule_t = rule isa Integer ? (rule, rule) : rule
    length(rule_t) == 2 || throw(ArgumentError("rule must have 1 or 2 integers"))

    r = regularize_values(x, y; ties=ties, na_rm=na_rm)
    x = r.x; y = r.y

    nx = (na_rm || !r.keptNA) ? length(x) : sum(r.notNA)
    isnan(float(nx)) && error("invalid length(x)")

    if isnothing(yleft)
        yleft = (rule_t[1] == 1) ? NaN : float(y[1])
    end
    if isnothing(yright)
        yright = (rule_t[2] == 1) ? NaN : float(y[end])
    end

    method = method === :linear ? :linear : method === :constant ? :constant :
        throw(ArgumentError("invalid interpolation method"))
    f = float(f)

    xv = collect(Float64[ismissing(v) ? NaN : Float64(v) for v in x])
    yv = collect(Float64[ismissing(v) ? NaN : Float64(v) for v in y])
    _approxtest(xv, yv, method, f; na_rm=na_rm)

    return v -> begin
        if v isa AbstractVector
            vv = collect(float.(v))
            out = Vector{Float64}(undef, length(vv))
            @inbounds for i in eachindex(vv)
                out[i] = ismissing(vv[i]) ? NaN : _approx1(vv[i], xv, yv, method, yleft, yright, f)
            end
            out
        else
            vv = float(v)
            ismissing(vv) ? NaN : _approx1(vv, xv, yv, method, yleft, yright, f)
        end
    end
end

function approxfun(
    ;x::AbstractVector,
    y::AbstractVector,
    method::String = "linear",
    yleft::Union{Float64, Nothing} = nothing,
    yright::Union{Float64, Nothing} = nothing,
    rule::Tuple = (1, 1),
    f::Real = 0.0,
    ties::Union{Function, String} = mean,
    na_rm::Bool = true,)

    method = Symbol(method)

    if ties isa String
        ties = Symbol(ties)
    end

    approxfun(
    x,
    y,
    method = method,
    yleft = yleft,
    yright = yright,
    rule = rule,
    f = f,
    ties = ties,
    na_rm = na_rm,)
end