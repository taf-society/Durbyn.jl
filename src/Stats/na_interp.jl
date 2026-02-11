"""
    na_contiguous(x::AbstractArray)

Extract the longest contiguous segment of non-missing values from `x`.

# Arguments
- `x::AbstractArray`: Input array that may contain missing values.

# Returns
The longest contiguous segment of `x` without missing values.

# Example
```julia
x = [missing, 1.0, 2.0, 3.0, missing, 4.0, missing]
na_contiguous(x)  # Returns [1.0, 2.0, 3.0]
```
"""
function na_contiguous(x::AbstractArray)
    good = [!ismissing(v) && !(v isa AbstractFloat && isnan(v)) for v in x]
    if sum(good) == 0
        error("all times contain an NA")
    end
    tt = cumsum(Int[!g for g in good])
    ln = [sum(tt .== i) for i = 0:maximum(tt)]
    seg = findfirst(v -> v == maximum(ln), ln) - 1
    keep = tt .== seg
    st = findfirst(keep)

    if !good[st]
        st += 1
    end

    en = findlast(keep)
    omit = Int[]
    n = length(x)

    if st > 1
        append!(omit, 1:(st-1))
    end

    if en < n
        append!(omit, (en+1):n)
    end

    if length(omit) > 0
        x = x[st:en]
    end

    return x
end

"""
    na_fail(x::AbstractArray)

Return `x` unchanged if it contains no missing values; otherwise throw an error.

# Arguments
- `x::AbstractArray`: Input array to check.

# Returns
The input array `x` if no missing values are present.

# Throws
`ArgumentError` if `x` contains any missing values.

# Example
```julia
na_fail([1.0, 2.0, 3.0])  # Returns [1.0, 2.0, 3.0]
na_fail([1.0, missing, 3.0])  # Throws ArgumentError
```
"""
function na_fail(x::AbstractArray)
    has_na = any(v -> ismissing(v) || (v isa AbstractFloat && isnan(v)), x)
    if !has_na
        return x
    else
        throw(ArgumentError("missing values in object"))
    end
end

"""
    na_action(x::AbstractArray, type::String="na_contiguous"; m::Union{Int,Nothing}=nothing)

Handle missing data in a vector `x` based on the specified `type` of action.

# Arguments
- `x::AbstractArray`: The input vector containing data which may have missing values.
- `type::String`: The type of action to take on the missing data:
    - `"na_contiguous"` (default): Extract the longest contiguous segment without missing values.
    - `"na_interp"`: Interpolate missing values (requires `m` for seasonal data).
    - `"na_fail"`: Throw an error if any missing values are present.
- `m::Union{Int,Nothing}`: Seasonal period for `na_interp`. Required for seasonal interpolation.

# Returns
The vector `x` after applying the specified missing data handling action.

# Example
```julia
x = [1.0, 2.0, missing, 4.0, 5.0]
na_action(x, "na_contiguous")  # Returns longest contiguous segment
na_action(x, "na_interp")      # Returns [1.0, 2.0, 3.0, 4.0, 5.0] (interpolated)
na_action(x, "na_fail")        # Throws ArgumentError
```

# See also
[`na_contiguous`](@ref), [`na_interp`](@ref), [`na_fail`](@ref)
"""
function na_action(x::AbstractArray, type::String="na_contiguous"; m::Union{Int,Nothing}=nothing)
    if type == "na_contiguous"
        return na_contiguous(x)
    elseif type == "na_interp"
        return na_interp(x; m=m)
    elseif type == "na_fail"
        return na_fail(x)
    else
        error("Invalid type: $type. Must be one of \"na_contiguous\", \"na_interp\", or \"na_fail\".")
    end
end

"""
    na_interp(x::AbstractVector{T};
              m::Union{Int,Nothing}=nothing,
              lambda::Union{Nothing,Real}=nothing,
              linear::Union{Nothing,Bool}=nothing) where T

Interpolate missing values in a time series.

By default, uses linear interpolation for non-seasonal series. For seasonal series, a
robust STL decomposition is first computed. Then a linear interpolation is applied to the
seasonally adjusted data, and the seasonal component is added back.

# Arguments
- `x`: Time series vector (may contain `missing` or `NaN` values)
- `m`: Seasonal period. If `nothing` or `1`, series is treated as non-seasonal
- `lambda`: Optional Box-Cox transformation parameter. If provided, the series is
  transformed before interpolation and back-transformed after
- `linear`: Force linear interpolation. If `nothing` (default), linear interpolation
  is used when `m <= 1` or when there are fewer than `2*m` non-missing values

# Returns
A vector with missing values replaced by interpolated values.

# Details
- For non-seasonal series (`m <= 1` or `linear=true`), uses simple linear interpolation
  with boundary value extrapolation
- For seasonal series, the algorithm:
  1. Fits a preliminary model with Fourier terms and polynomial trend to fill initial gaps
  2. Applies robust MSTL decomposition
  3. Linearly interpolates the seasonally adjusted series
  4. Adds back the seasonal component
  5. If results are unstable (values far outside original range), falls back to linear

# Examples
```julia
# Non-seasonal interpolation
y = [1.0, 2.0, missing, 4.0, 5.0]
y_filled = na_interp(y)

# Seasonal interpolation
y_seasonal = [missing, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]
y_filled = na_interp(y_seasonal; m=4)

# With Box-Cox transformation
y_filled = na_interp(y; lambda=0.5)
```

# References
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.), OTexts.

# See also
[`mstl`](@ref), [`approx`](@ref)
"""
function na_interp(x::AbstractVector{T};
                   m::Union{Int,Nothing}=nothing,
                   lambda::Union{Nothing,Real}=nothing,
                   linear::Union{Nothing,Bool}=nothing) where T

    n = length(x)
    freq = isnothing(m) ? 1 : max(1, m)

    missng = [ismissing(v) || (v isa AbstractFloat && isnan(v)) for v in x]

    if sum(missng) == 0
        return collect(float.(coalesce.(x, NaN)))
    end

    origx = collect(float.(coalesce.(x, NaN)))

    non_miss_vals = origx[.!missng]
    if isempty(non_miss_vals)
        error("All values are missing")
    end
    rangex = extrema(non_miss_vals)
    drangex = rangex[2] - rangex[1]

    xu = copy(origx)

    n_valid = sum(.!missng)
    use_linear = if !isnothing(linear)
        linear
    else
        freq <= 1 || n_valid <= 2 * freq
    end

    λ = lambda
    if !isnothing(lambda)
        xu_valid = xu[.!missng]
        xu_transformed, λ = box_cox(xu_valid, freq; lambda=lambda)
        xu[.!missng] .= xu_transformed
    end

    tt = 1:n
    idx = tt[.!missng]

    if use_linear
        result = approx(idx, xu[idx]; xout=collect(tt), rule=(2, 2))
        xu = result.y
    else
        K = min(div(freq, 2), 5)

        fourier_terms = _fourier_matrix(n, freq, K)
        poly_degree = min(3, n - 1)
        trend_terms = hcat(ones(n), _poly_matrix(collect(tt), poly_degree))
        X = hcat(fourier_terms, trend_terms)

        X_valid = X[.!missng, :]
        y_valid = xu[.!missng]

        use_linear_fill = false
        try
            fit = ols(y_valid, X_valid)
            pred = X * fit.coef
            pred_miss = pred[missng]
            pred_range_ok = all(pred_miss .>= rangex[1] - 0.5 * drangex) &&
                           all(pred_miss .<= rangex[2] + 0.5 * drangex)
            if pred_range_ok
                xu[missng] .= pred_miss
            else
                use_linear_fill = true
            end
        catch
            use_linear_fill = true
        end

        if use_linear_fill
            result = approx(idx, xu[idx]; xout=collect(tt), rule=(2, 2))
            xu = result.y
        end

        for i in 1:n
            if missng[i]
                month = ((i - 1) % freq) + 1
                same_month_idx = [j for j in 1:n if ((j - 1) % freq) + 1 == month && !missng[j]]
                if !isempty(same_month_idx)
                    same_month_vals = origx[same_month_idx]
                    seasonal_mean = mean(same_month_vals)
                    if abs(xu[i] - seasonal_mean) > 0.3 * drangex
                        if length(same_month_idx) >= 2
                            trend_slope = (same_month_vals[end] - same_month_vals[1]) /
                                         (same_month_idx[end] - same_month_idx[1])
                            nearest_idx = same_month_idx[argmin(abs.(same_month_idx .- i))]
                            xu[i] = origx[nearest_idx] + trend_slope * (i - nearest_idx)
                        else
                            xu[i] = seasonal_mean
                        end
                    end
                end
            end
        end

        try
            mstl_fit = mstl(xu, freq; robust=true)

            seas_total = if isempty(mstl_fit.seasonals)
                zeros(n)
            else
                reduce(+, mstl_fit.seasonals)
            end
            sa = xu .- seas_total

            result = approx(idx, sa[idx]; xout=collect(tt), rule=(2, 2))
            sa_interp = result.y

            xu[missng] .= sa_interp[missng] .+ seas_total[missng]
        catch e
            result = approx(idx, origx[idx]; xout=collect(tt), rule=(2, 2))
            xu = result.y
        end
    end

    if !isnothing(λ)
        xu = inv_box_cox(xu; lambda=λ)
    end

    if !use_linear
        xu_max, xu_min = maximum(xu), minimum(xu)
        if xu_max > rangex[2] + 0.5 * drangex || xu_min < rangex[1] - 0.5 * drangex
            return na_interp(origx; m=m, lambda=lambda, linear=true)
        end
    end

    return xu
end

"""
Helper function to create Fourier design matrix for na_interp.
"""
function _fourier_matrix(n::Int, period::Int, K::Int)
    if K <= 0
        return zeros(n, 0)
    end

    tt = 1:n
    ncols = 2 * K
    X = zeros(n, ncols)

    for k in 1:K
        freq = k / period
        X[:, 2k-1] .= sin.(2π .* freq .* tt)
        X[:, 2k] .= cos.(2π .* freq .* tt)
    end

    return X
end

"""
Helper function to create orthogonal polynomial design matrix for na_interp.
Uses three-term recurrence to match R's poly() function exactly.
"""
function _poly_matrix(tt::AbstractVector, degree::Int)
    n = length(tt)
    if degree <= 0
        return zeros(n, 0)
    end

    x = collect(Float64, tt)

    alpha = zeros(degree)
    norm2 = zeros(degree + 2)
    norm2[1] = 1.0
    norm2[2] = Float64(n)

    P = zeros(n, degree + 1)
    P[:, 1] .= 1.0

    xbar = mean(x)
    alpha[1] = xbar

    P[:, 2] .= x .- xbar
    norm2[3] = sum(P[:, 2].^2)

    for k in 2:degree
        alpha[k] = sum(x .* P[:, k].^2) / norm2[k + 1]
        P[:, k + 1] .= (x .- alpha[k]) .* P[:, k] .- (norm2[k + 1] / norm2[k]) .* P[:, k - 1]
        norm2[k + 2] = sum(P[:, k + 1].^2)
    end

    result = zeros(n, degree)
    for k in 1:degree
        result[:, k] .= P[:, k + 1] ./ sqrt(norm2[k + 2])
    end

    return result
end
