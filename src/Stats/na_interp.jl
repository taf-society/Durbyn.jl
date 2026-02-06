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

    # Identify missing values (both `missing` and `NaN`)
    missng = [ismissing(v) || (v isa AbstractFloat && isnan(v)) for v in x]

    # If no missing values, return copy of original
    if sum(missng) == 0
        return collect(float.(coalesce.(x, NaN)))
    end

    # Convert to Float64, replacing missing with NaN
    origx = collect(float.(coalesce.(x, NaN)))

    # Get range of non-missing values for stability check later
    non_miss_vals = origx[.!missng]
    if isempty(non_miss_vals)
        error("All values are missing")
    end
    rangex = extrema(non_miss_vals)
    drangex = rangex[2] - rangex[1]

    # Working copy
    xu = copy(origx)

    # Determine if we should use linear interpolation
    n_valid = sum(.!missng)
    use_linear = if !isnothing(linear)
        linear
    else
        freq <= 1 || n_valid <= 2 * freq
    end

    # Apply Box-Cox if requested
    λ = lambda
    if !isnothing(lambda)
        # Only transform non-missing values
        xu_valid = xu[.!missng]
        xu_transformed, λ = box_cox(xu_valid, freq; lambda=lambda)
        # Put transformed values back
        xu[.!missng] .= xu_transformed
    end

    tt = 1:n
    idx = tt[.!missng]

    if use_linear
        # Simple linear interpolation with boundary extrapolation
        result = approx(idx, xu[idx]; xout=collect(tt), rule=(2, 2))
        xu = result.y
    else
        # Seasonal interpolation using MSTL
        # First, fill gaps with a preliminary fit using Fourier + polynomial
        K = min(div(freq, 2), 5)

        # Build design matrix: Fourier terms + polynomial trend
        fourier_terms = _fourier_matrix(n, freq, K)
        poly_degree = min(max(div(n, 10), 1), 6)
        poly_terms = _poly_matrix(tt, poly_degree)
        X = hcat(fourier_terms, poly_terms)

        # Get non-missing rows for fitting
        X_valid = X[.!missng, :]
        y_valid = xu[.!missng]

        # Fit OLS and predict missing values
        try
            fit = ols(y_valid, X_valid)
            pred = X * fit.coef
            xu[missng] .= pred[missng]
        catch
            # If OLS fails, use simple linear interpolation for initial fill
            result = approx(idx, xu[idx]; xout=collect(tt), rule=(2, 2))
            xu = result.y
        end

        # Now apply robust MSTL
        try
            mstl_fit = mstl(xu, freq; robust=true)

            # Get seasonally adjusted data
            seas_total = if isempty(mstl_fit.seasonals)
                zeros(n)
            else
                reduce(+, mstl_fit.seasonals)
            end
            sa = xu .- seas_total

            # Linearly interpolate the seasonally adjusted series
            result = approx(idx, sa[idx]; xout=collect(tt), rule=(2, 2))
            sa_interp = result.y

            # Add seasonal component back for missing positions
            xu[missng] .= sa_interp[missng] .+ seas_total[missng]
        catch e
            # If MSTL fails, fall back to linear interpolation
            result = approx(idx, origx[idx]; xout=collect(tt), rule=(2, 2))
            xu = result.y
        end
    end

    # Back-transform if Box-Cox was applied
    if !isnothing(λ)
        xu = inv_box_cox(xu; lambda=λ)
    end

    # Stability check: if values are too far from original range, use linear
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
Helper function to create polynomial design matrix for na_interp.
"""
function _poly_matrix(tt::AbstractVector, degree::Int)
    n = length(tt)
    # Normalize to [-1, 1] for numerical stability
    t_norm = 2 .* (tt .- minimum(tt)) ./ max(1, maximum(tt) - minimum(tt)) .- 1

    X = zeros(n, degree)
    for d in 1:degree
        X[:, d] .= t_norm .^ d
    end

    return X
end
