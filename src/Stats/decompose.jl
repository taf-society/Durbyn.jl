raw"""
    Classical Time Series Decomposition

Decomposes a time series `x` into trend, seasonal, and residual components
under either an additive or multiplicative model:

```math
\text{Additive: } x_i = \hat{T}_i + S_i + R_i
```

```math
\text{Multiplicative: } x_i = \hat{T}_i \cdot S_i \cdot R_i
```

# Reference
- Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.), Sec. 1.5.2 (Method S1).
"""

"""
    DecomposedTimeSeries

A structure representing a decomposed time series.

# Fields
- `x::AbstractVector`: Original time series.
- `seasonal::AbstractVector`: Seasonal component.
- `trend::AbstractVector`: Trend component.
- `random::AbstractVector`: Residual component.
- `figure::AbstractVector`: Seasonal figures of length `m`.
- `type::Symbol`: `:additive` or `:multiplicative`.
- `m::Int`: Frequency.
"""
struct DecomposedTimeSeries
    x::AbstractVector
    seasonal::AbstractVector
    trend::AbstractVector
    random::AbstractVector
    figure::AbstractVector
    type::Symbol
    m::Int
end

_is_missing_or_nan(v) = ismissing(v) || (v isa AbstractFloat && isnan(v))
const _DECOMP_MODELS = (:additive, :multiplicative)

function _check_decompose_type(type::Symbol)::Symbol
    type in _DECOMP_MODELS && return type
    throw(ArgumentError("decomposition type must be :additive or :multiplicative"))
end

function _to_float_or_missing(x::AbstractVector)
    out = Vector{Union{Missing, Float64}}(undef, length(x))
    @inbounds for i in eachindex(x)
        xi = x[i]
        out[i] = _is_missing_or_nan(xi) ? missing : Float64(xi)
    end
    return out
end

function _to_nan_or_float(x::AbstractVector{<:Union{Missing, Real}})::Vector{Float64}
    out = Vector{Float64}(undef, length(x))
    @inbounds for i in eachindex(x)
        xi = x[i]
        out[i] = ismissing(xi) ? NaN : Float64(xi)
    end
    return out
end

raw"""
    symmetric_ma(x, m) -> Vector{Union{Missing,Float64}}

Estimate trend with a symmetric moving average of period `m`.

For even `m`:

```math
\hat{T}_i = \frac{0.5\,x_{i-m/2} + \sum_{j=-(m/2)+1}^{(m/2)-1} x_{i+j} + 0.5\,x_{i+m/2}}{m}
```

For odd `m`:

```math
\hat{T}_i = \frac{1}{m}\sum_{j=-(m-1)/2}^{(m-1)/2} x_{i+j}
```

# Reference
- Brockwell, P. J., & Davis, R. A. (2016). Eq. 1.5.12 (even `m`) and Eq. 1.5.5 (odd `m`).
"""
function symmetric_ma(
    x::AbstractVector{<:Union{Missing, Real}},
    m::Int,
)::Vector{Union{Missing, Float64}}
    n = length(x)
    T = Vector{Union{Missing, Float64}}(missing, n)

    m <= 0 && throw(ArgumentError("m must be positive"))
    n < m && return T

    if iseven(m)
        half = m ÷ 2
        for i in (half + 1):(n - half)
            left = x[i - half]
            right = x[i + half]
            if ismissing(left) || ismissing(right)
                continue
            end

            s = 0.5 * Float64(left) + 0.5 * Float64(right)
            valid = true
            @inbounds for j in (-half + 1):(half - 1)
                xij = x[i + j]
                if ismissing(xij)
                    valid = false
                    break
                end
                s += Float64(xij)
            end

            if valid
                T[i] = s / m
            end
        end
    else
        half = (m - 1) ÷ 2
        for i in (half + 1):(n - half)
            s = 0.0
            valid = true
            @inbounds for j in -half:half
                xij = x[i + j]
                if ismissing(xij)
                    valid = false
                    break
                end
                s += Float64(xij)
            end
            if valid
                T[i] = s / m
            end
        end
    end

    return T
end

function _symmetric_filter(
    x::AbstractVector{<:Union{Missing, Real}},
    w::AbstractVector{<:Real},
)::Vector{Union{Missing, Float64}}
    nx = length(x)
    nf = length(w)
    out = Vector{Union{Missing, Float64}}(missing, nx)
    shift = nf ÷ 2
    wf = Float64.(w)

    for i = 1:nx
        i1 = i + shift - (nf - 1)
        i2 = i + shift
        if i1 < 1 || i2 > nx
            continue
        end

        s = 0.0
        valid = true
        @inbounds for j = 1:nf
            idx = i + shift - j + 1
            xij = x[idx]
            if ismissing(xij)
                valid = false
                break
            end
            s += wf[j] * Float64(xij)
        end

        if valid
            out[i] = s
        end
    end

    return out
end

raw"""
    detrend(x, T, model) -> Vector{Union{Missing,Float64}}

Remove trend to isolate seasonality and residual variation.

```math
\text{Additive: } D_i = x_i - \hat{T}_i
```

```math
\text{Multiplicative: } D_i = x_i / \hat{T}_i
```

# Reference
- Brockwell, P. J., & Davis, R. A. (2016), Sec. 1.5.2.1.
"""
function detrend(x::AbstractVector, T::AbstractVector, model::Symbol)
    length(x) == length(T) || throw(ArgumentError("x and trend must have same length"))
    D = Vector{Union{Missing, Float64}}(missing, length(x))
    model = _check_decompose_type(model)

    if model === :additive
        @inbounds for i in eachindex(x, T)
            xi, Ti = x[i], T[i]
            if ismissing(xi) || ismissing(Ti)
                continue
            end
            D[i] = Float64(xi) - Float64(Ti)
        end
    else
        @inbounds for i in eachindex(x, T)
            xi, Ti = x[i], T[i]
            if ismissing(xi) || ismissing(Ti) || Ti == 0.0
                continue
            end
            D[i] = Float64(xi) / Float64(Ti)
        end
    end

    return D
end

raw"""
    seasonal_figures(D, m, model) -> Vector{Float64}

Estimate and normalize one seasonal figure per period index `k = 1, ..., m`.

Step 1 (cycle-wise averaging):

```math
F_k = \frac{1}{N_k}\sum_j D_{k + m(j-1)}
```

Step 2 (normalization):

```math
\bar{F} = \frac{1}{m}\sum_{k=1}^m F_k
```

```math
\text{Additive: } F_k \leftarrow F_k - \bar{F}
```

```math
\text{Multiplicative: } F_k \leftarrow F_k / \bar{F}
```

# Reference
- Brockwell, P. J., & Davis, R. A. (2016), Sec. 1.5.2.1 and Eq. 1.5.13 (additive normalization).
"""
function seasonal_figures(D::AbstractVector, m::Int, model::Symbol)::Vector{Float64}
    m <= 0 && throw(ArgumentError("m must be positive"))
    model = _check_decompose_type(model)
    seasonal_position_sum = zeros(Float64, m)
    seasonal_position_count = zeros(Int, m)

    @inbounds for observation_index in eachindex(D)
        detrended_value = D[observation_index]
        if !ismissing(detrended_value) && !_is_missing_or_nan(detrended_value)
            season_position = mod(observation_index - 1, m) + 1
            seasonal_position_sum[season_position] += Float64(detrended_value)
            seasonal_position_count[season_position] += 1
        end
    end

    seasonal_figures_raw = zeros(Float64, m)
    n_positions_with_data = 0
    seasonal_figure_sum = 0.0
    @inbounds for season_position in 1:m
        n_at_position = seasonal_position_count[season_position]
        if n_at_position > 0
            seasonal_figure = seasonal_position_sum[season_position] / n_at_position
            seasonal_figures_raw[season_position] = seasonal_figure
            seasonal_figure_sum += seasonal_figure
            n_positions_with_data += 1
        end
    end

    n_positions_with_data == 0 && throw(ArgumentError("cannot estimate seasonal figures from empty detrended subseries"))
    seasonal_figure_mean = seasonal_figure_sum / n_positions_with_data

    if model === :additive
        @inbounds for season_position in 1:m
            if seasonal_position_count[season_position] > 0
                seasonal_figures_raw[season_position] -= seasonal_figure_mean
            else
                seasonal_figures_raw[season_position] = 0.0
            end
        end
    else
        seasonal_figure_mean == 0.0 && throw(ArgumentError("seasonal figure mean is zero for multiplicative model"))
        @inbounds for season_position in 1:m
            if seasonal_position_count[season_position] > 0
                seasonal_figures_raw[season_position] /= seasonal_figure_mean
            else
                seasonal_figures_raw[season_position] = 1.0
            end
        end
    end

    return seasonal_figures_raw
end

raw"""
    seasonal_component(F, n) -> Vector{Float64}

Tile seasonal figures over the full series length:

```math
S_i = F_{((i-1)\bmod m)+1}, \quad i = 1,\ldots,n
```

# Reference
- Brockwell, P. J., & Davis, R. A. (2016), Sec. 1.5.2.1.
"""
function seasonal_component(F::AbstractVector, n::Int)::Vector{Float64}
    m = length(F)
    seasonal_series = Vector{Float64}(undef, n)
    @inbounds for observation_index in 1:n
        season_position = mod(observation_index - 1, m) + 1
        seasonal_series[observation_index] = Float64(F[season_position])
    end
    return seasonal_series
end

raw"""
    residual(x, S, T, model) -> Vector{Union{Missing,Float64}}

Compute residuals after removing trend and seasonality:

```math
\text{Additive: } R_i = x_i - S_i - \hat{T}_i
```

```math
\text{Multiplicative: } R_i = x_i / (S_i \cdot \hat{T}_i)
```

# Reference
- Brockwell, P. J., & Davis, R. A. (2016), Sec. 1.5.2.1.
"""
function residual(x::AbstractVector, S::AbstractVector, T::AbstractVector, model::Symbol)
    length(x) == length(S) == length(T) || throw(ArgumentError("x, S, and T must have same length"))
    R = Vector{Union{Missing, Float64}}(missing, length(x))
    model = _check_decompose_type(model)

    if model === :additive
        @inbounds for i in eachindex(x, S, T)
            xi, Si, Ti = x[i], S[i], T[i]
            if ismissing(xi) || ismissing(Ti) || _is_missing_or_nan(Si)
                continue
            end
            R[i] = Float64(xi) - Float64(Si) - Float64(Ti)
        end
    else
        @inbounds for i in eachindex(x, S, T)
            xi, Si, Ti = x[i], S[i], T[i]
            if ismissing(xi) || ismissing(Ti) || _is_missing_or_nan(Si) || Si == 0.0 || Ti == 0.0
                continue
            end
            R[i] = Float64(xi) / (Float64(Si) * Float64(Ti))
        end
    end

    return R
end

function _validate_decompose_inputs(x::AbstractVector, m::Int, type::Symbol)
    valid_n = count(v -> !_is_missing_or_nan(v), x)
    if m <= 1 || valid_n < 2 * m
        throw(ArgumentError("need at least two complete seasonal cycles with m >= 2"))
    end
    _check_decompose_type(type)
end

raw"""
    decompose(; x, m, type=:additive, filter=nothing) -> DecomposedTimeSeries

Classical decomposition by moving averages.

The decomposition model is:

```math
\text{Additive: } x_i = \hat{T}_i + S_i + R_i
```

```math
\text{Multiplicative: } x_i = \hat{T}_i \cdot S_i \cdot R_i
```

with symmetric moving-average trend estimate:

```math
\text{Even }m:\quad
\hat{T}_i =
\frac{0.5\,x_{i-m/2} + \sum_{j=-(m/2)+1}^{(m/2)-1} x_{i+j} + 0.5\,x_{i+m/2}}{m}
```

```math
\text{Odd }m:\quad
\hat{T}_i = \frac{1}{m}\sum_{j=-(m-1)/2}^{(m-1)/2} x_{i+j}
```

# Arguments
- `x::AbstractVector`: Input time series.
- `m::Int`: Seasonal frequency.
- `type::Symbol`: `:additive` or `:multiplicative`.
- `filter`: Optional custom symmetric filter coefficients.

# Returns
`DecomposedTimeSeries` with `x`, `trend`, `seasonal`, `random`, `figure`, `type`, and `m`.

# References
- Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.), Sec. 1.5.2.
"""
function decompose(;
    x::AbstractVector,
    m::Int,
    type::Symbol = :additive,
    filter::Union{Nothing, AbstractVector} = nothing,
)
    _validate_decompose_inputs(x, m, type)

    x_data = _to_float_or_missing(x)
    trend_missing = if isnothing(filter)
        symmetric_ma(x_data, m)
    else
        isempty(filter) && throw(ArgumentError("filter must not be empty"))
        _symmetric_filter(x_data, Float64.(filter))
    end

    detrended = detrend(x_data, trend_missing, type)
    figure = seasonal_figures(detrended, m, type)
    seasonal = seasonal_component(figure, length(x_data))
    random_missing = residual(x_data, seasonal, trend_missing, type)

    trend = _to_nan_or_float(trend_missing)
    random = _to_nan_or_float(random_missing)

    return DecomposedTimeSeries(x, seasonal, trend, random, figure, type, m)
end
