"""
    kolmogorov_wiener(y, filter_type; kwargs...)

Apply the Kolmogorov-Wiener optimal finite-sample filter to time series `y`.

The filter minimizes mean squared error relative to an ideal symmetric filter by
using the autocovariance structure estimated from an ARIMA model. This provides
optimal endpoint handling for bandpass, Hodrick-Prescott, and Butterworth filters.

Based on Schleicher (2004), "Kolmogorov-Wiener Filters for Finite Time-Series"
(SSRN, DOI: 10.2139/ssrn.769584).

# Arguments
- `y::AbstractVector`: Input time series
- `filter_type::Symbol`: One of :bandpass, :hp, :butterworth, :custom

# Keyword Arguments
- `arima_model::Union{ArimaFit,Nothing}=nothing`: Pre-fitted ARIMA model. If `nothing`,
  `auto_arima` is called automatically.
- `m::Int=1`: Seasonal period for auto_arima (ignored if `arima_model` is provided)
- `low::Real`: Lower period bound (bandpass only)
- `high::Real`: Upper period bound (bandpass only)
- `freq_unit::Symbol=:period`: Frequency unit for bandpass (:period or :frequency)
- `lambda::Real=1600.0`: Smoothing parameter (HP filter only)
- `order::Int=2`: Filter order (Butterworth only)
- `omega_c::Union{Real,Nothing}=nothing`: Cutoff frequency (Butterworth only)
- `transfer_fn::Union{Function,Nothing}=nothing`: Custom transfer function omega -> H(omega) on [0,pi]
- `maxcoef::Int=500`: Number of ideal filter coefficients on each side of center
- `output::Symbol=:cycle`: Output component — :cycle (highpass) or :trend (lowpass)
- `boxcox_lambda::Union{Nothing,Real}=nothing`: Box-Cox transformation parameter for `auto_arima`.
  Distinct from HP `lambda`. Ignored when `arima_model` is provided.
- `biasadj::Bool=false`: Bias adjustment for Box-Cox back-transform in `auto_arima`.
  Ignored when `arima_model` is provided.
- `arima_kwargs...`: Additional keyword arguments forwarded to `auto_arima` (e.g. `d`, `D`,
  `max_p`, `stepwise`, `ic`). Ignored when `arima_model` is provided.

# Returns
A [`KWFilterResult`](@ref) containing the filtered series, weight matrix, and diagnostics.

# Examples
```julia
y = air_passengers()
# HP filter with lambda=1600
r = kolmogorov_wiener(y, :hp; lambda=1600, m=12)

# Bandpass filter for business cycle (6-32 quarters)
r = kolmogorov_wiener(y, :bandpass; low=6, high=32, m=4)

# Butterworth filter
r = kolmogorov_wiener(y, :butterworth; order=2, omega_c=pi/16, m=12)

# With explicit ARIMA constraints
r = kolmogorov_wiener(y, :hp; m=12, d=1, D=1, max_p=3)

# With Box-Cox transform (boxcox_lambda is separate from HP lambda)
r = kolmogorov_wiener(y, :hp; m=12, boxcox_lambda=0.0, biasadj=true)
```
"""
function kolmogorov_wiener(
    y::AbstractVector,
    filter_type::Symbol;
    arima_model::Union{ArimaFit,Nothing}=nothing,
    m::Int=1,
    low::Union{Real,Nothing}=nothing,
    high::Union{Real,Nothing}=nothing,
    freq_unit::Symbol=:period,
    lambda::Real=1600.0,
    order::Int=2,
    omega_c::Union{Real,Nothing}=nothing,
    transfer_fn::Union{Function,Nothing}=nothing,
    maxcoef::Int=500,
    output::Symbol=:cycle,
    boxcox_lambda::Union{Nothing,Real}=nothing,
    biasadj::Bool=false,
    arima_kwargs...,
)
    _check_arg(filter_type, (:bandpass, :hp, :butterworth, :custom), "filter_type")
    _check_arg(output, (:cycle, :trend), "output")

    T = length(y)
    T >= 3 || throw(ArgumentError("Series must have at least 3 observations, got $T"))

    yf = Float64.(y)

    # Step 1: Fit ARIMA model if not provided
    if isnothing(arima_model)
        arima_model = auto_arima(yf, m; lambda=boxcox_lambda, biasadj=biasadj, arima_kwargs...)
    end

    # Step 2: Extract ARMA parameters and autocovariance
    phi, theta, sigma2, d = extract_arma_for_kw(arima_model)
    gamma_maxlag = maxcoef + T
    gamma = arma_autocovariance(phi, theta, sigma2, gamma_maxlag)

    # Step 3: Compute ideal filter coefficients
    filter_kwargs = _build_filter_kwargs(filter_type; low=low, high=high,
        freq_unit=freq_unit, lambda=lambda, order=order, omega_c=omega_c,
        transfer_fn=transfer_fn)
    ideal_B = ideal_filter_coefs(filter_type, maxcoef; filter_kwargs...)

    # For trend output, we want the lowpass component: I - H (complement of highpass)
    # For cycle output with a highpass filter (hp, butterworth), use as-is
    # For bandpass, output=:trend means complement
    if output == :trend
        center = maxcoef + 1
        trend_B = -copy(ideal_B)
        trend_B[center] += 1.0  # I - H: identity minus highpass
        working_B = trend_B
    else
        working_B = ideal_B
    end

    # Step 4: Compute optimal filter weights for each observation
    weights = zeros(T, T)
    filtered = zeros(T)

    for t in 1:T
        n1 = t - 1      # observations before t
        n2 = T - t       # observations after t

        B_hat = compute_optimal_filter(gamma, working_B, n1, n2, d)
        weights[t, :] = B_hat
        filtered[t] = dot(B_hat, yf)
    end

    # Build params dict
    params = Dict{Symbol,Any}()
    for (k, v) in filter_kwargs
        params[k] = v
    end

    return KWFilterResult(
        filtered, weights, ideal_B, filter_type,
        arima_model, gamma, d, yf, output, params
    )
end

function _build_filter_kwargs(filter_type::Symbol;
    low, high, freq_unit, lambda, order, omega_c, transfer_fn)
    if filter_type == :bandpass
        isnothing(low) && throw(ArgumentError("bandpass filter requires `low` parameter"))
        isnothing(high) && throw(ArgumentError("bandpass filter requires `high` parameter"))
        return pairs((low=low, high=high, freq_unit=freq_unit))
    elseif filter_type == :hp
        return pairs((lambda=lambda,))
    elseif filter_type == :butterworth
        isnothing(omega_c) && throw(ArgumentError("butterworth filter requires `omega_c` parameter"))
        return pairs((order=order, omega_c=omega_c))
    elseif filter_type == :custom
        isnothing(transfer_fn) && throw(ArgumentError("custom filter requires `transfer_fn` parameter"))
        return pairs((transfer_fn=transfer_fn,))
    end
end
