"""
    kw_decomposition(y; kwargs...) -> Decomposition

Decompose time series `y` into trend and cycle components using the Kolmogorov-Wiener
optimal finite-sample filter.

Returns a [`Decomposition`](@ref) where `trend + remainder = data` exactly. The cycle
(remainder) is computed via the KW optimal filter and the trend is obtained by subtraction.

# Keyword Arguments
- `filter_type::Symbol=:hp`: Filter type (:hp, :bandpass, :butterworth, :custom)
- `m::Int=1`: Seasonal period for auto_arima
- `arima_model::Union{ArimaFit,Nothing}=nothing`: Pre-fitted ARIMA model
- `lambda::Real=1600.0`: Smoothing parameter (HP filter)
- `low::Union{Real,Nothing}=nothing`: Lower period bound (bandpass)
- `high::Union{Real,Nothing}=nothing`: Upper period bound (bandpass)
- `freq_unit::Symbol=:period`: Frequency unit for bandpass (:period or :frequency)
- `order::Int=2`: Filter order (Butterworth)
- `omega_c::Union{Real,Nothing}=nothing`: Cutoff frequency (Butterworth)
- `transfer_fn::Union{Function,Nothing}=nothing`: Custom transfer function
- `maxcoef::Int=500`: Number of ideal filter coefficients per side
- `boxcox_lambda::Union{Nothing,Real}=nothing`: Box-Cox transformation parameter for `auto_arima`.
  Distinct from HP `lambda`. Ignored when `arima_model` is provided.
- `biasadj::Bool=false`: Bias adjustment for Box-Cox back-transform in `auto_arima`.
  Ignored when `arima_model` is provided.
- `arima_kwargs...`: Additional keyword arguments forwarded to `auto_arima` (e.g. `d`, `D`,
  `max_p`, `stepwise`, `ic`). Ignored when `arima_model` is provided.

# Examples
```julia
y = air_passengers()
d = kw_decomposition(y; m=12)
d.trend       # smooth trend component
d.remainder   # cyclical component
d.trend .+ d.remainder == d.data  # true (exact)
```
"""
function kw_decomposition(
    y::AbstractVector;
    filter_type::Symbol=:hp,
    m::Int=1,
    arima_model::Union{ArimaFit,Nothing}=nothing,
    lambda::Real=1600.0,
    low::Union{Real,Nothing}=nothing,
    high::Union{Real,Nothing}=nothing,
    freq_unit::Symbol=:period,
    order::Int=2,
    omega_c::Union{Real,Nothing}=nothing,
    transfer_fn::Union{Function,Nothing}=nothing,
    maxcoef::Int=500,
    boxcox_lambda::Union{Nothing,Real}=nothing,
    biasadj::Bool=false,
    arima_kwargs...,
)
    # Compute cycle (highpass) via KW optimal filter
    r_cycle = kolmogorov_wiener(y, filter_type;
        arima_model=arima_model, output=:cycle, m=m, lambda=lambda,
        low=low, high=high, freq_unit=freq_unit, order=order,
        omega_c=omega_c, transfer_fn=transfer_fn, maxcoef=maxcoef,
        boxcox_lambda=boxcox_lambda, biasadj=biasadj,
        arima_kwargs...)

    return _kw_decomposition_from_cycle(
        r_cycle.y,
        r_cycle.filtered;
        m=m,
        metadata=Dict{Symbol,Any}(
            :filter_type => filter_type,
            :arima_model => r_cycle.arima_model,
            :gamma => r_cycle.gamma,
            :d => r_cycle.d,
            :params => r_cycle.params,
        ),
    )
end
