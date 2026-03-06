"""
    bandpass_coefs(low, high, maxcoef; freq_unit=:period)

Compute ideal bandpass filter impulse response coefficients B_{-maxcoef}, ..., B_0, ..., B_{maxcoef}.

The passband is specified by `low` and `high` in the frequency unit given by `freq_unit`:
- `:period` — low and high are periods (e.g., 6 and 32 quarters for business cycle)
- `:frequency` — low and high are angular frequencies in (0, pi)

Closed-form (Eq. 30 in Schleicher 2002):
    B_0 = (b - a) / pi
    B_j = [sin(b*j) - sin(a*j)] / (pi*j)   for j != 0

Returns a vector of length `2*maxcoef + 1` centered at index `maxcoef + 1`.
"""
function bandpass_coefs(low::Real, high::Real, maxcoef::Int; freq_unit::Symbol=:period)
    _check_arg(freq_unit, (:period, :frequency), "freq_unit")

    if freq_unit == :period
        low > 0 || throw(ArgumentError("low period must be positive, got $low"))
        high > low || throw(ArgumentError("high period must be greater than low period"))
        # Convert periods to angular frequencies: omega = 2*pi / period
        # Higher period = lower frequency, so a < b
        a = 2 * pi / high   # lower cutoff frequency
        b = 2 * pi / low    # upper cutoff frequency
    else
        a = low
        b = high
    end

    0 < a < b || throw(ArgumentError("Need 0 < low_freq < high_freq, got a=$a, b=$b"))
    b <= pi || throw(ArgumentError("Upper frequency must be <= pi, got $b"))

    n = 2 * maxcoef + 1
    B = zeros(n)
    center = maxcoef + 1
    B[center] = (b - a) / pi
    for j in 1:maxcoef
        B[center + j] = (sin(b * j) - sin(a * j)) / (pi * j)
        B[center - j] = B[center + j]  # symmetric
    end
    return B
end

"""
    hp_coefs(lambda, maxcoef)

Compute ideal Hodrick-Prescott highpass filter impulse response coefficients.

The HP filter transfer function is:
    H(omega) = 4*lambda*(1 - cos(omega))^2 / (1 + 4*lambda*(1 - cos(omega))^2)

Coefficients are computed via numerical inverse Fourier transform (cosine integral).
Returns a vector of length `2*maxcoef + 1` centered at index `maxcoef + 1`.
"""
function hp_coefs(lambda::Real, maxcoef::Int)
    lambda > 0 || throw(ArgumentError("lambda must be positive, got $lambda"))
    n = 2 * maxcoef + 1
    B = zeros(n)
    center = maxcoef + 1

    # B_0 = (1/pi) * integral_0^pi H(omega) d_omega
    B[center] = (1 / pi) * adaptive_simpsons(
        omega -> 4 * lambda * (1 - cos(omega))^2 / (1 + 4 * lambda * (1 - cos(omega))^2),
        0.0, pi
    )

    # B_j = (1/pi) * integral_0^pi H(omega) * cos(j*omega) d_omega
    for j in 1:maxcoef
        B[center + j] = (1 / pi) * adaptive_simpsons(
            omega -> 4 * lambda * (1 - cos(omega))^2 / (1 + 4 * lambda * (1 - cos(omega))^2) * cos(j * omega),
            0.0, pi
        )
        B[center - j] = B[center + j]
    end
    return B
end

"""
    butterworth_coefs(order, omega_c, maxcoef)

Compute ideal Butterworth highpass filter impulse response coefficients.

The Butterworth transfer function (squared magnitude) is:
    |H(omega)|^2 = lambda * (2 - 2*cos(omega))^(2n) / (1 + lambda * (2 - 2*cos(omega))^(2n))
where lambda = (1 / tan(omega_c / 2))^(2n) and n = order.

Returns a vector of length `2*maxcoef + 1` centered at index `maxcoef + 1`.
"""
function butterworth_coefs(order::Int, omega_c::Real, maxcoef::Int)
    order > 0 || throw(ArgumentError("Butterworth order must be positive, got $order"))
    0 < omega_c < pi || throw(ArgumentError("Cutoff frequency must be in (0, pi), got $omega_c"))

    n = order
    lam = (1 / tan(omega_c / 2))^(2 * n)

    ncoefs = 2 * maxcoef + 1
    B = zeros(ncoefs)
    center = maxcoef + 1

    function H(omega)
        g = (2 - 2 * cos(omega))^n
        return lam * g^2 / (1 + lam * g^2)
    end

    B[center] = (1 / pi) * adaptive_simpsons(H, 0.0, pi)
    for j in 1:maxcoef
        B[center + j] = (1 / pi) * adaptive_simpsons(
            omega -> H(omega) * cos(j * omega),
            0.0, pi
        )
        B[center - j] = B[center + j]
    end
    return B
end

"""
    ideal_filter_coefs(filter_type, maxcoef; kwargs...)

Dispatch to the appropriate ideal filter coefficient function.

# Arguments
- `filter_type::Symbol`: One of :bandpass, :hp, :butterworth, :custom
- `maxcoef::Int`: Number of coefficients on each side of center
- `kwargs...`: Filter-specific parameters:
  - :bandpass — `low`, `high`, `freq_unit`
  - :hp — `lambda`
  - :butterworth — `order`, `omega_c`
  - :custom — `transfer_fn` (function omega -> H(omega) on [0, pi])
"""
function ideal_filter_coefs(filter_type::Symbol, maxcoef::Int; kwargs...)
    kw = Dict(kwargs)
    if filter_type == :bandpass
        low = kw[:low]
        high = kw[:high]
        freq_unit = get(kw, :freq_unit, :period)
        return bandpass_coefs(low, high, maxcoef; freq_unit=freq_unit)
    elseif filter_type == :hp
        lambda = kw[:lambda]
        return hp_coefs(lambda, maxcoef)
    elseif filter_type == :butterworth
        order = kw[:order]
        omega_c = kw[:omega_c]
        return butterworth_coefs(order, omega_c, maxcoef)
    elseif filter_type == :custom
        transfer_fn = kw[:transfer_fn]
        return _custom_filter_coefs(transfer_fn, maxcoef)
    else
        throw(ArgumentError("Unknown filter_type: :$filter_type. Must be one of: :bandpass, :hp, :butterworth, :custom"))
    end
end

function _custom_filter_coefs(transfer_fn, maxcoef::Int)
    n = 2 * maxcoef + 1
    B = zeros(n)
    center = maxcoef + 1

    B[center] = (1 / pi) * adaptive_simpsons(transfer_fn, 0.0, pi)
    for j in 1:maxcoef
        B[center + j] = (1 / pi) * adaptive_simpsons(
            omega -> transfer_fn(omega) * cos(j * omega),
            0.0, pi
        )
        B[center - j] = B[center + j]
    end
    return B
end
