"""
    adaptive_simpsons(f, a, b; atol=1e-10, maxdepth=50)

Adaptive Simpson's quadrature with Richardson extrapolation.
Integrates `f` over `[a, b]` to absolute tolerance `atol`.
"""
function adaptive_simpsons(f, a::Real, b::Real; atol::Real=1e-10, maxdepth::Int=50)
    mid = (a + b) / 2
    fa, fb, fm = f(a), f(b), f(mid)
    whole = (b - a) / 6 * (fa + 4 * fm + fb)
    return _adaptive_simpsons_rec(f, a, b, fa, fb, fm, whole, atol, maxdepth)
end

function _adaptive_simpsons_rec(f, a, b, fa, fb, fm, whole, atol, depth)
    mid = (a + b) / 2
    lm = (a + mid) / 2
    rm = (mid + b) / 2
    flm = f(lm)
    frm = f(rm)
    left = (mid - a) / 6 * (fa + 4 * flm + fm)
    right = (b - mid) / 6 * (fm + 4 * frm + fb)
    combined = left + right
    err = (combined - whole) / 15
    if depth <= 0 || abs(err) < atol
        return combined + err
    end
    return _adaptive_simpsons_rec(f, a, mid, fa, fm, flm, left, atol / 2, depth - 1) +
           _adaptive_simpsons_rec(f, mid, b, fm, fb, frm, right, atol / 2, depth - 1)
end
