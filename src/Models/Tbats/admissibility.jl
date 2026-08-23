# ─── TBATS admissibility checks ───────────────────────────────────────────────
#
# Validates parameter admissibility: Box-Cox bounds, damping range,
# AR/MA polynomial roots, and eigenvalue stability.

function check_admissibility_tbats(
    D::AbstractMatrix{<:Real};
    box_cox::Union{Nothing,Float64} = nothing,
    small_phi::Union{Nothing,Float64} = nothing,
    ar_coefs::Union{Nothing,Vector{Float64}} = nothing,
    ma_coefs::Union{Nothing,Vector{Float64}} = nothing,
    tau::Int = 0,
    bc_lower::Real = 0.0,
    bc_upper::Real = 1.0,
)::Bool
    EPS = 1e-8
    RAD = 1.0 + 1e-2

    if box_cox !== nothing
        (box_cox <= bc_lower || box_cox >= bc_upper) && return false
    end

    if small_phi !== nothing
        (small_phi < 0.8 || small_phi > 1.0) && return false
    end

    if ar_coefs !== nothing
        p = 0
        @inbounds for i in eachindex(ar_coefs)
            if abs(ar_coefs[i]) > EPS
                p = i
            end
        end
        if p > 0
            coeffs = Vector{Float64}(undef, p + 1)
            coeffs[1] = 1.0
            @inbounds for i in 1:p
                coeffs[i + 1] = -ar_coefs[i]
            end
            rts = roots(Polynomial(coeffs))
            @inbounds for r in rts
                abs(r) < RAD && return false
            end
        end
    end

    if ma_coefs !== nothing
        q = 0
        @inbounds for i in eachindex(ma_coefs)
            if abs(ma_coefs[i]) > EPS
                q = i
            end
        end
        if q > 0
            coeffs = Vector{Float64}(undef, q + 1)
            coeffs[1] = 1.0
            @inbounds for i in 1:q
                coeffs[i + 1] = ma_coefs[i]
            end
            rts = roots(Polynomial(coeffs))
            @inbounds for r in rts
                abs(r) < RAD && return false
            end
        end
    end

    vals = eigvals(D)
    @inbounds for v in vals
        abs(v) >= RAD && return false
    end

    return true
end
