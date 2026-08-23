# ─── BATS admissibility and seasonal seed initialization ─────────────────────
#
# Validation routines for parameter admissibility (Box-Cox bounds, damping
# range, AR/MA polynomial roots, eigenvalue stability) and helpers for
# resolving seasonal aliasing and computing initial seasonal states from
# regression coefficients.

function check_admissibility(
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


function cut_w(
    use_beta::Bool,
    w_tilda_transpose::AbstractMatrix{T},
    seasonal_periods::AbstractVector{Int},
    p::Int = 0,
    q::Int = 0,
) where {T}

    n_seasons = length(seasonal_periods)

    mask_vector = zeros(Int, n_seasons)


    i = n_seasons
    while i > 1
        for j = 1:(i-1)
            if seasonal_periods[i] % seasonal_periods[j] == 0
                mask_vector[j] = 1
            end
        end
        i -= 1
    end


    if n_seasons > 1
        for s = n_seasons:-1:2
            for j = (s-1):-1:1
                hcf = gcd(seasonal_periods[s], seasonal_periods[j])
                if hcf != 1
                    if mask_vector[s] != 1 && mask_vector[j] != 1
                        mask_vector[s] = -hcf
                    end
                end
            end
        end
    end


    n_cols = size(w_tilda_transpose, 2)

    cols = collect(1:n_cols)

    w_pos_counter = 1
    w_pos = 1
    if use_beta
        w_pos += 1
    end

    for s in seasonal_periods
        mv = mask_vector[w_pos_counter]

        if mv == 1

            first_del = w_pos + 1
            last_del = w_pos + s
            deleteat!(cols, first_del:last_del)

        elseif mv < 0

            w_pos += s
            first_del = w_pos + mv + 1
            last_del = w_pos
            deleteat!(cols, first_del:last_del)
            w_pos += mv

        else
            w_pos += s
            deleteat!(cols, w_pos)
            w_pos -= 1
        end

        w_pos_counter += 1
    end

    if p != 0 || q != 0
        total_cut = p + q
        if total_cut > 0
            start_cut = length(cols) - total_cut + 1
            deleteat!(cols, start_cut:length(cols))
        end
    end

    w_cut = w_tilda_transpose[:, cols]

    return (matrix = w_cut, mask_vector = mask_vector)
end

function calc_seasonal_seeds(
    use_beta::Bool,
    coefs::AbstractVector{T},
    seasonal_periods::AbstractVector{Int},
    mask_vector::AbstractVector{Int},
    p::Int = 0,
    q::Int = 0,
) where {T<:Real}

    x_pos_counter = 1
    sum_k = zero(T)

    if use_beta
        x_pos = 2
        new_x_nought = reshape(coefs[1:2], :, 1)
    else
        x_pos = 1
        new_x_nought = reshape(coefs[1:1], :, 1)
    end

    x_pos_counter = 1

    for s in seasonal_periods
        mv = mask_vector[x_pos_counter]

        if mv == 1
            season = zeros(T, s, 1)
            new_x_nought = vcat(new_x_nought, season)

        elseif mv < 0

            last_idx = x_pos + s + mv
            extract = coefs[(x_pos+1):last_idx]

            k = sum(extract)
            sum_k += k / s

            current_periodicity = extract .- (k / s)
            current_periodicity_mat = reshape(current_periodicity, :, 1)

            additional = fill(-k / s, -mv, 1)
            current_periodicity_mat = vcat(current_periodicity_mat, additional)

            new_x_nought = vcat(new_x_nought, current_periodicity_mat)

            x_pos = x_pos + s + mv

        else

            last_idx = x_pos + s - 1
            slice = coefs[(x_pos+1):last_idx]

            k = sum(slice)
            sum_k += k / s

            current_periodicity = vcat(slice .- (k / s), -k / s)
            current_periodicity_mat = reshape(current_periodicity, :, 1)

            new_x_nought = vcat(new_x_nought, current_periodicity_mat)

            x_pos = x_pos + s - 1
        end

        x_pos_counter += 1
    end

    if p != 0 || q != 0
        arma_len = p + q
        arma_seed_states = zeros(T, arma_len, 1)
        x_nought = vcat(new_x_nought, arma_seed_states)
    else
        x_nought = new_x_nought
    end

    return x_nought
end
