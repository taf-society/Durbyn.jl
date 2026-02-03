function setup_params(y_in::AbstractVector;
                      max_ar_depth::Union{Int, Nothing}=nothing,
                      max_lag::Union{Int, Nothing}=nothing)

    n = length(y_in)

    # Hard minimum: need at least 5 observations for gamma computation (i=0:4 requires n >= 5)
    if n < 5
        throw(ArgumentError("Series too short (n=$n) for ARARMA. Need at least 5 observations."))
    end

    if n < 10
        @warn "Training data is too short (length=$n). The model may be unreliable."
    end

    if isnothing(max_ar_depth)
        if n > 40
            max_ar_depth = 26
        elseif n >= 13
            max_ar_depth = 13
        else  # 10 ≤ n < 13
            max_ar_depth = max(4, ceil(Int, n/3))
        end
    end

    if isnothing(max_lag)
        if n > 40
            max_lag = 40
        elseif n >= 13
            max_lag = 13
        else
            max_lag = max(4, ceil(Int, n/2))
        end
    end

    # Validate minimum values required for fallback lag (1,2,3,4) and gamma[5]
    if max_ar_depth < 4
        throw(ArgumentError("max_ar_depth must be at least 4. Got max_ar_depth=$max_ar_depth"))
    end
    if max_lag < 4
        throw(ArgumentError("max_lag must be at least 4. Got max_lag=$max_lag"))
    end

    # Clamp max_lag to n-1 to prevent indexing errors in gamma computation
    # (Since n >= 5 is guaranteed above, n-1 >= 4 always holds)
    if max_lag >= n
        clamped_max_lag = n - 1
        @warn "max_lag ($max_lag) exceeds series length - 1 ($(n - 1)). Clamping to $clamped_max_lag."
        max_lag = clamped_max_lag
    end

    # Also clamp max_ar_depth if it exceeds max_lag after clamping
    if max_ar_depth > max_lag
        clamped_max_ar_depth = max_lag
        @warn "max_ar_depth ($max_ar_depth) exceeds max_lag ($max_lag). Clamping to $clamped_max_ar_depth."
        max_ar_depth = clamped_max_ar_depth
    end

    if max_lag < max_ar_depth
        throw(
            ArgumentError(
                "max_lag must be greater than or equal to max_ar_depth. Got max_lag=$(max_lag), max_ar_depth=$(max_ar_depth)",
            ),
        )
    end

    return max_ar_depth, max_lag
end
