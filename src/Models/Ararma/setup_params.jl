function setup_params(y_in::AbstractVector;
                      max_ar_depth::Union{Int, Nothing}=nothing,
                      max_lag::Union{Int, Nothing}=nothing)

    n = length(y_in)

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

    if max_lag < max_ar_depth
        throw(
            ArgumentError(
                "max_lag must be greater than or equal to max_ar_depth. Got max_lag=$(max_lag), max_ar_depth=$(max_ar_depth)",
            ),
        )
    end

    return max_ar_depth, max_lag
end
