function loess_estimate!(y::AbstractVector{Float64}, n::Int, bandwidth::Int, degree::Int,
                         eval_point::Float64, left_bound::Int, right_bound::Int,
                         w::AbstractVector{Float64}, use_weights::Bool, robustness_weights::AbstractVector{Float64})

    data_range = float(n) - 1.0
    half_width = max(eval_point - float(left_bound), float(right_bound) - eval_point)
    if bandwidth > n
        half_width += float((bandwidth - n) ÷ 2)
    end
    upper_threshold = 0.999 * half_width
    lower_threshold = 0.001 * half_width

    weight_sum = 0.0
    for j in left_bound:right_bound
        dist = abs(float(j) - eval_point)
        if dist <= upper_threshold
            if dist <= lower_threshold || half_width == 0.0
                w[j] = 1.0
            else
                normalized_dist = dist / half_width
                w[j] = (1.0 - normalized_dist^3)^3
            end
            if use_weights
                w[j] *= robustness_weights[j]
            end
            weight_sum += w[j]
        else
            w[j] = 0.0
        end
    end

    if weight_sum <= 0.0
        return 0.0, false
    end

    inv_weight_sum = 1.0 / weight_sum
    for j in left_bound:right_bound
        w[j] *= inv_weight_sum
    end

    if half_width > 0.0 && degree > 0

        weighted_mean_x = 0.0
        for j in left_bound:right_bound
            weighted_mean_x += w[j] * float(j)
        end
        slope_num = eval_point - weighted_mean_x
        slope_denom = 0.0
        for j in left_bound:right_bound
            dev = float(j) - weighted_mean_x
            slope_denom += w[j] * dev^2
        end

        if sqrt(slope_denom) > 0.001 * data_range
            slope_num /= slope_denom
            for j in left_bound:right_bound
                w[j] = w[j] * (slope_num * (float(j) - weighted_mean_x) + 1.0)
            end
        end
    end

    ys = 0.0
    for j in left_bound:right_bound
        ys += w[j] * y[j]
    end
    return ys, true
end

function loess_smooth!(y::AbstractVector{Float64}, n::Int, bandwidth::Int, degree::Int, jump::Int,
                       use_weights::Bool, robustness_weights::AbstractVector{Float64},
                       ys::AbstractVector{Float64}, res::AbstractVector{Float64})

    if n < 2

        ys[firstindex(ys)] = y[1]
        return
    end

    step_size = min(jump, n - 1)

    left_bound = 1
    right_bound = min(bandwidth, n)

    if bandwidth >= n
        left_bound = 1
        right_bound = n
        i = 1
        while i <= n
            eval_point = float(i)
            ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
            if ok
                ys[firstindex(ys) - 1 + i] = ysi
            else
                ys[firstindex(ys) - 1 + i] = y[i]
            end
            i += step_size
        end
    else
        if step_size == 1
            half_bandwidth = (bandwidth + 1) ÷ 2
            left_bound = 1
            right_bound = bandwidth
            for i in 1:n
                if (i > half_bandwidth) && (right_bound != n)
                    left_bound += 1
                    right_bound += 1
                end
                eval_point = float(i)
                ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
                if ok
                    ys[firstindex(ys) - 1 + i] = ysi
                else
                    ys[firstindex(ys) - 1 + i] = y[i]
                end
            end
        else
            half_bandwidth = (bandwidth + 1) ÷ 2
            i = 1
            while i <= n
                if i < half_bandwidth
                    left_bound = 1
                    right_bound = bandwidth
                elseif i >= n - half_bandwidth + 1
                    left_bound = n - bandwidth + 1
                    right_bound = n
                else
                    left_bound = i - half_bandwidth + 1
                    right_bound = bandwidth + i - half_bandwidth
                end
                eval_point = float(i)
                ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
                if ok
                    ys[firstindex(ys) - 1 + i] = ysi
                else
                    ys[firstindex(ys) - 1 + i] = y[i]
                end
                i += step_size
            end
        end
    end

    if step_size != 1
        i = 1
        while i <= n - step_size
            ysi = ys[firstindex(ys) - 1 + i]
            ysj = ys[firstindex(ys) - 1 + i + step_size]
            interp_slope = (ysj - ysi) / float(step_size)
            for j in (i + 1):(i + step_size - 1)
                ys[firstindex(ys) - 1 + j] = ysi + interp_slope * float(j - i)
            end
            i += step_size
        end

        k = ((n - 1) ÷ step_size) * step_size + 1
        if k != n

            eval_point = float(n)
            ysn, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
            if ok
                ys[firstindex(ys) - 1 + n] = ysn
            else
                ys[firstindex(ys) - 1 + n] = y[n]
            end
            if k != n - 1

                val_at_k = ys[firstindex(ys) - 1 + k]
                val_at_n = ys[firstindex(ys) - 1 + n]
                interp_slope = (val_at_n - val_at_k) / float(n - k)
                for j in (k + 1):(n - 1)
                    ys[firstindex(ys) - 1 + j] = val_at_k + interp_slope * float(j - k)
                end
            end
        end
    end
    return
end
