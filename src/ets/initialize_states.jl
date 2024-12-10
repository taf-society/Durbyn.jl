function handle_seasonality(y, m, seasontype, n)
    if n < 4
        throw(ArgumentError("y is too short!"))
    end

    if n < 3 * m
        fouriery = fourier(x=y, period=m, K=1)
        fit = tslm_ets(y, fouriery)
        trend = 1:n
        seasonal = if seasontype == "A"
            y .- coef(fit)[1] .- coef(fit)[2] .* trend
        else
            y ./ (coef(fit)[1] .+ coef(fit)[2] .* trend)
        end

        return Dict(:seasonal => seasonal)
    else
        decompose_type = ifelse(seasontype == "A", "additive", "multiplicative")
        seasonal = decompose(x=y, m=m, type=decompose_type)
        return Dict(:seasonal => seasonal.seasonal)
    end
end

function tslm_ets(y, fouriery)
    trend = 1:length(y)
    df = DataFrame(y=y, trend=trend, S1_12=fouriery[:, 1], C1_12=fouriery[:, 2])
    fit = lm(@formula(y ~ trend + S1_12 + C1_12), df)
    return fit
end

function initialize_seasonal_components(y_d, m, seasontype)
    seasonal = y_d[:seasonal]
    init_seas = reverse(seasonal[2:m])
    if seasontype != "A"
        init_seas = [max(val, 0.01) for val in init_seas]
        if sum(init_seas) > m
            factor = sum(init_seas) + 0.01
            init_seas = [val / factor for val in init_seas]
        end
    end

    return init_seas
end

function adjust_y_sa(y, y_d, seasontype)
    seasonal = y_d[:seasonal]
    if seasontype == "A"
        return y .- seasonal
    else
        return y ./ max.(seasonal, 0.01)
    end
end

function lsfit_ets(x::Matrix{Float64}, y::Vector{Float64})
    # Create a DataFrame with y and x
    col_names = [:y; Symbol.("x$i" for i in 1:size(x, 2))]
    df = DataFrame(hcat(y, x), col_names)

    # Construct the formula y ~ x1 + x2 + ...
    formula = Term(:y) ~ sum(Term(Symbol("x$i")) for i in 1:size(x, 2))

    # Fit the linear model
    fit = lm(formula, df)
    return fit
end

function calculate_initial_values(y_sa, trendtype, maxn)
    if trendtype == "N"
        l0 = mean(y_sa[1:maxn])
        b0 = nothing
    else
        x = reshape(collect(1.0:maxn), maxn, 1) # Ensure x is a matrix of Float64
        fit = lsfit_ets(x, y_sa[1:maxn])
        if trendtype == "A"
            l0 = coef(fit)[1]
            b0 = coef(fit)[2]

            if abs(l0 + b0) < 1e-8
                l0 *= (1 + 0.001)
                b0 *= (1 - 0.001)
            end
        else
            l0 = coef(fit)[1] + coef(fit)[2]

            if abs(l0) < 0.00000001
                l0 = 0.0000001
            end
            b0 = (coef(fit)[1] + 2 * coef(fit)[2]) / l0
            l0 = l0 / b0

            if abs(b0) > 1e10
                b0 = sign(b0) * 1e10
            end

            if l0 < 1e-8 || b0 < 1e-8
                l0 = max(y_sa[1], 0.001)
                if isapprox(y_sa[1], 0.0, atol=1e-8)
                    denominator = y_sa[1] + 1e-10
                else
                    denominator = y_sa[1]
                end
                b0 = max(y_sa[2] / denominator, 0.001)
            end
        end
    end
    return Dict(:l0 => l0, :b0 => b0)
end

function initialize_states(y, m, trendtype, seasontype)
    if seasontype != "N"
        n = length(y)
        y_d = handle_seasonality(y, m, seasontype, n)
        init_seas = initialize_seasonal_components(y_d, m, seasontype)
        y_sa = adjust_y_sa(y, y_d, seasontype)
    else
        m = 1
        init_seas = nothing
        y_sa = y
    end

    maxn = min(max(10, 2 * m), length(y_sa))
    initial_values = calculate_initial_values(y_sa, trendtype, maxn)

    l0 = initial_values[:l0]
    b0 = initial_values[:b0]
    out = vcat([l0, b0], init_seas)
    out = [x for x in out if !isnothing(x)]
    return out
end