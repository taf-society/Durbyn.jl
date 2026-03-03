arma_vector(order::SARIMAOrder) = [order.p, order.q, order.P, order.Q, order.s, order.d, order.D]

function build_delta(order::SARIMAOrder)
    Delta = [1.0]

    for _ in 1:order.d
        Delta = time_series_convolution(Delta, [1.0, -1.0])
    end

    for _ in 1:order.D
        seasonal_filter = [1.0; zeros(order.s - 1); -1.0]
        Delta = time_series_convolution(Delta, seasonal_filter)
    end

    return -Delta[2:end]
end
