function inverse_parameter_transform(phi::AbstractArray, p::Int)
    if p > 100
        throw(ArgumentError("The function can only transform 100 parameters in arima0"))
    end
    new = copy(phi[1:p])
    work = copy(new)
    for j = p:-1:2
        a = new[j]
        for i = 1:(j-1)
            work[i] = (new[i] + a * new[j-i]) / (1 - a^2)
        end
        for i = 1:(j-1)
            new[i] = work[i]
        end
    end
    for i = 1:p
        new[i] = atanh(new[i])
    end
    return new
end

function arima_inverse_transform(x::AbstractArray, arma::Vector{Int})
    mp, mq, msp = arma[1:3]
    y = copy(x)
    if mp > 0
        y[1:mp] = inverse_parameter_transform(x[1:mp], mp)
    end
    v = mp + mq
    if msp > 0
        y[v+1:v+msp] = inverse_parameter_transform(x[v+1:v+msp], msp)
    end
    return y
end