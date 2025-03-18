function arima_param_transform!(p::Int, raw::AbstractVector, new::AbstractVector)
    if p > 100
        throw(ArgumentError("The function can only transform 100 parameters in arima0"))
    end

    new[1:p] .= tanh.(raw[1:p])
    work = copy(new[1:p])
    
    for j in 2:p
        a = new[j]
        for k in 1:(j-1)
            work[k] -= a * new[j - k]
        end
        new[1:(j-1)] .= work[1:(j-1)]
    end
end

function arima_gradient_transform(x::AbstractArray, arma::AbstractArray)
    eps = 1e-3
    mp, mq, msp = arma[1:3]
    n = length(x)
    y = Matrix{Float64}(I, n, n)
    
    w1 = Vector{Float64}(undef, 100)
    w2 = Vector{Float64}(undef, 100)
    w3 = Vector{Float64}(undef, 100)
    
    if mp > 0
        
        for i in 1:mp
            w1[i] = x[i]
        end
        
        arima_param_transform!(mp, w1, w2)
        
        for i in 1:mp
            w1[i] += eps
            arima_param_transform!(mp, w1, w3)
            for j in 1:mp
                y[i, j] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end

    if msp > 0
        v = mp + mq
        for i in 1:msp
            w1[i] = x[i + v]
        end
        arima_param_transform!(msp, w1, w2)
        for i in 1:msp
            w1[i] += eps
            arima_param_transform!(msp, w1, w3)
            for j in 1:msp
                y[i + v, j + v] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end
    return y
end

function arima_undo_params(x::AbstractArray, arma::AbstractArray)
    mp, mq, msp = arma[1:3]
    res = copy(x)
    if mp > 0
        arima_param_transform!(mp, x, res)
    end
    v = mp + mq
    if msp > 0
        arima_param_transform!(msp, @view(x[v+1:end]), @view(res[v+1:end]))
    end
    return res
end
