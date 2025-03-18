function arima_param_transform!(p::Int, raw::AbstractVector, new::AbstractVector)
    if p > 100
        throw(ArgumentError("The function can only transform 100 parameters in arima0"))
    end

    new[1:p] .= tanh.(raw[1:p])
    work = copy(new[1:p])

    for j = 2:p
        a = new[j]
        for k = 1:(j-1)
            work[k] -= a * new[j-k]
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

        for i = 1:mp
            w1[i] = x[i]
        end

        arima_param_transform!(mp, w1, w2)

        for i = 1:mp
            w1[i] += eps
            arima_param_transform!(mp, w1, w3)
            for j = 1:mp
                y[i, j] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end

    if msp > 0
        v = mp + mq
        for i = 1:msp
            w1[i] = x[i+v]
        end
        arima_param_transform!(msp, w1, w2)
        for i = 1:msp
            w1[i] += eps
            arima_param_transform!(msp, w1, w3)
            for j = 1:msp
                y[i+v, j+v] = (w3[j] - w2[j]) / eps
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

function arima_transpar(
    params_in::Vector{Float64},
    arma::Vector{Int},
    trans::Bool;
    partrans = arima_param_transform!,
)
    mp, mq, msp, msq, ns = arma
    p = mp + ns * msp
    q = mq + ns * msq

    phi = zeros(Float64, p)
    theta = zeros(Float64, q)
    params = copy(params_in)

    if trans
        if mp > 0
            partrans(mp, params_in, params)
        end
        v = mp + mq
        if msp > 0
            partrans(msp, params_in[v+1:end], params[v+1:end])
        end
    end

    if ns > 0
        phi[1:mp] .= params[1:mp]
        theta[1:mq] .= params[mp+1:mp+mq]

        for j = 0:(msp-1)
            phi[(j+1)*ns] += params[mp+mq+j+1]
            for i = 0:(mp-1)
                phi[((j+1)*ns)+(i+1)] -= params[i+1] * params[mp+mq+j+1]
            end
        end

        for j = 0:(msq-1)
            theta[(j+1)*ns] += params[mp+mq+msp+j+1]
            for i = 0:(mq-1)
                theta[((j+1)*ns)+(i+1)] += params[mp+i+1] * params[mp+mq+msp+j+1]
            end
        end
    else
        phi[1:mp] .= params[1:mp]
        theta[1:mq] .= params[mp+1:mp+mq]
    end

    return (phi, theta)
end