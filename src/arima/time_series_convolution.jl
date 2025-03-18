function time_series_convolution(a::AbstractArray, b::AbstractArray)
    na = length(a)
    nb = length(b)
    nab = na + nb - 1
    ab = zeros(Float64, nab)
    
    for i in 1:na
        for j in 1:nb
            ab[i + j - 1] += a[i] * b[j]
        end
    end
    return ab
end