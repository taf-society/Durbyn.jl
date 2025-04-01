using LinearAlgebra
using GPUArrays 
using Adapt

function compute_phi(y::AbstractArray, max_iter::Int=15)
    phi = [dot(y[i+1:end], y[1:end-i]) / sum(y[1:end-i].^2) for i in 1:max_iter]
    return phi
end

# GPU Forecasting/ Hartware agnostic statistical forecasting
@kernel function phi_kernel(y, phi, n, max_iter)
    i = @index(Global)
    if i <= max_iter
        acc_dot = zero(eltype(y))
        acc_norm = zero(eltype(y))
        for j = 1:(n - i)
            acc_dot += y[i + j] * y[j]
            acc_norm += y[j]^2
        end
        phi[i] = acc_dot / acc_norm
    end
end

function compute_phi(y::AbstractGPUArray{T}, max_iter::Int=15) where {T<:AbstractFloat}
    n = length(y)
    phi = similar(y, max_iter)
    backend = get_backend(y)

    kernel_inst = phi_kernel(backend)
    kernel_inst(y, phi, n, max_iter; ndrange=(max_iter,))

    return phi
end


using CUDA

y = CUDA.CuArray(y)
phi = compute_phi(y, 15)

