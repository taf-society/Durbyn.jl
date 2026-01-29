"""
    Diffusion Model Types

Type definitions for diffusion curve forecasting models.
"""

"""
    DiffusionModelType

Enum representing the four diffusion model types:
- `Bass`: Bass diffusion model (innovation/imitation)
- `Gompertz`: Gompertz growth curve
- `GSGompertz`: Gamma/Shifted Gompertz model
- `Weibull`: Weibull distribution-based model
"""
@enum DiffusionModelType begin
    Bass
    Gompertz
    GSGompertz
    Weibull
end

"""
    DiffusionFit

Struct holding a fitted diffusion model.

# Fields
- `model_type::DiffusionModelType`: Type of diffusion model (Bass, Gompertz, GSGompertz, Weibull)
- `params::NamedTuple`: Model parameters - varies by model type:
  - Bass: `(m, p, q)` - market potential, innovation coefficient, imitation coefficient
  - Gompertz: `(m, a, b)` - market potential, displacement, growth rate
  - GSGompertz: `(m, a, b, c)` - market potential, shape, scale, shift
  - Weibull: `(m, a, b)` - market potential, scale, shape
- `fitted::Vector{Float64}`: Fitted adoption per period
- `cumulative::Vector{Float64}`: Fitted cumulative adoption
- `residuals::Vector{Float64}`: In-sample residuals (actual - fitted)
- `mse::Float64`: Mean squared error
- `y::Vector{Float64}`: Original adoption data (per period)
- `init_params::NamedTuple`: Initial parameters before optimization
- `loss::Int`: Loss function power (1=MAE, 2=MSE)
- `optim_cumulative::Bool`: Whether optimization was on cumulative values
"""
struct DiffusionFit
    model_type::DiffusionModelType
    params::NamedTuple
    fitted::Vector{Float64}
    cumulative::Vector{Float64}
    residuals::Vector{Float64}
    mse::Float64
    y::Vector{Float64}
    init_params::NamedTuple
    loss::Int
    optim_cumulative::Bool
end

function Base.show(io::IO, fit::DiffusionFit)
    println(io, "Diffusion Model ($(fit.model_type))")
    println(io, "─────────────────────────────")
    println(io, "Observations: ", length(fit.y))
    println(io, "Parameters:")
    for (k, v) in pairs(fit.params)
        println(io, "  $(k): ", round(v, digits=6))
    end
    println(io, "MSE: ", round(fit.mse, digits=6))
    println(io, "Loss function: L$(fit.loss)")
    println(io, "Optimized on: ", fit.optim_cumulative ? "cumulative" : "adoption")
end
