module Generics

using Plots
import Base: show, summary
include("forecast.jl")
include("plot.jl")
include("generics.jl")

export Forecast, forecast
export plot, fitted, residuals, summary, predict
export coef, coefficients, coefs
end