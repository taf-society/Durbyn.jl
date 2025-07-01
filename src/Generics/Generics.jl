module Generics

using Plots
include("forecast.jl")
include("plot.jl")
include("generics.jl")

export Forecast, forecast
export plot, fitted, residuals
end