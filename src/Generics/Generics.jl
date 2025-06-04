module Generics

using Plots
include("forecast.jl")
include("plot.jl")

export Forecast, forecast
export plot
end