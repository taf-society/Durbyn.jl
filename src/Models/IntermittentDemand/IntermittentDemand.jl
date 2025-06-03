module IntermittentDemand
using Statistics
using Plots
import Statistics: mean
import Optim: optimize, NelderMead, Brent, Options
import ..Utils: evaluation_metrics
import ..Utils: match_arg

include("crost_utils.jl")
include("crost.jl")

export plot
export croston_classic, croston_sba, croston_sbj, IntermittentDemandForecast, plot

end