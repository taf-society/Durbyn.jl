module IntermittentDemand
using Statistics
using Plots
using ..Generics
import Statistics: mean
import Optim: optimize, NelderMead, Brent, Options
import ..Utils: evaluation_metrics
import ..Utils: match_arg
import ..Generics: plot
import ..Generics: forecast

include("crost_utils.jl")
include("crost.jl")

export croston_classic, croston_sba, croston_sbj, IntermittentDemandForecast, IntermittentDemandCrostonFit, residuals, fitted

end