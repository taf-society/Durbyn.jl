module IntermittentDemand
using Statistics
using Plots
import Base: show
import Statistics: mean
import Optim: optimize, NelderMead, Brent, Options

using ..Generics
import ..Utils: evaluation_metrics
import ..Utils: match_arg
import ..Generics: plot
import ..Generics: forecast

include("crost_utils.jl")
include("crost.jl")

export croston_classic, croston_sba, croston_sbj
export IntermittentDemandForecast
export IntermittentDemandCrostonFit, residuals, fitted

end