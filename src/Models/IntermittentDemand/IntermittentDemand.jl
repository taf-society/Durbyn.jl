module IntermittentDemand
using Statistics
using Plots
import Base: show
import Statistics: mean

using ..Optimize
using ..Generics
import ..Utils: evaluation_metrics
import ..Utils: match_arg
import ..Generics: plot
import ..Generics: forecast, fitted, residuals

include("crost_utils.jl")
include("crost.jl")

export croston_classic, croston_sba, croston_sbj
export IntermittentDemandForecast
export IntermittentDemandCrostonFit

end