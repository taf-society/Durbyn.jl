module Arar
using Statistics
using Distributions
import ..Generics: Forecast

include("arar.jl")

export ARAR, arar, forecast, fitted, residuals

end