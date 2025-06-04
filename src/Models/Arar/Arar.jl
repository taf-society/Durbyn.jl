module Arar
using Statistics
using Distributions
import Base: show
import ..Generics: Forecast
import ..Generics: forecast

include("arar.jl")

export ARAR, arar, fitted, residuals

end