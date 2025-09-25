module Ararma
using Statistics
using Distributions
using LinearAlgebra
import Base: show
import ..Generics: Forecast
import ..Generics: forecast, fitted, residuals
import ..Optimize: NelderMeadOptions, nmmin

include("setup_params.jl")
include("arar.jl")
include("ararma.jl")

export ARAR, ArarmaModel, arar, ararma, auto_ararma

end