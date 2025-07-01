module Durbyn

include("Utils/Utils.jl")

include("Optimize/Optimize.jl")
include("Stats/Stats.jl")
include("Generics/Generics.jl")

include("Models/Naive/Naive.jl")
include("Models/ExponentialSmoothing/ExponentialSmoothing.jl")
include("Models/Arima/Arima.jl")
include("Models/IntermittentDemand/IntermittentDemand.jl")
include("Models/Ararma/Ararma.jl")


using .Utils
using .Generics  
using .ExponentialSmoothing
using .Optimize
using .Stats
using .Naive
using .Arima
using .Ararma
using .IntermittentDemand

import .Utils: air_passengers
import .Generics: plot
import .Optimize: NelderMeadOptions

export air_passengers
export plot, forecast, fitted, residuals
export NelderMeadOptions

end
