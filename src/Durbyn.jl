module Durbyn

include("Utils/Utils.jl")

include("Optimize/Optimize.jl")
include("Stats/Stats.jl")
include("Generics/Generics.jl")

include("Models/Naive/Naive.jl")
include("Models/ExponentialSmoothing/ExponentialSmoothing.jl")
include("Models/Arima/Arima.jl")
include("Models/IntermittentDemand/IntermittentDemand.jl")
include("Models/Arar/Arar.jl")


using .Utils
using .Generics  
using .ExponentialSmoothing
using .Optimize
using .Stats
using .Naive
using .Arima
using .Arar
using .IntermittentDemand

import .Utils: air_passengers
import .Generics: plot
import .Optimize: NelderMeadOptions

export air_passengers
export plot, forecast
export NelderMeadOptions

end
