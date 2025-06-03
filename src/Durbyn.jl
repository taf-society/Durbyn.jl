module Durbyn

include("Utils/Utils.jl")

include("Optim/Optim.jl")
include("Stats/Stats.jl")
include("Generics/Generics.jl")

include("Models/Naive/Naive.jl")
include("Models/ExponentialSmoothing/ExponentialSmoothing.jl")
include("Models/Arima/Arima.jl")
include("Models/IntermittentDemand/IntermittentDemand.jl")
include("Models/Arar/Arar.jl")


using .Utils  
using .ExponentialSmoothing
using .Optim
using .Stats
using .Naive
using .Arima
using .Arar
using .Generics 
using .IntermittentDemand

import .Utils: air_passengers
import .Generics: plot

export air_passengers
export plot

end
