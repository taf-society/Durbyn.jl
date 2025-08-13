module Durbyn

include("Utils/Utils.jl")

include("Optimize/Optimize.jl")
include("Generics/Generics.jl")
include("Stats/Stats.jl")

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

import .Utils: air_passengers, NamedMatrix, get_elements, get_vector, align_columns, add_drift_term, cbind
import .Utils: Formula, parse_formula, compile, model_matrix, model_frame
import .Generics: plot, fitted, residuals, summary, predict, forecast
import .Optimize: NelderMeadOptions

export air_passengers, NamedMatrix, get_elements, get_vector, align_columns, add_dift_term, cbind
export Formula, parse_formula, compile, model_matrix, model_frame
export plot, fitted, residuals, summary, predict, forecast
export coef, coefficients, coefs
export NelderMeadOptions

end
