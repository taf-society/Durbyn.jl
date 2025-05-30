module ExponentialSmoothing

import ..Utils: is_constant, match_arg, na_action, na_omit
import ..Optim: nmmin
import ..Stats: box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier
import DataStructures: OrderedDict
using Polynomials
using LinearAlgebra
using Distributions
import Statistics: mean
import ..Generics: Forecast

include("ets_utils.jl")
include("ets.jl")
include("forecast.jl")
include("holt.jl")
include("holt_winters.jl")
include("ses.jl")
include("croston.jl")

export ets, forecast, holt, holt_winters, ses, croston

end
