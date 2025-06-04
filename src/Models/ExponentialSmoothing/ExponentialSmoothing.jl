module ExponentialSmoothing

import ..Utils: is_constant, match_arg, na_action, na_omit
import ..Optim: nmmin
import ..Stats: box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier
import DataStructures: OrderedDict
using Polynomials
using LinearAlgebra
using Distributions
using Plots
import Statistics: mean
import ..Generics: Forecast, forecast
import ..Generics: plot

include("ets_utils.jl")
include("ets.jl")
include("holt.jl")
include("holt_winters.jl")
include("ses.jl")
include("croston.jl")
include("forecast.jl")

export ets, holt, holt_winters, ses, croston

end
