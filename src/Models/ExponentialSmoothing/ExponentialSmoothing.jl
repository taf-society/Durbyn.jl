module ExponentialSmoothing
# Standard libs
using LinearAlgebra
import Statistics: mean

# External packages
using Plots
using Polynomials
using Distributions
import DataStructures: OrderedDict

# Internal modules
import ..Utils: is_constant, match_arg, na_action, na_omit
import ..Stats: box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier
import ..Generics: Forecast, forecast, plot
import ..Optimize: nmmin, NelderMeadOptions

include("ets_utils.jl")
include("ets.jl")
include("holt.jl")
include("holt_winters.jl")
include("ses.jl")
include("croston.jl")
include("forecast.jl")

export ets, holt, holt_winters, ses, croston

end
