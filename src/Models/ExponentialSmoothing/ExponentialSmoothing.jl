module ExponentialSmoothing
# Standard libs
using LinearAlgebra
import Statistics: mean

# External packages
using Polynomials
using Distributions

# Internal modules
import ..Utils: is_constant, match_arg, na_omit, check_component
import ..Stats: box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier, na_action
import ..Generics: Forecast, forecast, plot, fitted
import ..Optimize: nmmin, NelderMeadOptions, scaler, descaler

include("ets_utils.jl")
include("ets.jl")
include("holt.jl")
include("holt_winters.jl")
include("ses.jl")
include("croston.jl")
include("forecast.jl")
include("show.jl")

export ets, holt, holt_winters, ses, croston, CrostonForecast, CrostonFit

end
