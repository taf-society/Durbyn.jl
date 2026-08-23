module ExponentialSmoothing
# Standard libs
using LinearAlgebra
import Statistics: mean

# External packages
using Polynomials
using Distributions

# Internal modules
import ..Utils: is_constant, _check_arg, dropmissing, check_component
import ..Stats: box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier
import ..Stats: handle_missing, MissingMethod, Contiguous, Interpolate, FailMissing
import ..Generics: Forecast, forecast, plot, fitted, residuals
export forecast, fitted, residuals
import ..Optimize: nelder_mead, NelderMeadOptions, scaler, descaler

include("types.jl")
include("parameters.jl")
include("recursion.jl")
include("initialization.jl")
include("evaluation.jl")
include("hw_conventional.jl")
include("selection.jl")
include("ets.jl")
include("holt.jl")
include("holt_winters.jl")
include("ses.jl")
include("croston.jl")
include("forecast.jl")
include("show.jl")

export ets, holt, holt_winters, ses, croston, CrostonForecast, CrostonFit

end
