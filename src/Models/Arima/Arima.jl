module Arima
export arima, ArimaFit, PDQ, predict_arima, ArimaPredictions
export ArimaRJHFit, arima_rjh, auto_arima
# Standard libs
using LinearAlgebra
import Statistics: mean
import LinearAlgebra: rank

# External packages
using Plots
using Polynomials
using Distributions
import DataStructures: OrderedDict

# Internal modules
using ..Stats
import ..Utils: is_constant, match_arg, na_action, na_omit, NamedMatrix, align_columns
import ..Utils: cbind, add_drift_term, na_omit_pair, setrow!, get_elements, as_vector, as_integer
import ..Stats: box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier
import ..Generics: Forecast, forecast, plot, fitted, residuals
import ..Optimize: nmmin, NelderMeadOptions, optim_hessian, scaler, descaler

import Base: show
include("arima.jl")
include("arima_rjh.jl")
include("auto_arima_utils.jl")
include("auto_arima.jl")

end