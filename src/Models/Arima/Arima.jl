module Arima

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
import ..Generics: Forecast, forecast, plot, fitted, residuals
import ..Optimize: nmmin, NelderMeadOptions, optim_hessian

import Base: show
include("arima.jl")
export arima, ArimaFit, PDQ, ArimaCoef, predict_arima, ArimaPredictions

end