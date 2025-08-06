module Arima

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
import ..Utils: cbind, add_dift_term, na_omit_pair
import ..Stats: box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier
import ..Generics: Forecast, forecast, plot, fitted, residuals
import ..Optimize: nmmin, NelderMeadOptions, optim_hessian

import Base: show
include("arima.jl")
export arima, ArimaFit, PDQ, predict_arima, ArimaPredictions

end