module Arima
export arima, ArimaFit, PDQ, predict_arima, ArimaPredictions
export ArimaRJHFit, arima_rjh, auto_arima
# Standard libs
using LinearAlgebra
import Statistics: mean
import LinearAlgebra: rank

# External packages
using Polynomials
using Distributions
import Tables

# Internal modules
using ..Stats
using ..Grammar
import ..Utils: is_constant, match_arg, na_omit, NamedMatrix, align_columns, isna
import ..Stats: na_action
import ..Utils: is_constant_all, drop_constant_columns, is_rank_deficient, row_sums
import ..Utils: cbind, add_drift_term, na_omit_pair, setrow!, get_elements, select_rows, as_vector, as_integer
import ..Utils: mean2
import ..Stats: box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier
import ..Generics: Forecast, forecast, plot, fitted, residuals
import ..Optimize: optim, optim_hessian
import ..Grammar: ModelFormula, ArimaOrderTerm, VarTerm, compile_arima_formula

import Base: show
include("arima_base.jl")
include("arima_rjh.jl")
include("auto_arima_utils.jl")
include("auto_arima.jl")
include("arima_formula_interface.jl")
include("forecast.jl")

end
