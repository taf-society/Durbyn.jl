module Arima
export arima, ArimaFit, PDQ, predict_arima, ArimaPredictions
export arima_rjh, auto_arima

using LinearAlgebra
import Statistics: mean
import LinearAlgebra: rank

using Distributions
import Tables

using ..Stats
using ..Grammar
import ..Utils: is_constant, _check_arg, dropmissing, NamedMatrix, align_columns, ismissingish
import ..Stats: handle_missing
import ..Utils: is_constant_all, drop_constant_columns, is_rank_deficient, row_sums
import ..Utils: cbind, add_drift_term, setrow!, get_elements, select_rows, as_vector, as_integer
import ..Utils: mean2
import ..Stats: box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier
import ..Generics: Forecast, forecast, plot, fitted, residuals
import ..Optimize: optimize, numerical_hessian
import ..Grammar: ModelFormula, ArimaOrderTerm, VarTerm, compile_arima_formula

import Base: show
include("core/types.jl")
include("core/equations.jl")
include("core/covariance.jl")
include("core/hyperparameters.jl")
include("core/order.jl")
include("core/kalman.jl")
include("core/system.jl")
include("core/compat.jl")
include("core/fit.jl")
include("core/arima_rjh.jl")
include("auto/auto_arima_utils.jl")
include("auto/auto_arima.jl")
include("core/formula_interface.jl")
include("core/simulate.jl")
include("core/forecast.jl")
include("core/show.jl")

end
