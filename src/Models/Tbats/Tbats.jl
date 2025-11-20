module Tbats

import ..Utils: na_contiguous, is_constant
import ..Stats: box_cox, inv_box_cox, box_cox_lambda
import ..Arima: auto_arima
import ..Generics: Forecast, forecast, fitted
import ..Bats: bats, BATSModel

using LinearAlgebra: I, eigvals, dot
using Statistics: mean
using Distributions: Normal, quantile
using Polynomials: Polynomial, roots
using Optim: optimize, NelderMead, BFGS, Optim

export tbats, TBATSModel

include("tbats_base.jl")

end
