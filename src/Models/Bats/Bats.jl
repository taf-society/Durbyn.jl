module Bats

import ..Utils: na_contiguous, is_constant
import ..Stats: box_cox, inv_box_cox, box_cox_lambda
import ..Arima: auto_arima
import ..Generics: Forecast, forecast, fitted
import ..Optimize: optim

using LinearAlgebra: I, eigvals, dot
using Statistics: mean
using Distributions: Normal, quantile
using Polynomials: Polynomial, roots

export bats, BATSModel

include("bats_base.jl")

end