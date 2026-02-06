module Bats

import ..Utils: is_constant
import ..Stats: box_cox, box_cox!, inv_box_cox, box_cox_lambda, na_contiguous
import ..Arima: auto_arima
import ..Generics: Forecast, forecast, fitted
import ..Optimize: optim
import ..Grammar: bats

using LinearAlgebra: I, eigvals, dot, mul!
using Statistics: mean
using Distributions: Normal, quantile
using Polynomials: Polynomial, roots
using ..Grammar: ModelFormula, BatsTerm, _extract_single_term
using Tables

export bats, BATSModel

include("bats_base.jl")
include("bats_formula_interface.jl")

end