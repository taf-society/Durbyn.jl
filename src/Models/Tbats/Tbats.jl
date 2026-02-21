module Tbats

import ..Utils: is_constant
import ..Stats: box_cox, box_cox!, inv_box_cox, box_cox_lambda, longest_contiguous
import ..Arima: auto_arima
import ..Generics: Forecast, forecast, fitted
import ..Bats: bats, BATSModel
import ..Optimize: optimize
import ..Grammar: tbats

using LinearAlgebra: I, eigvals, dot, mul!
using Statistics: mean
using Distributions: Normal, quantile
using Polynomials: Polynomial, roots
using ..Grammar: ModelFormula, TbatsTerm, _extract_single_term
using Tables

export tbats, TBATSModel

include("tbats_base.jl")
include("tbats_formula_interface.jl")

end
