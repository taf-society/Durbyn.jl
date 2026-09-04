module Bats

import ..Utils: is_constant, _normalize_levels
import ..Stats: box_cox, box_cox!, inv_box_cox, box_cox_lambda, longest_contiguous
import ..Arima: auto_arima
import ..Generics: Forecast, forecast, fitted, residuals
import ..Optimize: optimize
import ..Grammar: bats

using LinearAlgebra: I, eigvals, dot, mul!
using Statistics: mean
using Distributions: Normal, quantile
using Polynomials: Polynomial, roots
using ..Grammar: ModelFormula, BatsTerm, _extract_single_term
using Tables

export bats, BATSModel

include("types.jl")
include("matrices.jl")
include("recursion.jl")
include("admissibility.jl")
include("likelihood.jl")
include("fitting.jl")
include("api.jl")
include("bats_formula_interface.jl")

end
