module Ararma
using Statistics
using Distributions
using LinearAlgebra
import Base: show
import ..Generics: Forecast
import ..Generics: forecast, fitted, residuals
import ..Optimize: NelderMeadOptions, nmmin, scaler, descaler
import ..Grammar: arar
using ..Grammar: ModelFormula, ArarTerm, ArimaOrderTerm, _extract_single_term
using Tables

include("setup_params.jl")
include("arar.jl")
include("ararma.jl")
include("arar_formula_interface.jl")
include("ararma_formula_interface.jl")

export ARAR, ArarmaModel, arar, ararma, auto_ararma

end