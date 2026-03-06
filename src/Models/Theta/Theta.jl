module Theta

export theta, auto_theta, ThetaFit, ThetaModelType
export STM, OTM, DSTM, DOTM

using Statistics
using Random: MersenneTwister, randn
using Tables

import ..Utils: is_constant
import ..Optimize: optimize
import ..Stats: acf, decompose
import ..Generics: forecast, Forecast
import ..Grammar: theta
using ..Grammar: ModelFormula, ThetaTerm, _extract_single_term

include("core/types.jl")
include("core/system.jl")
include("core/fit.jl")
include("core/formula_interface.jl")
include("core/forecast.jl")

end
