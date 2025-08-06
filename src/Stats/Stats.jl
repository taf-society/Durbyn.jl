module Stats
import Base: show
import Statistics: mean, std, quantile, minimum, maximum
using Plots
import LinearAlgebra: diag

import ..Utils: na_omit, as_integer, mean2
import ..Generics: plot, summary, fitted, residuals, predict

include("box_cox.jl")
include("decompose.jl")
include("diff.jl")
include("fourier.jl")
include("stl.jl")
include("ols.jl")

export box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, 
fourier, STLResult, stl, ols, OlsFit, coef, coefficients, coefs

end
