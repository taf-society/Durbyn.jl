module Stats
import Base: show, summary
import Statistics: mean, std, quantile, minimum, maximum
using Plots
import LinearAlgebra: diag, dot

import ..Utils: na_omit, as_integer, mean2, is_constant, match_arg, NamedMatrix
import ..Generics: plot, summary, fitted, residuals, predict

include("box_cox.jl")
include("decompose.jl")
include("diff.jl")
include("fourier.jl")
include("stl.jl")
include("ols.jl")
include("UnitTests/utils.jl")
include("UnitTests/adf.jl")
include("UnitTests/kpss.jl")
include("UnitTests/phillips_perron.jl")
include("embed.jl")
include("UnitTests/ndiffs.jl")
include("UnitTests/ocsb.jl")
include("mstl.jl")

export box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, 
fourier, STLResult, stl, ols, OlsFit, coef, coefficients, coefs, adf, ADF, 
kpss, KPSS, phillips_perron, PhillipsPerron, embed, ndiffs, ocsb, mstl, MSTLResult

end
