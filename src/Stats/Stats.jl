module Stats
import Base: show, summary
import Statistics: mean, std, quantile, minimum, maximum, var, median
import LinearAlgebra: diag, dot, qr

using ..Optimize
import ..Utils: na_omit, as_integer, mean2, is_constant, match_arg, NamedMatrix, isna, duplicated, complete_cases
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
include("seasonal_strength.jl")
include("UnitTests/nsdiffs.jl")
include("approx.jl")
include("acf.jl")
include("na_interp.jl")

export box_cox_lambda, box_cox, box_cox!, inv_box_cox, decompose, DecomposedTimeSeries, diff,
fourier, STLResult, stl, ols, OlsFit, coef, coefficients, coefs, adf, ADF,
kpss, KPSS, phillips_perron, PhillipsPerron, embed, ndiffs, ocsb, mstl, MSTLResult,
seasonal_strength, nsdiffs, approx, approxfun, acf, pacf, ACFResult, PACFResult,
na_interp, na_contiguous, na_fail, na_action
end
