module Optimize

import ..Utils: _check_arg
import LinearAlgebra

include("nelder_mead.jl")
include("bfgs.jl")
include("lbfgsb.jl")
include("brent.jl")
include("itp.jl")
include("numerical_hessian.jl")
include("scalers.jl")
include("optimize_base.jl")


export nelder_mead, bfgs, lbfgsb, brent, itp, numerical_hessian
export NelderMeadOptions, BFGSOptions, LBFGSBOptions, BrentOptions, ITPOptions
export OptimizeResult
export scaler, descaler
export optimize

end
