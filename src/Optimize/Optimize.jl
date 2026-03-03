module Optimize

import ..Utils: _check_arg

include("numgrad.jl")
include("nelder_mead.jl")
include("line_search.jl")
include("bfgs.jl")
include("lbfgsb.jl")
include("brent.jl")
include("numerical_hessian.jl")
include("scalers.jl")
include("optimize_base.jl")


export nelder_mead, bfgs, lbfgsb, brent, numerical_hessian
export NelderMeadOptions, BFGSOptions, LBFGSBOptions, BrentOptions
export OptimizeResult
export scaler, descaler
export optimize
export numgrad, numgrad!, numgrad_with_cache!
export NumericalGradientCache
export bfgs_hessian_update!
export BFGSWorkspace

end
