module Stats

import Statistics: mean
import ..Utils: na_omit, as_integer, mean2

include("box_cox.jl")
include("decompose.jl")
include("diff.jl")
include("fourier.jl")

export box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier

end
