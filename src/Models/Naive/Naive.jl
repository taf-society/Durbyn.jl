module Naive
# Standard libs
import Statistics: std

import Distributions: quantile, TDist

# Internal modules
import ..Utils: mean2
import ..Stats: box_cox_lambda, box_cox, inv_box_cox
import ..Generics: Forecast, forecast, plot

include("meanf.jl")
export MeanFit, meanf

end
