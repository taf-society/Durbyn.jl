module Naive
# Standard libs
import Statistics: std, mean
import Random: rand

import Distributions: quantile, TDist, Normal

# Internal modules
import ..Utils: mean2
import ..Stats: box_cox_lambda, box_cox, inv_box_cox
import ..Generics: Forecast, forecast, plot

include("meanf.jl")
export MeanFit, meanf

include("naive_fit.jl")
include("naive_forecast.jl")
export NaiveFit, naive, snaive, rw, rwf

end
