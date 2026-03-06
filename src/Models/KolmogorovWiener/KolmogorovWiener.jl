module KolmogorovWiener

using LinearAlgebra
using Distributions: Normal, quantile as dist_quantile
import ..Arima: auto_arima, ArimaFit
import ..Utils: _check_arg
import ..Generics: fitted, residuals, forecast, Forecast
import ..Stats: Decomposition, inv_box_cox
import Base: show

include("kw_types.jl")
include("autocovariance.jl")
include("quadrature.jl")
include("ideal_filters.jl")
include("toeplitz.jl")
include("propositions.jl")
include("kw_filter.jl")
include("decomposition.jl")
include("forecast.jl")
include("show.jl")

export kolmogorov_wiener, KWFilterResult, kw_decomposition

end
