"""
    Diffusion

Implementation of diffusion curve forecasting models for technology adoption
and market penetration analysis.

# Model Types
- `Bass`: Bass diffusion model - captures innovation and imitation effects
- `Gompertz`: Gompertz growth curve - asymmetric S-curve
- `GSGompertz`: Gamma/Shifted Gompertz - generalization with shape parameter
- `Weibull`: Weibull distribution-based model

# References
- Bass, F.M. (1969). A new product growth for model consumer durables.
  Management Science, 15(5), 215-227.
- Gompertz, B. (1825). On the nature of the function expressive of the law
  of human mortality. Philosophical Transactions, 115, 513-583.
- Bemmaor, A.C. (1994). Modeling the diffusion of new durable goods: Word-of-mouth
  effect versus consumer heterogeneity. In G. Laurent et al. (Eds.), Research
  Traditions in Marketing. Kluwer Academic Publishers.
- Sharif, M.N. & Islam, M.N. (1980). The Weibull distribution as a general model
  for forecasting technological change. Technological Forecasting and Social Change,
  18(3), 247-256.

# Example
```julia
using Durbyn

y = [5, 10, 25, 45, 70, 85, 75, 50, 30, 15]

fit = diffusion(y)
fit = diffusion(y, model_type=Bass)

fit = diffusion(y, model_type=Gompertz)

fc = forecast(fit, h=5)

fc.mean
fc.lower[1]
fc.upper[2]

fit.params.m
fit.params.p
fit.params.q
```
"""
module Diffusion

using Statistics
using LinearAlgebra

import ..Optimize: optimize
import ..Generics: forecast, Forecast, predict

export diffusion, fit_diffusion, DiffusionFit
export DiffusionModelType, Bass, Gompertz, GSGompertz, Weibull
export bass_curve, gompertz_curve, gsgompertz_curve, weibull_curve
export bass_init, gompertz_init, gsgompertz_init, weibull_init

include("types.jl")
include("curves.jl")
include("initialization.jl")
include("fitting.jl")
include("forecast.jl")

end
