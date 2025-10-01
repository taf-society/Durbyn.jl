module Generics

using Plots
import Base: show, summary
include("forecast.jl")
include("plot.jl")

# Generic function definitions
function fitted end
function residuals end
function predict end
function coef end
function coefficients end
function coefs end

export Forecast, forecast
export plot, fitted, residuals, summary, predict
export coef, coefficients, coefs
end