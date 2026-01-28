"""
    Grammar

Durbyn's forecasting grammar module provides a Domain-Specific Language (DSL) for
declaratively specifying forecasting models.

# Exports

## ARIMA Order Functions
- `p`, `q`, `P`, `Q` - AR and MA order specifications
- `d`, `D` - Differencing order specifications

## Core Types
- `AbstractTerm` - Base type for all grammar terms
- `ArimaOrderTerm` - ARIMA model order terms
- `VarTerm` - Exogenous variable terms
- `ModelFormula` - Complete formula specification

## Utilities
- `compile_arima_formula` - Compile formula to parameter dict

# Examples
```julia
using Durbyn
using Durbyn.Grammar

# Create data
data = (
    sales = randn(120),
    temperature = randn(120),
    promotion = rand(0:1, 120)
)

# ARIMA formula
formula = :sales = p(1,2) + d(1) + q(2,3)

# SARIMAX formula with exogenous variables
formula = :sales = p(1,2) + d(1) + q(2,3) + P(0,1) + D(1) + Q(0,1) + :temperature + :promotion

# Fit model
fit = auto_arima(formula, data, 12)
```
"""
module Grammar

export p, q, d, P, Q, D, auto
export e, t, s, drift, ses, holt, hw, holt_winters, croston, arar, bats, tbats, theta
export naive_term, snaive_term, rw_term
export AbstractTerm, ArimaOrderTerm, VarTerm, AutoVarTerm, ModelFormula, ArarTerm, BatsTerm, TbatsTerm, ThetaTerm
export NaiveTerm, SnaiveTerm, RwTerm
export EtsComponentTerm, EtsDriftTerm, SesTerm, HoltTerm, HoltWintersTerm, CrostonTerm
export compile_arima_formula, compile_ets_formula
export @formula

include("grammar_base.jl")

end
