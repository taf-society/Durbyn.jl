"""
    ModelSpecs

Model specification module for Durbyn's fable-like forecasting interface.

Provides abstract types and concrete specifications for defining forecasting
models in a declarative way, separate from data fitting.

# Exports

## Abstract Types
- `AbstractModelSpec` - Base type for model specifications
- `AbstractFittedModel` - Base type for fitted models

## Concrete Types
- `ArimaSpec` - ARIMA model specification
- `FittedArima` - Fitted ARIMA model
- `ModelCollection` - Collection of model specs
- `FittedModelCollection` - Collection of fitted models

## Functions
- `model` - Create model specification(s)
- `extract_metrics` - Extract fit quality metrics

# Examples
```julia
using Durbyn
using Durbyn.ModelSpecs

# Create specification
spec = ArimaSpec(@formula(sales = p() + q()))

# Or multiple for comparison
models = model(
    ArimaSpec(@formula(sales = p() + q())),
    ArimaSpec(@formula(sales = p(1) + d(1) + q(1))),
    names = ["auto", "fixed"]
)

# Fit to data (requires fit() implementation)
fitted = fit(spec, data, m = 12)

# Forecast (requires forecast() implementation)
fc = forecast(fitted, h = 12)
```
"""
module ModelSpecs

using ..Grammar: ModelFormula, VarTerm, AutoVarTerm, compile_ets_formula
using ..Utils: Formula


export AbstractModelSpec, AbstractFittedModel

export ArimaSpec, FittedArima
export EtsSpec, FittedEts
export SesSpec, FittedSes
export HoltSpec, FittedHolt
export HoltWintersSpec, FittedHoltWinters
export CrostonSpec, FittedCroston
export ModelCollection, FittedModelCollection, ForecastModelCollection
export GroupedFittedModels, GroupedForecasts
export PanelData

export model, extract_metrics, forecast_table
export successful_models, failed_groups, errors

include("abstract.jl")
include("arima_spec.jl")
include("ets_spec.jl")
include("smoothing_specs.jl")
include("panel_data.jl")
include("xreg_utils.jl")
include("model.jl")
include("grouped.jl")
include("fit.jl")
include("fit_grouped.jl")

end
