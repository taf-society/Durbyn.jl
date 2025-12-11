module Durbyn

include("Utils/Utils.jl")

include("Optimize/Optimize.jl")
include("Generics/Generics.jl")
include("Stats/Stats.jl")
include("Grammar/Grammar.jl")
include("ModelSpecs/ModelSpecs.jl")
include("TableOps/TableOps.jl")

include("Models/Naive/Naive.jl")
include("Models/ExponentialSmoothing/ExponentialSmoothing.jl")
include("Models/Arima/Arima.jl")
include("Models/IntermittentDemand/IntermittentDemand.jl")
include("Models/Ararma/Ararma.jl")
include("Models/Bats/Bats.jl")
include("Models/Tbats/Tbats.jl")


using .Utils
using .Generics
using .ExponentialSmoothing
using .Optimize
using .Stats
using .Grammar
using .ModelSpecs
using .TableOps
using .Naive
using .Arima
using .IntermittentDemand
using .Bats
using .Tbats

import .Utils: air_passengers, NamedMatrix, get_elements, get_vector, align_columns, add_drift_term, cbind
import .Utils: Formula, parse_formula, compile, model_matrix, model_frame
import .Generics: plot, fitted, residuals, summary, predict, forecast, fit, accuracy, list_series
import .Optimize: NelderMeadOptions
import .Grammar: p, q, d, P, Q, D, auto, ModelFormula, @formula, VarTerm, AutoVarTerm, ArarTerm
import .Grammar: e, t, s, drift, ses, holt, hw, holt_winters, croston, arar
import .ModelSpecs: AbstractModelSpec, AbstractFittedModel, ArimaSpec, FittedArima, ArarSpec, FittedArar, ArarmaSpec, FittedArarma, EtsSpec, TbatsSpec, BatsSpec, FittedEts
import .ModelSpecs: SesSpec, FittedSes, HoltSpec, FittedHolt, HoltWintersSpec, FittedHoltWinters
import .ModelSpecs: CrostonSpec, FittedCroston, ModelCollection, FittedModelCollection
import .ModelSpecs: ForecastModelCollection, model, PanelData, forecast_table
import .ModelSpecs: GroupedFittedModels, GroupedForecasts, successful_models, failed_groups
import .Arima: arima, arima_rjh, auto_arima, ArimaFit, PDQ
import .Ararma: ARAR, ArarmaModel, arar, ararma, auto_ararma
import .Bats: bats, BATSModel
import .Tbats: tbats, TBATSModel
import .TableOps: select, query, arrange, groupby, mutate, summarise, summarize, pivot_longer, pivot_wider, glimpse, GroupedTable

export air_passengers, NamedMatrix, get_elements, get_vector, align_columns, add_drift_term, cbind
export Formula, parse_formula, compile, model_matrix, model_frame
export plot, fitted, residuals, summary, predict, forecast, fit, accuracy, list_series
export coef, coefficients, coefs
export NelderMeadOptions
export p, q, d, P, Q, D, auto, ModelFormula, @formula, VarTerm, AutoVarTerm, ArarTerm
export e, t, s, drift, ses, holt, hw, holt_winters, croston, arar
export AbstractModelSpec, AbstractFittedModel, ArimaSpec, FittedArima, ArarSpec, FittedArar, ArarmaSpec, FittedArarma, EtsSpec, FittedEts
export SesSpec, FittedSes, HoltSpec, FittedHolt, HoltWintersSpec, FittedHoltWinters
export BatsSpec, FittedBats, TbatsSpec, FittedTbats
export CrostonSpec, FittedCroston, ModelCollection, FittedModelCollection, ForecastModelCollection
export model, PanelData, forecast_table
export GroupedFittedModels, GroupedForecasts, successful_models, failed_groups
export arima, arima_rjh, auto_arima, ArimaFit, PDQ
export ARAR, ArarmaModel, ararma, auto_ararma
export bats, BATSModel
export tbats, TBATSModel
export select, query, arrange, groupby, mutate, summarise, summarize, pivot_longer, pivot_wider, glimpse, GroupedTable

include("glimpse_extensions.jl")

end
