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
include("Models/Theta/Theta.jl")
include("Models/Diffusion/Diffusion.jl")


using .Utils
using .Generics
using .ExponentialSmoothing
using .Optimize
using .Stats
using .Grammar
using .ModelSpecs
using .TableOps
using .Naive
import .Naive: NaiveFit, MeanFit, naive, snaive, rw, rwf, meanf
using .Arima
using .IntermittentDemand
using .Bats
using .Tbats
using .Theta
using .Diffusion

import .Utils: air_passengers, NamedMatrix, get_elements, get_vector, align_columns, add_drift_term, cbind
import .Utils: Formula, parse_formula, compile, model_matrix, model_frame
import .Generics: plot, fitted, residuals, summary, predict, forecast, fit, accuracy, list_series
import .Optimize: NelderMeadOptions
import .Grammar: p, q, d, P, Q, D, auto, ModelFormula, @formula, VarTerm, AutoVarTerm, ArarTerm, ThetaTerm, DiffusionTerm
import .Grammar: e, t, s, drift, ses, holt, hw, holt_winters, croston, arar
import .Grammar: naive_term, snaive_term, rw_term, meanf_term, NaiveTerm, SnaiveTerm, RwTerm, MeanfTerm
import .ModelSpecs: AbstractModelSpec, AbstractFittedModel, ArimaSpec, FittedArima, ArarSpec, FittedArar, ArarmaSpec, FittedArarma, EtsSpec, TbatsSpec, BatsSpec, FittedEts, DiffusionSpec, FittedDiffusion
import .ModelSpecs: SesSpec, FittedSes, HoltSpec, FittedHolt, HoltWintersSpec, FittedHoltWinters
import .ModelSpecs: CrostonSpec, FittedCroston, ModelCollection, FittedModelCollection
import .ModelSpecs: NaiveSpec, FittedNaive, SnaiveSpec, FittedSnaive, RwSpec, FittedRw, MeanfSpec, FittedMeanf
import .ModelSpecs: ForecastModelCollection, model, PanelData, as_table
import .ModelSpecs: GroupedFittedModels, GroupedForecasts, successful_models, failed_groups
import .Arima: arima, arima_rjh, auto_arima, ArimaFit, PDQ
import .Ararma: ARAR, ArarmaModel, arar, ararma, auto_ararma
import .Bats: bats, BATSModel
import .Tbats: tbats, TBATSModel
import .Theta: theta, auto_theta, ThetaFit, ThetaModelType, STM, OTM, DSTM, DOTM
import .Diffusion: diffusion, fit_diffusion, DiffusionFit, DiffusionModelType, Bass, Gompertz, GSGompertz, Weibull
import .TableOps: select, query, arrange, groupby, mutate, summarise, summarize, pivot_longer, pivot_wider, glimpse, GroupedTable
import .Stats: acf, pacf, ACFResult, PACFResult, na_interp, na_contiguous, na_fail, na_action

export air_passengers, NamedMatrix, get_elements, get_vector, align_columns, add_drift_term, cbind
export Formula, parse_formula, compile, model_matrix, model_frame
export plot, fitted, residuals, summary, predict, forecast, fit, accuracy, list_series
export coef, coefficients, coefs
export NelderMeadOptions
export p, q, d, P, Q, D, auto, ModelFormula, @formula, VarTerm, AutoVarTerm, ArarTerm, ThetaTerm, DiffusionTerm
export e, t, s, drift, ses, holt, hw, holt_winters, croston, arar
export naive_term, snaive_term, rw_term, meanf_term, NaiveTerm, SnaiveTerm, RwTerm, MeanfTerm
export AbstractModelSpec, AbstractFittedModel, ArimaSpec, FittedArima, ArarSpec, FittedArar, ArarmaSpec, FittedArarma, EtsSpec, FittedEts
export SesSpec, FittedSes, HoltSpec, FittedHolt, HoltWintersSpec, FittedHoltWinters
export BatsSpec, FittedBats, TbatsSpec, FittedTbats, ThetaSpec, FittedTheta, DiffusionSpec, FittedDiffusion
export CrostonSpec, FittedCroston, ModelCollection, FittedModelCollection, ForecastModelCollection
export NaiveSpec, FittedNaive, SnaiveSpec, FittedSnaive, RwSpec, FittedRw, MeanfSpec, FittedMeanf
export model, PanelData, as_table
export GroupedFittedModels, GroupedForecasts, successful_models, failed_groups
export arima, arima_rjh, auto_arima, ArimaFit, PDQ
export ARAR, ArarmaModel, ararma, auto_ararma
export NaiveFit, MeanFit, naive, snaive, rw, rwf, meanf
export bats, BATSModel
export tbats, TBATSModel
export theta, auto_theta, ThetaFit, ThetaModelType, STM, OTM, DSTM, DOTM
export diffusion, fit_diffusion, DiffusionFit, DiffusionModelType, Bass, Gompertz, GSGompertz, Weibull
export select, query, arrange, groupby, mutate, summarise, summarize, pivot_longer, pivot_wider, glimpse, GroupedTable
export acf, pacf, ACFResult, PACFResult, na_interp, na_contiguous, na_fail, na_action

include("glimpse_extensions.jl")

function __init__()
    @info """Durbyn.jl is under active development.
    API may change without notice. Bugs and performance issues may exist.
    Please report issues at: https://github.com/taf-society/Durbyn.jl/issues"""
end

end
