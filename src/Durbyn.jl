module Durbyn

include("Utils/Utils.jl")

include("Optimize/Optimize.jl")
include("Generics/Generics.jl")
include("Stats/Stats.jl")
include("Grammar/Grammar.jl")

include("Models/Naive/Naive.jl")
include("Models/ExponentialSmoothing/ExponentialSmoothing.jl")
include("Models/Arima/Arima.jl")
include("Models/IntermittentDemand/IntermittentDemand.jl")
include("Models/Ararma/Ararma.jl")
include("Models/Bats/Bats.jl")
include("Models/Tbats/Tbats.jl")
include("Models/Theta/Theta.jl")
include("Models/Diffusion/Diffusion.jl")
include("Models/KolmogorovWiener/KolmogorovWiener.jl")

include("ModelSpecs/ModelSpecs.jl")
include("TableOps/TableOps.jl")


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
using .KolmogorovWiener
import .KolmogorovWiener: kolmogorov_wiener, KWFilterResult, kw_decomposition

import .Utils: air_passengers, NamedMatrix, get_elements, get_vector, align_columns, add_drift_term, cbind
import .Utils: Formula, parse_formula, compile
import .Generics: plot, fitted, residuals, summary, predict, forecast, fit, accuracy, list_series, head, tail
import .Optimize: NelderMeadOptions
import .Grammar: p, q, d, P, Q, D, auto, ModelFormula, @formula, VarTerm, AutoVarTerm, ArarTerm, ThetaTerm, DiffusionTerm, KwFilterTerm
import .Grammar: e, t, s, drift, ses, holt, hw, holt_winters, croston, arar, kw_filter
import .Grammar: naive_term, snaive_term, rw_term, meanf_term, NaiveTerm, SnaiveTerm, RwTerm, MeanfTerm
import .ModelSpecs: AbstractModelSpec, AbstractFittedModel, ArimaSpec, FittedArima, ArarSpec, FittedArar, ArarmaSpec, FittedArarma, EtsSpec, TbatsSpec, BatsSpec, FittedEts, DiffusionSpec, FittedDiffusion, KwFilterSpec, FittedKwFilter
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
import .Stats: Decomposition
import .Stats: acf, pacf, ACFResult, PACFResult, interpolate_missing, longest_contiguous, check_missing, handle_missing
import .Stats: MissingMethod, Contiguous, Interpolate, FailMissing

# ── export: auto-imported with `using Durbyn` ──────────────────────────────

# Core workflow
export fit, forecast, accuracy, plot, summary, fitted, residuals, predict, list_series, head, tail

# Data
export air_passengers, PanelData, as_table, model

# Grammar (needed inside @formula)
export @formula
export p, q, d, P, Q, D, auto, drift
export e, t, s
export ses, holt, hw, holt_winters, croston, arar, kw_filter
export naive_term, snaive_term, rw_term, meanf_term

# Model Specs
export ArimaSpec, EtsSpec, BatsSpec, TbatsSpec, ThetaSpec, DiffusionSpec, KwFilterSpec
export SesSpec, HoltSpec, HoltWintersSpec, CrostonSpec
export ArarSpec, ArarmaSpec
export NaiveSpec, SnaiveSpec, RwSpec, MeanfSpec

# Array-interface model functions
export arima, auto_arima, bats, tbats
export theta, auto_theta
export ararma, auto_ararma
export naive, snaive, rw, rwf, meanf
export diffusion, fit_diffusion
export kolmogorov_wiener, kw_decomposition

# Stats
export acf, pacf
export interpolate_missing, longest_contiguous, check_missing, handle_missing
export MissingMethod, Contiguous, Interpolate, FailMissing

# TableOps
export select, query, arrange, groupby, mutate, summarise, summarize
export pivot_longer, pivot_wider, glimpse

end
