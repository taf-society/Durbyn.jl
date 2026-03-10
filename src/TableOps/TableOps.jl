module TableOps

using Tables
import ..ModelSpecs: ForecastModelCollection, as_table, PanelData,
                     GroupedFittedModels, GroupedForecasts, successful_models, failed_groups, errors
import ..Generics: head, tail

export select, query, arrange, groupby, mutate, summarise, summarize, pivot_longer, pivot_wider, glimpse
export GroupedTable
# Additional table manipulation exports
export rename, distinct, bind_rows, ungroup
# Column selection helpers
export all_of, everything, across, AllOf, Everything, Across, ColumnSelector
# Reshaping and cleaning exports
export separate, unite, fill_missing, complete
# Join functions
export inner_join, left_join, right_join, full_join, semi_join, anti_join
export head, tail

include("ops.jl")
include("head_tail.jl")
include("glimpse_extensions.jl")

end
