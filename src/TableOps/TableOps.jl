module TableOps

using Tables
import ..ModelSpecs: ForecastModelCollection, as_table, PanelData

export select, query, arrange, groupby, mutate, summarise, summarize, pivot_longer, pivot_wider, glimpse
export GroupedTable
# Additional dplyr-style exports
export rename, distinct, bind_rows, ungroup
# Column selection helpers
export all_of, everything, across, AllOf, Everything, Across, ColumnSelector
# tidyr-style exports
export separate, unite, fill_missing, complete
# Join functions
export inner_join, left_join, right_join, full_join, semi_join, anti_join

include("ops.jl")

end
