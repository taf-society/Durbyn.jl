module TableOps

using Tables
import ..ModelSpecs: ForecastModelCollection, forecast_table

export select, query, arrange, groupby, mutate, summarise, summarize, pivot_longer, pivot_wider, glimpse
export GroupedTable

include("ops.jl")

end
