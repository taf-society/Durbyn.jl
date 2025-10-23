module TableOps

using Tables

export select, query, arrange, groupby, mutate, summarise, summarize, pivot_longer, pivot_wider, glimpse
export GroupedTable

include("ops.jl")

end
