"""
    PanelData(data; groupby=nothing, date=nothing, m=nothing)

Lightweight container for panel/panel-like datasets, carrying grouping columns,
an optional date column, and seasonal period metadata alongside the underlying table.
"""
struct PanelData
    data
    groups::Vector{Symbol}
    date::Union{Symbol, Nothing}
    m::Union{Int, Nothing}
end

_normalize_groups(::Nothing) = Symbol[]
_normalize_groups(col::Symbol) = Symbol[col]
function _normalize_groups(cols::AbstractVector)
    return Symbol[Symbol(c) for c in cols]
end

function PanelData(data;
                   groupby::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing,
                   date::Union{Symbol, Nothing} = nothing,
                   m::Union{Int, Nothing} = nothing)
    PanelData(data, _normalize_groups(groupby), date, m)
end

groups(panel::PanelData) = panel.groups
datecol(panel::PanelData) = panel.date
seasonal_period(panel::PanelData) = panel.m

function Base.show(io::IO, panel::PanelData)
    group_str = isempty(panel.groups) ? "(none)" : join(string.(panel.groups), ", ")
    date_str = isnothing(panel.date) ? "nothing" : string(panel.date)
    m_str = isnothing(panel.m) ? "nothing" : string(panel.m)
    print(io, "PanelData(groups=", group_str, ", date=", date_str, ", m=", m_str, ")")
end
