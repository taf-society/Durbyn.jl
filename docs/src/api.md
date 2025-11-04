# API Reference

!!! note
    Table operations functions (`glimpse`, `select`, `query`, `arrange`, `groupby`, `mutate`, `summarise`, `pivot_longer`, `pivot_wider`) are documented in the [Table Operations](tableops.md) guide.

```@autodocs
Modules = [Durbyn]
Order   = [:module, :type, :function, :macro]
Filter  = t -> begin
    # Exclude TableOps functions (documented separately in tableops.md)
    name = string(nameof(t))
    !(name in ["glimpse", "select", "query", "arrange", "groupby", "mutate", "summarise", "summarize", "pivot_longer", "pivot_wider", "GroupedTable"])
end
```
