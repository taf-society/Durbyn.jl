"""
    Model constructor functions

Provides the `model()` function for creating model specifications
and collections.
"""

"""
    model(specs::AbstractModelSpec...; names=nothing)

Create a model specification or collection of specifications.

# Behavior
- **Single spec**: Returns the spec directly
- **Multiple specs**: Returns `ModelCollection`

# Arguments
- `specs...` - One or more model specifications

# Keyword Arguments
- `names::Union{Vector{String}, Nothing}` - Names for each model
  - If `nothing`, auto-generates names: "model_1", "model_2", etc.
  - If provided, length must match number of specs

# Returns
- Single `AbstractModelSpec` if one spec provided
- `ModelCollection` if multiple specs provided

# Examples
```julia
# Single model
spec = model(ArimaSpec(@formula(y = p() + q())))
# Returns: ArimaSpec (not wrapped)

# Multiple models for comparison
models = model(
    ArimaSpec(@formula(sales = p() + q())),
    ArimaSpec(@formula(sales = p(1) + d(1) + q(1))),
    ArimaSpec(@formula(sales = p(2) + d(1) + q(2)))
)
# Returns: ModelCollection with 3 models

# With custom names
models = model(
    ArimaSpec(@formula(sales = p() + q())),
    ArimaSpec(@formula(sales = p(1) + d(1) + q(1))),
    names = ["auto", "arima_111"]
)

# Fit all and compare
fitted = fit(models, data, m = 12)
best = select_best(fitted, metric = :aic)
```

# See Also
- [`ModelCollection`](@ref)
- [`ArimaSpec`](@ref)
- [`fit`](@ref)
"""
function model(specs::AbstractModelSpec...; names::Union{Vector{String}, Nothing} = nothing)
    n_specs = length(specs)

    # Single spec - return directly (not wrapped)
    if n_specs == 1
        if !isnothing(names) && length(names) > 1
            @warn "Single spec provided but $(length(names)) names given. Ignoring names."
        end
        return specs[1]
    end

    # Multiple specs - create collection
    specs_vec = collect(specs)

    # Generate or validate names
    if isnothing(names)
        names = ["model_$i" for i in 1:n_specs]
    else
        length(names) == n_specs ||
            error("Number of names ($(length(names))) must match number of specs ($n_specs)")
    end

    return ModelCollection(specs_vec, names)
end
