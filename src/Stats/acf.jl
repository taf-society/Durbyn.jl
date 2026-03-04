

"""
    plot(result::ACFResult; kwargs...)
    plot(result::PACFResult; kwargs...)

Plot ACF or PACF with confidence bands.

This function is implemented in the DurbynPlotsExt extension module.
Load Plots.jl to enable plotting: `using Plots`

# Example
```julia
using Plots
result = acf(y, 12)
plot(result)
```
"""
function plot end
