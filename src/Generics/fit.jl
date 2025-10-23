"""
    fit.jl

Generic fit() function for model specifications.

This file provides the generic interface for fitting model specifications to data,
following the fable-like workflow: spec → fit() → fitted model → forecast()
"""

"""
    fit(spec, data; kwargs...)

Generic function for fitting a model specification to data.

# Arguments
- `spec::AbstractModelSpec` - Model specification to fit
- `data` - Tables.jl-compatible data containing target and exogenous variables

# Keyword Arguments
- Vary by model type (e.g., `m` for seasonal period in ARIMA)

# Returns
`AbstractFittedModel` - Fitted model ready for forecasting

# Examples
```julia
# Fit ARIMA specification
spec = ArimaSpec(@formula(sales = p() + q()))
fitted = fit(spec, data, m = 12)

# With exogenous variables
spec = ArimaSpec(@formula(sales = p() + q() + temperature))
fitted = fit(spec, data, m = 7)
```

# See Also
- [`ArimaSpec`](@ref)
- [`forecast`](@ref)
"""
function fit end

export fit
