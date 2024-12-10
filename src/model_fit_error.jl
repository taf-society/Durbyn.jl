export ModelFitError

"""
    struct ModelFitError <: Exception

A custom exception to indicate that no model was able to be fitted.
Use this error to notify users when the model fitting process fails
due to the inability to find a suitable model.

# Fields
- `msg::String`: A message providing details about why the model fitting failed.

# Example

```julia
if best_ic == Inf
    throw(ModelFitError("No model was able to be fitted with the provided data and parameters."))
end
```
"""
struct ModelFitError <: Exception
    msg::String
end

"""
    Base.showerror(io::IO, e::ModelFitError)

Prints the custom error message for `ModelFitError`.

# Example

```julia
try
    throw(ModelFitError("Test error"))
catch e
    showerror(stdout, e)
end
```
"""
function Base.showerror(io::IO, e::ModelFitError)
    print(io, "ModelFitError: ", e.msg, "\n")
end