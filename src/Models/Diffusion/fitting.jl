"""
    Diffusion Model Fitting

Core fitting logic for diffusion curve models.
"""

"""
    diffusion_cost(params, y, Y, model_type, fixed_params, loss, cumulative, mscal_factor)

Compute the cost function for diffusion model optimization.

# Arguments
- `params::Vector{Float64}`: Parameters to optimize (scaled)
- `y::Vector{Float64}`: Adoption per period
- `Y::Vector{Float64}`: Cumulative adoption
- `model_type::DiffusionModelType`: Type of diffusion model
- `fixed_params::Dict{Symbol, Float64}`: Parameters held fixed (not optimized)
- `loss::Int`: Loss function power (1=MAE, 2=MSE)
- `cumulative::Bool`: Whether to optimize on cumulative values
- `mscal_factor::Float64`: Market potential scaling factor

# Returns
Cost value (sum of errors raised to loss power).
"""
function diffusion_cost(params::Vector{Float64}, y::Vector{Float64}, Y::Vector{Float64},
                        model_type::DiffusionModelType, fixed_params::Dict{Symbol, Float64},
                        loss::Int, cumulative::Bool, mscal_factor::Float64)
    T = Float64
    n = length(y)

    # Reconstruct full parameter set
    full_params = _reconstruct_params(params, model_type, fixed_params, mscal_factor)

    # Always check for positive parameters - diffusion curves require positive params
    # This also handles the ibound case and protects against numerical gradient steps
    # that might temporarily violate bounds
    for (k, v) in pairs(full_params)
        if v <= 0 || !isfinite(v)
            return T(1e200)
        end
    end

    # Generate curve (now safe since all params are positive and finite)
    curve = get_curve(model_type, n, full_params)

    # Compute cost
    if cumulative
        errors = Y .- curve.cumulative
    else
        errors = y .- curve.adoption
    end

    # Handle loss function
    if loss == 2
        return sum(errors .^ 2)
    elseif loss == 1
        return sum(abs.(errors))
    else
        return sum(abs.(errors) .^ loss)
    end
end

"""
    _reconstruct_params(opt_params, model_type, fixed_params, mscal_factor) -> NamedTuple

Reconstruct full parameter NamedTuple from optimization vector and fixed values.
"""
function _reconstruct_params(opt_params::Vector{Float64}, model_type::DiffusionModelType,
                              fixed_params::Dict{Symbol, Float64}, mscal_factor::Float64)

    # Determine which parameters are being optimized
    param_names = _get_param_names(model_type)
    opt_idx = 1
    result = Dict{Symbol, Float64}()

    for name in param_names
        if haskey(fixed_params, name)
            result[name] = fixed_params[name]
        else
            result[name] = opt_params[opt_idx]
            opt_idx += 1
        end
    end

    # Rescale market potential ONLY if it was optimized (not fixed)
    # Fixed m values should remain as the user specified them
    if haskey(result, :m) && !haskey(fixed_params, :m)
        result[:m] = result[:m] * mscal_factor
    end

    return _dict_to_params(result, model_type)
end

"""
    _get_param_names(model_type) -> Vector{Symbol}

Get parameter names for a given model type.
"""
function _get_param_names(model_type::DiffusionModelType)
    if model_type == Bass
        return [:m, :p, :q]
    elseif model_type == Gompertz
        return [:m, :a, :b]
    elseif model_type == GSGompertz
        return [:m, :a, :b, :c]
    elseif model_type == Weibull
        return [:m, :a, :b]
    else
        error("Unknown model type: $model_type")
    end
end

"""
    _dict_to_params(d, model_type) -> NamedTuple

Convert dictionary to properly typed NamedTuple for model type.
"""
function _dict_to_params(d::Dict{Symbol, Float64}, model_type::DiffusionModelType)
    if model_type == Bass
        return (m=d[:m], p=d[:p], q=d[:q])
    elseif model_type == Gompertz
        return (m=d[:m], a=d[:a], b=d[:b])
    elseif model_type == GSGompertz
        return (m=d[:m], a=d[:a], b=d[:b], c=d[:c])
    elseif model_type == Weibull
        return (m=d[:m], a=d[:a], b=d[:b])
    else
        error("Unknown model type: $model_type")
    end
end

"""
    _params_to_dict(params) -> Dict{Symbol, Float64}

Convert NamedTuple to dictionary.
"""
function _params_to_dict(params::NamedTuple)
    return Dict{Symbol, Float64}(k => Float64(v) for (k, v) in pairs(params))
end

"""
    _scale_m(m, y; up=false) -> Float64

Scale market potential parameter for optimization stability.

# Arguments
- `m::Float64`: Market potential
- `y::Vector`: Adoption data
- `up::Bool=false`: If true, scale up; if false, scale down

# Returns
Scaled market potential.
"""
function _scale_m(m::Float64, y::AbstractVector; up::Bool=false)
    scale_factor = 10 * sum(y)
    if up
        return m * scale_factor
    else
        return m / scale_factor
    end
end

"""
    fit_diffusion(y; model_type=Bass, cleanlead=true, w=nothing, loss=2, cumulative=true,
                  mscal=true, maxiter=500, method="L-BFGS-B", initpar="linearize") -> DiffusionFit

Fit a diffusion model to adoption data.

# Arguments
- `y::Vector{<:Real}`: Adoption per period data

# Keyword Arguments
- `model_type::DiffusionModelType=Bass`: Type of model to fit
- `cleanlead::Bool=true`: If true, remove leading zeros before fitting (matches R behavior).
  When true, fitted values are returned for the cleaned series only. The offset field
  indicates how many leading zeros were removed.
- `w::Union{Nothing, NamedTuple}=nothing`: Fixed parameter values. Use `nothing` values
  to indicate parameters to estimate. For example, `(m=nothing, p=0.03, q=nothing)` fixes
  p at 0.03 while estimating m and q. All parameters can be fixed (matches R behavior).
- `loss::Int=2`: Loss function power (1=MAE, 2=MSE)
- `cumulative::Bool=true`: Optimize on cumulative adoption values
- `mscal::Bool=true`: Scale market parameter for optimization stability
- `maxiter::Int=500`: Maximum optimization iterations
- `method::String="L-BFGS-B"`: Optimization method
- `initpar`: Initialization method. Can be:
  - `"linearize"` or `"linearise"` (default): analytical methods with Bass optimization
    for Gompertz/GSGompertz init
  - `"preset"`: fixed preset values
  - `Vector{<:Real}`: numeric vector of initial parameter values (length must match model)

# Returns
`DiffusionFit` struct containing fitted parameters and diagnostics.

# Examples
```julia
# Basic Bass model fit
y = [5, 10, 25, 45, 70, 85, 75, 50, 30, 15]
fit = fit_diffusion(y)

# Gompertz model with fixed market potential
fit = fit_diffusion(y, model_type=Gompertz, w=(m=500.0, a=nothing, b=nothing))

# Fully fixed parameters (matches R behavior)
fit = fit_diffusion(y, model_type=Bass, w=(m=500.0, p=0.03, q=0.38))

# Using L1 loss
fit = fit_diffusion(y, loss=1)

# Custom initial values (numeric vector)
fit = fit_diffusion(y, model_type=Bass, initpar=[500.0, 0.03, 0.38])

# Keep leading zeros in output
y = [0, 0, 5, 10, 25, 45]
fit = fit_diffusion(y, cleanlead=false)  # fitted values for full series
```
"""
function fit_diffusion(y::AbstractVector{<:Real};
                       model_type::DiffusionModelType=Bass,
                       cleanlead::Bool=true,
                       w::Union{Nothing, NamedTuple}=nothing,
                       loss::Int=2,
                       cumulative::Bool=true,
                       mscal::Bool=true,
                       maxiter::Int=500,
                       method::String="L-BFGS-B",
                       initpar::Union{String, AbstractVector{<:Real}}="linearize")

    T = Float64
    y_original = collect(T.(y))

    # Handle leading zeros based on cleanlead parameter
    if cleanlead
        y_clean, offset = _cleanzero(y_original)
    else
        y_clean = y_original
        offset = 0
    end

    n_clean = length(y_clean)
    n_nonzero = count(!=(0), y_clean)

    if n_clean < 3
        throw(ArgumentError("Need at least 3 observations for diffusion fitting"))
    end

    if n_nonzero < 3
        throw(ArgumentError("Need at least 3 non-zero observations for diffusion fitting"))
    end

    # Compute cumulative for cleaned data
    Y_clean = cumsum(y_clean)

    # Get initial parameters
    # Handle different initpar types (matches R behavior)
    param_names = _get_param_names(model_type)
    n_params = length(param_names)

    if initpar isa AbstractVector
        # Numeric init vector provided (matches R's is.numeric(initpar) path)
        if length(initpar) != n_params
            throw(ArgumentError("$model_type requires $n_params parameters for initpar, got $(length(initpar))"))
        end
        init_dict = Dict{Symbol, Float64}(param_names[i] => Float64(initpar[i]) for i in 1:n_params)
        init_params = _dict_to_params(init_dict, model_type)
        is_linearize = false
    else
        # String initpar - normalize spelling
        initpar_norm = initpar == "linearise" ? "linearize" : initpar

        init_params = get_init(model_type, y_clean;
                               initpar=initpar_norm,
                               loss=loss,
                               mscal=mscal,
                               method=method,
                               maxiter=maxiter)
        is_linearize = initpar_norm == "linearize"
    end

    # R check: if init[1] < max(y) then init[1] = max(y)
    # This only applies in the linearize path (not preset or numeric)
    if is_linearize
        max_y = maximum(y_clean)
        if init_params.m < max_y
            init_dict_check = _params_to_dict(init_params)
            init_dict_check[:m] = max_y
            init_params = _dict_to_params(init_dict_check, model_type)
        end
    end

    # Determine which parameters to optimize vs fix
    fixed_params = Dict{Symbol, Float64}()
    opt_param_names = Symbol[]

    if !isnothing(w)
        for name in param_names
            if hasproperty(w, name) && !isnothing(getproperty(w, name))
                fixed_params[name] = T(getproperty(w, name))
            else
                push!(opt_param_names, name)
            end
        end
    else
        opt_param_names = param_names
    end

    # Handle fully fixed parameters (match R behavior - no optimization needed)
    if isempty(opt_param_names)
        # All parameters are fixed - just compute the fit directly
        full_params = _dict_to_params(Dict{Symbol, Float64}(
            name => T(getproperty(w, name)) for name in param_names
        ), model_type)

        curve = get_curve(model_type, n_clean, full_params)
        residuals = y_clean .- curve.adoption
        mse = mean(residuals .^ 2)

        return DiffusionFit(
            model_type,
            full_params,
            curve.adoption,
            curve.cumulative,
            residuals,
            mse,
            y_clean,
            y_original,
            offset,
            full_params,  # init_params same as final for fixed
            loss,
            cumulative
        )
    end

    # Compute scaling factor (ensure non-zero to avoid division errors)
    y_sum = sum(y_clean)
    mscal_factor = mscal && y_sum > 0 ? 10 * y_sum : one(T)

    # Build initial parameter vector for optimization
    init_dict = _params_to_dict(init_params)
    x0 = T[]

    for name in param_names
        if !(name in keys(fixed_params))
            val = init_dict[name]
            # Scale m if needed
            if name == :m && mscal
                val = val / mscal_factor
            end
            push!(x0, val)
        end
    end

    # Handle bounds based on method (matches R's checkInit behavior)
    # L-BFGS-B: uses explicit lower bounds
    # Other methods: no explicit bounds (cost function handles invalid params internally)
    if method == "L-BFGS-B"
        lower = fill(T(1e-9), length(x0))
        upper = fill(T(Inf), length(x0))
    else
        lower = fill(T(-Inf), length(x0))
        upper = fill(T(Inf), length(x0))
    end

    # Create objective function
    function objective(params)
        return diffusion_cost(params, y_clean, Y_clean, model_type,
                             fixed_params, loss, cumulative, mscal_factor)
    end

    # Optimize
    result = optim(x0, objective;
                   method=method,
                   lower=lower,
                   upper=upper,
                   control=Dict("maxit" => maxiter))

    # Extract optimized parameters
    opt_params = result.par
    final_params = _reconstruct_params(opt_params, model_type, fixed_params, mscal_factor)

    # Generate fitted curve for cleaned data only (matches R behavior)
    curve = get_curve(model_type, n_clean, final_params)

    # Compute residuals and MSE on cleaned data
    residuals = y_clean .- curve.adoption
    mse = mean(residuals .^ 2)

    return DiffusionFit(
        model_type,
        final_params,
        curve.adoption,
        curve.cumulative,
        residuals,
        mse,
        y_clean,
        y_original,
        offset,
        init_params,
        loss,
        cumulative
    )
end

"""
    diffusion(y; model_type=Bass, kwargs...) -> DiffusionFit

Convenience wrapper for `fit_diffusion`.

# Arguments
Same as `fit_diffusion`.

# Returns
`DiffusionFit` struct.

# Example
```julia
y = [5, 10, 25, 45, 70, 85, 75, 50, 30, 15]
fit = diffusion(y)
fit = diffusion(y, model_type=Gompertz)
```
"""
function diffusion(y::AbstractVector{<:Real}; kwargs...)
    return fit_diffusion(y; kwargs...)
end
