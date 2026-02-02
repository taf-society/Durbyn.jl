"""
    Durbyn Forecasting Grammar

This module provides a Domain-Specific Language (DSL) for specifying forecasting models
in a declarative way. The grammar is designed to be extensible and support various
forecasting methods including ARIMA, exponential smoothing, and ML models.

- ARIMA order specification: p(), q(), P(), Q(), d(), D()

- Feature engineering: dow(), month(), woy(), lag(), ma(), rollsum(), etc.
- ML model specification
- Custom transformations
"""


"""
    AbstractTerm

Base type for all grammar terms in the Durbyn forecasting DSL.
All specific term types should inherit from this.
"""
abstract type AbstractTerm end

"""
    ArimaOrderTerm <: AbstractTerm

Represents an ARIMA model order term (p, d, q, P, D, or Q) with search range.

# Fields
- `term::Symbol` - The term type (:p, :d, :q, :P, :D, or :Q)
- `min::Int` - Minimum order to search
- `max::Int` - Maximum order to search

# Constructor
    ArimaOrderTerm(term::Symbol, min::Int, max::Int)

Creates an ARIMA order term with validation.

# Examples
```julia
# These are typically created via the p(), d(), q(), P(), D(), Q() functions
ArimaOrderTerm(:p, 1, 2)  # Search p from 1 to 2
ArimaOrderTerm(:d, 1, 1)  # Fixed d = 1
ArimaOrderTerm(:q, 2, 2)  # Fixed q = 2
```
"""
struct ArimaOrderTerm <: AbstractTerm
    term::Symbol
    min::Int
    max::Int

    function ArimaOrderTerm(term::Symbol, min::Int, max::Int)
        term ∈ (:p, :d, :q, :P, :D, :Q) ||
            throw(ArgumentError("ARIMA term must be one of :p, :d, :q, :P, :D, :Q, got :$(term)"))
        min >= 0 ||
            throw(ArgumentError("min must be non-negative, got $(min)"))
        max >= min ||
            throw(ArgumentError("max must be >= min, got max=$(max) < min=$(min)"))
        new(term, min, max)
    end
end


"""
    EtsComponentTerm <: AbstractTerm

Represents a component (error, trend, seasonal) in an ETS specification.

# Fields
- `component::Symbol` - One of `:error`, `:trend`, `:seasonal`
- `code::String` - Component code (`"A"`, `"M"`, `"N"`, `"Z"`)
"""
struct EtsComponentTerm <: AbstractTerm
    component::Symbol
    code::String

    function EtsComponentTerm(component::Symbol, code::AbstractString)
        component ∈ (:error, :trend, :seasonal) ||
            throw(ArgumentError("ETS component must be :error, :trend, or :seasonal, got :$(component)"))
        normalized = _validate_ets_code(component, code)
        new(component, normalized)
    end
end

"""
    EtsDriftTerm <: AbstractTerm

Represents the damping/drift setting for an ETS specification.
"""
struct EtsDriftTerm <: AbstractTerm
    damped::Union{Bool, Nothing}
end

"""
    SesTerm <: AbstractTerm

Sentinel term for Simple Exponential Smoothing specifications.
"""
struct SesTerm <: AbstractTerm
end

"""
    HoltTerm <: AbstractTerm

Represents Holt's method options within a formula (`holt()`).
"""
struct HoltTerm <: AbstractTerm
    damped::Union{Bool, Nothing}
    exponential::Bool
end

"""
    HoltWintersTerm <: AbstractTerm

Represents Holt-Winters seasonal exponential smoothing options (`hw()`/`holt_winters()`).
"""
struct HoltWintersTerm <: AbstractTerm
    seasonal::String
    damped::Union{Bool, Nothing}
    exponential::Bool
end

"""
    CrostonTerm <: AbstractTerm

Represents Croston's intermittent-demand model options within a formula.

# Fields
- `method::String` - Croston method variant:
  - `"hyndman"` - Simple Croston from ExponentialSmoothing module (default)
  - `"classic"` - Classical Croston from IntermittentDemand module
  - `"sba"` - Syntetos-Boylan Approximation (bias-corrected)
  - `"sbj"` - Shale-Boylan-Johnston Bias Correction
- `init_strategy::Union{String, Nothing}` - Initialization: "mean" or "naive" (IntermittentDemand only)
- `number_of_params::Union{Int, Nothing}` - Number of parameters to optimize: 1 or 2 (IntermittentDemand only)
- `cost_metric::Union{String, Nothing}` - Optimization metric: "mar", "msr", "mae", "mse" (IntermittentDemand only)
- `optimize_init::Union{Bool, Nothing}` - Optimize initial values (IntermittentDemand only)
- `rm_missing::Union{Bool, Nothing}` - Remove missing values (IntermittentDemand only)
"""
struct CrostonTerm <: AbstractTerm
    method::String
    init_strategy::Union{String, Nothing}
    number_of_params::Union{Int, Nothing}
    cost_metric::Union{String, Nothing}
    optimize_init::Union{Bool, Nothing}
    rm_missing::Union{Bool, Nothing}
end

"""
    ArarTerm <: AbstractTerm

Represents ARAR (AutoRegressive with Adaptive Reduction) model options within a formula.

# Fields
- `max_ar_depth::Union{Int, Nothing}` - Maximum AR depth to consider
- `max_lag::Union{Int, Nothing}` - Maximum lag for autocovariance computation
"""
struct ArarTerm <: AbstractTerm
    max_ar_depth::Union{Int, Nothing}
    max_lag::Union{Int, Nothing}
end

"""
    BatsTerm <: AbstractTerm

Represents BATS (Box-Cox transformation, ARMA errors, Trend and Seasonal) model options within a formula.

# Fields
- `seasonal_periods::Union{Vector{Int}, Int, Nothing}` - Seasonal period(s) for the model
- `use_box_cox::Union{Bool, Nothing}` - Whether to use Box-Cox transformation (nothing = automatic selection)
- `use_trend::Union{Bool, Nothing}` - Whether to include trend component (nothing = automatic selection)
- `use_damped_trend::Union{Bool, Nothing}` - Whether to use damped trend (nothing = automatic selection)
- `use_arma_errors::Union{Bool, Nothing}` - Whether to include ARMA error structure
"""
struct BatsTerm <: AbstractTerm
    seasonal_periods::Union{Vector{Int}, Int, Nothing}
    use_box_cox::Union{Bool, Nothing}
    use_trend::Union{Bool, Nothing}
    use_damped_trend::Union{Bool, Nothing}
    use_arma_errors::Union{Bool, Nothing}
end

"""
    TbatsTerm <: AbstractTerm

Represents a TBATS model specification term in a formula.

TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and
Seasonal components) extends BATS by using Fourier representation for seasonal
components, enabling non-integer seasonal periods and efficient handling of long cycles.

# Fields
- `seasonal_periods::Union{Vector{<:Real}, Real, Nothing}` - Seasonal period(s), can be non-integer (e.g., 52.18)
- `k::Union{Vector{Int}, Int, Nothing}` - Fourier order(s) per seasonal period (nothing = auto-select)
- `use_box_cox::Union{Bool, Nothing}` - Whether to use Box-Cox transformation (nothing = automatic selection)
- `use_trend::Union{Bool, Nothing}` - Whether to include trend component (nothing = automatic selection)
- `use_damped_trend::Union{Bool, Nothing}` - Whether to use damped trend (nothing = automatic selection)
- `use_arma_errors::Union{Bool, Nothing}` - Whether to include ARMA error structure (nothing = default true)

# Examples
```julia
# Created via tbats() function in formulas
@formula(sales = tbats(seasonal_periods=52.18))
@formula(sales = tbats(seasonal_periods=[7, 365.25]))
@formula(sales = tbats(seasonal_periods=[7, 365.25], k=[3, 10]))
```
"""
struct TbatsTerm <: AbstractTerm
    seasonal_periods::Union{Vector{<:Real}, Real, Nothing}
    k::Union{Vector{Int}, Int, Nothing}
    use_box_cox::Union{Bool, Nothing}
    use_trend::Union{Bool, Nothing}
    use_damped_trend::Union{Bool, Nothing}
    use_arma_errors::Union{Bool, Nothing}
end

"""
    ThetaTerm <: AbstractTerm

Represents a Theta model specification term in a formula.

The Theta method decomposes a time series into "theta lines" capturing long-term
trend and short-term dynamics, then combines their forecasts. Supports four variants:
STM, OTM, DSTM, DOTM.

# Fields
- `model_type::Union{Symbol, Nothing}` - Model variant: :STM, :OTM, :DSTM, :DOTM, or nothing (auto-select)
- `alpha::Union{Float64, Nothing}` - Smoothing parameter (0 < α < 1), nothing = optimize
- `theta::Union{Float64, Nothing}` - Theta parameter (≥ 1), nothing = optimize/fixed based on model
- `decomposition_type::Union{String, Nothing}` - Seasonal adjustment: "multiplicative", "additive", or nothing (auto)
- `nmse::Union{Int, Nothing}` - Steps for multi-step MSE calculation (1-30)

# Examples
```julia
@formula(sales = theta())                           # Auto-select best variant
@formula(sales = theta(model=:OTM))                 # Optimized Theta Model
@formula(sales = theta(model=:STM, alpha=0.3))      # Simple Theta with fixed alpha
@formula(sales = theta(decomposition="additive"))   # Force additive decomposition
```
"""
struct ThetaTerm <: AbstractTerm
    model_type::Union{Symbol, Nothing}
    alpha::Union{Float64, Nothing}
    theta::Union{Float64, Nothing}
    decomposition_type::Union{String, Nothing}
    nmse::Union{Int, Nothing}
end

"""
    DiffusionTerm <: AbstractTerm

Represents a diffusion model specification term in a formula.

Diffusion models capture technology adoption and market penetration patterns using
classical S-curve models.

# Fields
- `model_type::Union{Symbol, Nothing}` - Model type: :Bass, :Gompertz, :GSGompertz, :Weibull, or nothing (default Bass)
- `m::Union{Float64, Nothing}` - Market potential (nothing = estimate)
- `p::Union{Float64, Nothing}` - Innovation coefficient for Bass (nothing = estimate)
- `q::Union{Float64, Nothing}` - Imitation coefficient for Bass (nothing = estimate)
- `a::Union{Float64, Nothing}` - Parameter a for Gompertz/GSGompertz/Weibull (nothing = estimate)
- `b::Union{Float64, Nothing}` - Parameter b for Gompertz/GSGompertz/Weibull (nothing = estimate)
- `c::Union{Float64, Nothing}` - Parameter c for GSGompertz only (nothing = estimate)
- `loss::Union{Int, Nothing}` - Loss function power (1=MAE, 2=MSE, nothing = default 2)
- `cumulative::Union{Bool, Nothing}` - Optimize on cumulative values (nothing = default true)

# Examples
```julia
@formula(adoption = diffusion())                     # Default Bass model
@formula(adoption = diffusion(model=:Bass))          # Bass diffusion
@formula(adoption = diffusion(model=:Gompertz))      # Gompertz growth
@formula(adoption = diffusion(model=:Bass, m=1000))  # Bass with fixed market potential
@formula(adoption = diffusion(loss=1))               # Use L1 loss (MAE)
```
"""
struct DiffusionTerm <: AbstractTerm
    model_type::Union{Symbol, Nothing}
    m::Union{Float64, Nothing}
    p::Union{Float64, Nothing}
    q::Union{Float64, Nothing}
    a::Union{Float64, Nothing}
    b::Union{Float64, Nothing}
    c::Union{Float64, Nothing}
    loss::Union{Int, Nothing}
    cumulative::Union{Bool, Nothing}
end

"""
    NaiveTerm <: AbstractTerm

Represents a naive forecasting model specification in a formula.

The naive method uses the last observation as the forecast for all future periods.
"""
struct NaiveTerm <: AbstractTerm
end

"""
    SnaiveTerm <: AbstractTerm

Represents a seasonal naive forecasting model specification in a formula.

The seasonal naive method uses the observation from m periods ago as the forecast.
"""
struct SnaiveTerm <: AbstractTerm
end

"""
    RwTerm <: AbstractTerm

Represents a random walk forecasting model specification in a formula.

# Fields
- `drift::Bool` - Whether to include drift term
"""
struct RwTerm <: AbstractTerm
    drift::Bool
end

"""
    naive_term()

Specify a naive forecasting model in a formula.

The naive forecast uses the last observed value for all future periods.

# Examples
```julia
@formula(sales = naive_term())
```

# See Also
- [`snaive_term`](@ref) - Seasonal naive
- [`rw_term`](@ref) - Random walk with optional drift
"""
naive_term() = NaiveTerm()

"""
    snaive_term()

Specify a seasonal naive forecasting model in a formula.

The seasonal naive forecast uses the value from m periods ago.

# Examples
```julia
@formula(sales = snaive_term())
```

# See Also
- [`naive_term`](@ref) - Non-seasonal naive
- [`rw_term`](@ref) - Random walk with optional drift
"""
snaive_term() = SnaiveTerm()

"""
    rw_term(; drift::Bool=false)

Specify a random walk forecasting model in a formula.

# Keyword Arguments
- `drift::Bool=false` - Include drift term (linear trend)

# Examples
```julia
# Random walk without drift (equivalent to naive)
@formula(sales = rw_term())

# Random walk with drift
@formula(sales = rw_term(drift=true))
```

# See Also
- [`naive_term`](@ref) - Naive method
- [`snaive_term`](@ref) - Seasonal naive
"""
rw_term(; drift::Bool=false) = RwTerm(drift)

"""
    MeanfTerm <: AbstractTerm

Represents a mean forecasting model specification in a formula.

The mean method uses the sample mean as the forecast for all future periods.
"""
struct MeanfTerm <: AbstractTerm
end

"""
    meanf_term()

Specify a mean forecasting model in a formula.

The mean forecast uses the sample mean for all future periods.

# Examples
```julia
@formula(sales = meanf_term())
```

# See Also
- [`naive_term`](@ref) - Naive method
- [`snaive_term`](@ref) - Seasonal naive
"""
meanf_term() = MeanfTerm()

"""
    VarTerm <: AbstractTerm

Represents an exogenous variable to be used as a regressor in the model.

# Fields
- `name::Symbol` - Name of the variable (column) to include

# Examples
```julia
# Typically created by using bare symbols in @formula
VarTerm(:temperature)
VarTerm(:promotion)

# Used in formulas like:
# @formula(sales = p(1,2) + q(2,3) + temperature + promotion)
```
"""
struct VarTerm <: AbstractTerm
    name::Symbol
end

"""
    AutoVarTerm <: AbstractTerm

Sentinel term indicating that all eligible columns (excluding date, group, and target)
should be treated as exogenous regressors automatically.

Created via `auto()` in formulas, e.g. `@formula(value = auto())`.
"""
struct AutoVarTerm <: AbstractTerm
end

"""
    auto()

Shorthand for automatic exogenous-variable selection (see [`AutoVarTerm`](@ref)).
Use in formulas as `@formula(y = auto())` (optionally combined with ARIMA terms).
"""
auto() = AutoVarTerm()


"""
    p()
    p(value::Int)
    p(min::Int, max::Int)

Specify non-seasonal AR order(s) for ARIMA model selection.

# Arguments
- No arguments - Use auto_arima defaults (start_p=2, max_p=5)
- `value::Int` - Fixed AR order (searches only this value)
- `min::Int, max::Int` - Range of AR orders to search [min, max]

# Returns
`ArimaOrderTerm` representing the non-seasonal AR component

# Examples
```julia
p()        # Use defaults: search p ∈ {2, 3, 4, 5}
p(1)       # Fix p = 1
p(0, 3)    # Search p ∈ {0, 1, 2, 3}
p(2, 5)    # Search p ∈ {2, 3, 4, 5}
```

# Notes
In ARIMA(p,d,q) notation, `p` is the order of the autoregressive component.
"""
p() = ArimaOrderTerm(:p, 2, 5)  # auto_arima defaults
p(value::Int) = ArimaOrderTerm(:p, value, value)
p(min::Int, max::Int) = ArimaOrderTerm(:p, min, max)

"""
    q()
    q(value::Int)
    q(min::Int, max::Int)

Specify non-seasonal MA order(s) for ARIMA model selection.

# Arguments
- No arguments - Use auto_arima defaults (start_q=2, max_q=5)
- `value::Int` - Fixed MA order (searches only this value)
- `min::Int, max::Int` - Range of MA orders to search [min, max]

# Returns
`ArimaOrderTerm` representing the non-seasonal MA component

# Examples
```julia
q()        # Use defaults: search q ∈ {2, 3, 4, 5}
q(2)       # Fix q = 2
q(1, 3)    # Search q ∈ {1, 2, 3}
q(0, 2)    # Search q ∈ {0, 1, 2}
```

# Notes
In ARIMA(p,d,q) notation, `q` is the order of the moving average component.
"""
q() = ArimaOrderTerm(:q, 2, 5)  # auto_arima defaults
q(value::Int) = ArimaOrderTerm(:q, value, value)
q(min::Int, max::Int) = ArimaOrderTerm(:q, min, max)

"""
    d()
    d(value::Int)
    d(min::Int, max::Int)

Specify non-seasonal differencing order(s) for ARIMA model selection.

# Arguments
- No arguments - Let auto_arima determine via unit root tests (returns nothing)
- `value::Int` - Fixed differencing order
- `min::Int, max::Int` - Range of differencing orders (not recommended)

# Returns
- `nothing` when called with no arguments (let auto_arima determine)
- `ArimaOrderTerm` representing the non-seasonal differencing component otherwise

# Examples
```julia
d()        # Auto-determine via tests (KPSS, ADF, PP) - returns nothing
d(1)       # Fix d = 1 (first differencing)
d(0, 2)    # Search d ∈ {0, 1, 2} (not typical)
```

# Notes
In ARIMA(p,d,q) notation, `d` is the degree of non-seasonal differencing.
Typically d ∈ {0, 1, 2}. When d() is called without arguments or omitted entirely,
auto_arima uses unit root tests to determine the appropriate order.
"""
d() = nothing
d(value::Int) = ArimaOrderTerm(:d, value, value)
d(min::Int, max::Int) = ArimaOrderTerm(:d, min, max)

"""
    P()
    P(value::Int)
    P(min::Int, max::Int)

Specify seasonal AR order(s) for SARIMA model selection.

# Arguments
- No arguments - Use auto_arima defaults (start_P=1, max_P=2)
- `value::Int` - Fixed seasonal AR order (searches only this value)
- `min::Int, max::Int` - Range of seasonal AR orders to search [min, max]

# Returns
`ArimaOrderTerm` representing the seasonal AR component

# Examples
```julia
P()        # Use defaults: search P ∈ {1, 2}
P(1)       # Fix P = 1
P(0, 2)    # Search P ∈ {0, 1, 2}
```

# Notes
In ARIMA(p,d,q)(P,D,Q)[m] notation, `P` is the order of the seasonal autoregressive component.
"""
P() = ArimaOrderTerm(:P, 1, 2)  # auto_arima defaults
P(value::Int) = ArimaOrderTerm(:P, value, value)
P(min::Int, max::Int) = ArimaOrderTerm(:P, min, max)

"""
    Q()
    Q(value::Int)
    Q(min::Int, max::Int)

Specify seasonal MA order(s) for SARIMA model selection.

# Arguments
- No arguments - Use auto_arima defaults (start_Q=1, max_Q=2)
- `value::Int` - Fixed seasonal MA order (searches only this value)
- `min::Int, max::Int` - Range of seasonal MA orders to search [min, max]

# Returns
`ArimaOrderTerm` representing the seasonal MA component

# Examples
```julia
Q()        # Use defaults: search Q ∈ {1, 2}
Q(1)       # Fix Q = 1
Q(0, 2)    # Search Q ∈ {0, 1, 2}
```

# Notes
In ARIMA(p,d,q)(P,D,Q)[m] notation, `Q` is the order of the seasonal moving average component.
"""
Q() = ArimaOrderTerm(:Q, 1, 2)  # auto_arima defaults
Q(value::Int) = ArimaOrderTerm(:Q, value, value)
Q(min::Int, max::Int) = ArimaOrderTerm(:Q, min, max)

"""
    D()
    D(value::Int)
    D(min::Int, max::Int)

Specify seasonal differencing order(s) for SARIMA model selection.

# Arguments
- No arguments - Let auto_arima determine via seasonal strength tests (returns nothing)
- `value::Int` - Fixed seasonal differencing order
- `min::Int, max::Int` - Range of seasonal differencing orders (not recommended)

# Returns
- `nothing` when called with no arguments (let auto_arima determine)
- `ArimaOrderTerm` representing the seasonal differencing component otherwise

# Examples
```julia
D()        # Auto-determine via seasonal strength tests - returns nothing
D(1)       # Fix D = 1 (seasonal differencing)
D(0, 1)    # Search D ∈ {0, 1}
```

# Notes
In ARIMA(p,d,q)(P,D,Q)[m] notation, `D` is the degree of seasonal differencing.
Typically D ∈ {0, 1}. D > 1 is rarely recommended.
"""
D() = nothing
D(value::Int) = ArimaOrderTerm(:D, value, value)
D(min::Int, max::Int) = ArimaOrderTerm(:D, min, max)


const _ETS_ERROR_CODES = Set(["A", "M", "Z"])
const _ETS_TREND_CODES = Set(["N", "A", "M", "Z"])
const _ETS_SEASON_CODES = Set(["N", "A", "M", "Z"])

function _validate_ets_code(component::Symbol, code::AbstractString)
    normalized = uppercase(code)
    if component === :error
        normalized ∈ _ETS_ERROR_CODES ||
            throw(ArgumentError("Invalid ETS error code '$(code)'. Use \"A\", \"M\", or \"Z\"."))
    elseif component === :trend
        normalized ∈ _ETS_TREND_CODES ||
            throw(ArgumentError("Invalid ETS trend code '$(code)'. Use \"N\", \"A\", \"M\", or \"Z\"."))
    elseif component === :seasonal
        normalized ∈ _ETS_SEASON_CODES ||
            throw(ArgumentError("Invalid ETS seasonal code '$(code)'. Use \"N\", \"A\", \"M\", or \"Z\"."))
    else
        throw(ArgumentError("Unknown ETS component :$(component)."))
    end
    return normalized
end

"""
    ses()

Specify Simple Exponential Smoothing in a model formula.
"""
ses() = SesTerm()

function _validate_bool_or_auto(name::Symbol, value)
    if !(value === nothing || value isa Bool)
        throw(ArgumentError("Keyword $(name) must be Bool or nothing, got $(typeof(value))"))
    end
    return value
end

"""
    holt(; damped=nothing, exponential=false)

Specify Holt's linear trend method. `damped` may be `true`, `false`, or `nothing`
(use default). `exponential=true` requests an exponential trend.
"""
function holt(; damped=nothing, exponential::Bool=false)
    _validate_bool_or_auto(:damped, damped)
    return HoltTerm(damped, exponential)
end

_normalize_hw_seasonal(x::Symbol) = _normalize_hw_seasonal(string(x))
function _normalize_hw_seasonal(x::AbstractString)
    val = lowercase(x)
    val in ("additive", "multiplicative") ||
        throw(ArgumentError("seasonal must be \"additive\" or \"multiplicative\", got $(x)"))
    return val
end

"""
    hw(; seasonal="additive", damped=nothing, exponential=false)
    holt_winters(; seasonal="additive", damped=nothing, exponential=false)

Specify Holt-Winters seasonal exponential smoothing within a formula.
"""
function hw(; seasonal::Union{AbstractString,Symbol}="additive",
             damped=nothing,
             exponential::Bool=false)
    _validate_bool_or_auto(:damped, damped)
    seasonal_norm = _normalize_hw_seasonal(seasonal)
    if exponential && seasonal_norm == "additive"
        throw(ArgumentError("exponential trend is only supported with multiplicative seasonality in Holt-Winters."))
    end
    return HoltWintersTerm(seasonal_norm, damped, exponential)
end
holt_winters(; kwargs...) = hw(; kwargs...)

"""
    croston(; method="hyndman", init_strategy=nothing, number_of_params=nothing,
            cost_metric=nothing, optimize_init=nothing, rm_missing=nothing)

Specify Croston's intermittent demand model in a formula.

# Arguments
- `method::String` - Croston method variant (default: "hyndman"):
  - `"hyndman"` - Simple Croston from ExponentialSmoothing module
  - `"classic"` - Classical Croston from IntermittentDemand module
  - `"sba"` - Syntetos-Boylan Approximation (bias-corrected, recommended)
  - `"sbj"` - Shale-Boylan-Johnston Bias Correction

**IntermittentDemand-specific parameters** (only apply to "classic", "sba", "sbj"):
- `init_strategy::Union{String, Nothing}` - Initialization: "mean" or "naive" (default: "mean")
- `number_of_params::Union{Int, Nothing}` - Parameters to optimize: 1 or 2 (default: 2)
- `cost_metric::Union{String, Nothing}` - Optimization metric: "mar", "msr", "mae", "mse" (default: "mar")
- `optimize_init::Union{Bool, Nothing}` - Optimize initial values (default: true)
- `rm_missing::Union{Bool, Nothing}` - Remove missing values (default: false)

# Examples
```julia
# Simple Croston (ExponentialSmoothing)
@formula(demand = croston())
@formula(demand = croston(method="hyndman"))

# IntermittentDemand methods
@formula(demand = croston(method="classic"))
@formula(demand = croston(method="sba"))      # Recommended for intermittent demand
@formula(demand = croston(method="sbj"))

# With IntermittentDemand parameters
@formula(demand = croston(method="sba", cost_metric="mse"))
@formula(demand = croston(method="classic", init_strategy="naive", number_of_params=1))
```

# See Also
- [`CrostonSpec`](@ref) - Model specification wrapper
- ExponentialSmoothing.croston - Simple Croston implementation
- IntermittentDemand.croston_classic, croston_sba, croston_sbj - Advanced implementations
"""
function croston(; method::Union{AbstractString, Symbol} = "hyndman",
                   init_strategy::Union{AbstractString, Symbol, Nothing} = nothing,
                   number_of_params::Union{Int, Nothing} = nothing,
                   cost_metric::Union{AbstractString, Symbol, Nothing} = nothing,
                   optimize_init = nothing,
                   rm_missing = nothing)

    if !isnothing(optimize_init) && !(optimize_init isa Bool)
        throw(ArgumentError("optimize_init must be Bool or nothing, got $(typeof(optimize_init))"))
    end
    if !isnothing(rm_missing) && !(rm_missing isa Bool)
        throw(ArgumentError("rm_missing must be Bool or nothing, got $(typeof(rm_missing))"))
    end

    method_str = lowercase(String(method))

    valid_methods = ("hyndman", "classic", "sba", "sbj")
    if !(method_str in valid_methods)
        throw(ArgumentError(
            "method must be one of $(valid_methods), got \"$(method_str)\""
        ))
    end

    if !isnothing(init_strategy)
        init_str = lowercase(String(init_strategy))
        if !(init_str in ("mean", "naive"))
            throw(ArgumentError(
                "init_strategy must be \"mean\" or \"naive\", got \"$(init_str)\""
            ))
        end
        init_strategy = init_str
    end

    if !isnothing(number_of_params)
        if !(number_of_params in (1, 2))
            throw(ArgumentError(
                "number_of_params must be 1 or 2, got $(number_of_params)"
            ))
        end
    end

    if !isnothing(cost_metric)
        cost_str = lowercase(String(cost_metric))
        if !(cost_str in ("mar", "msr", "mae", "mse"))
            throw(ArgumentError(
                "cost_metric must be one of (\"mar\", \"msr\", \"mae\", \"mse\"), got \"$(cost_str)\""
            ))
        end
        cost_metric = cost_str
    end

    if method_str == "hyndman" &&
       (!isnothing(init_strategy) || !isnothing(number_of_params) ||
        !isnothing(cost_metric) || !isnothing(optimize_init) || !isnothing(rm_missing))
        @warn "IntermittentDemand-specific parameters (init_strategy, number_of_params, cost_metric, optimize_init, rm_missing) " *
              "are ignored for method=\"hyndman\". These parameters only apply to \"classic\", \"sba\", and \"sbj\" methods."
    end

    return CrostonTerm(method_str, init_strategy, number_of_params, cost_metric, optimize_init, rm_missing)
end

"""
    arar(; max_ar_depth=nothing, max_lag=nothing)

Specify ARAR (AutoRegressive with Adaptive Reduction) model in a formula.

# Arguments
- `max_ar_depth::Union{Int, Nothing}=nothing` - Maximum lag to consider when selecting the best 4-lag AR model (must be at least 4)
- `max_lag::Union{Int, Nothing}=nothing` - Maximum lag for computing autocovariance sequence

# Examples
```julia
# Basic ARAR with defaults
@formula(y = arar())

# ARAR with custom max_ar_depth
@formula(y = arar(max_ar_depth=15))

# ARAR with both parameters
@formula(y = arar(max_ar_depth=20, max_lag=20))
```
"""
function arar(; max_ar_depth::Union{Int, Nothing}=nothing, max_lag::Union{Int, Nothing}=nothing)
    if !isnothing(max_ar_depth) && max_ar_depth < 4
        throw(ArgumentError("max_ar_depth must be at least 4, got $(max_ar_depth)"))
    end
    if !isnothing(max_lag) && max_lag < 0
        throw(ArgumentError("max_lag must be non-negative, got $(max_lag)"))
    end
    return ArarTerm(max_ar_depth, max_lag)
end

"""
    bats(; seasonal_periods=nothing, use_box_cox=nothing, use_trend=nothing,
         use_damped_trend=nothing, use_arma_errors=nothing)

Specify BATS (Box-Cox transformation, ARMA errors, Trend and Seasonal) model in a formula.

# Arguments
- `seasonal_periods::Union{Int, Vector{Int}, Nothing}=nothing` - Seasonal period(s) for the model (e.g., 12 for monthly data, [24, 168] for multiple seasonality)
- `use_box_cox::Union{Bool, Nothing}=nothing` - Whether to use Box-Cox transformation (nothing = automatic selection)
- `use_trend::Union{Bool, Nothing}=nothing` - Whether to include trend component (nothing = automatic selection)
- `use_damped_trend::Union{Bool, Nothing}=nothing` - Whether to use damped trend (nothing = automatic selection)
- `use_arma_errors::Union{Bool, Nothing}=nothing` - Whether to include ARMA error structure (nothing = use default: true)

# Examples
```julia
# Basic BATS with defaults (automatic component selection)
@formula(y = bats())

# BATS with monthly seasonality
@formula(y = bats(seasonal_periods=12))

# BATS with multiple seasonal periods
@formula(y = bats(seasonal_periods=[24, 168]))

# BATS with Box-Cox and trend specified
@formula(y = bats(seasonal_periods=12, use_box_cox=true, use_trend=true))

# BATS with all options
@formula(y = bats(seasonal_periods=12, use_box_cox=true, use_trend=true,
                  use_damped_trend=false, use_arma_errors=true))
```
"""
function bats(; seasonal_periods::Union{Int, Vector{Int}, Nothing}=nothing,
               use_box_cox::Union{Bool, Nothing}=nothing,
               use_trend::Union{Bool, Nothing}=nothing,
               use_damped_trend::Union{Bool, Nothing}=nothing,
               use_arma_errors::Union{Bool, Nothing}=nothing)

    if !isnothing(seasonal_periods)
        if seasonal_periods isa Int
            seasonal_periods > 0 || throw(ArgumentError("seasonal_periods must be positive, got $(seasonal_periods)"))
        elseif seasonal_periods isa Vector{Int}
            all(m -> m > 0, seasonal_periods) || throw(ArgumentError("All seasonal_periods must be positive"))
        end
    end

    return BatsTerm(seasonal_periods, use_box_cox, use_trend, use_damped_trend, use_arma_errors)
end

"""
    tbats(; seasonal_periods, k, use_box_cox, use_trend, use_damped_trend, use_arma_errors)

Specify a TBATS model in a formula. TBATS (Trigonometric seasonality, Box-Cox
transformation, ARMA errors, Trend and Seasonal components) uses Fourier-based
seasonal representation, enabling:
- Non-integer seasonal periods (e.g., 52.18 weeks per year)
- Very long seasonal cycles (hundreds or thousands of periods)
- Multiple complex seasonalities (daily + weekly + yearly)
- Dual calendar effects (e.g., Gregorian + Hijri calendars)

# Arguments
- `seasonal_periods::Union{Real, Vector{<:Real}, Nothing}=nothing` - Seasonal period(s).
  Can be non-integer (e.g., 52.18 for weekly data with yearly seasonality).
  Use a vector for multiple seasonalities: `[7, 365.25]`
- `k::Union{Int, Vector{Int}, Nothing}=nothing` - Fourier order(s) per seasonal period.
  Controls complexity of the seasonal shape. Higher k captures more complex patterns
  but increases computation. If `nothing`, auto-selected via AIC.
  Must match length of `seasonal_periods` if both are vectors.
- `use_box_cox::Union{Bool, Nothing}=nothing` - Box-Cox variance stabilization.
  `true`/`false` forces selection; `nothing` tries both, selects by AIC.
- `use_trend::Union{Bool, Nothing}=nothing` - Include trend component (ℓ_t + φb_t).
  `nothing` tries both options and selects by AIC.
- `use_damped_trend::Union{Bool, Nothing}=nothing` - Use damped trend (φ < 1).
  Ignored if `use_trend=false`. `nothing` tries both.
- `use_arma_errors::Union{Bool, Nothing}=nothing` - Model residuals with ARMA(p,q).
  Orders auto-selected via AIC when enabled. Defaults to true if nothing.

# Returns
`TbatsTerm` for use in formula specification.

# Examples
```julia
# Auto-select everything
@formula(sales = tbats())

# Weekly data with yearly seasonality (52.18 weeks/year)
@formula(demand = tbats(seasonal_periods=52.18))

# Multiple seasonalities: daily (7) and yearly (365.25)
@formula(sales = tbats(seasonal_periods=[7, 365.25]))

# With explicit Fourier orders (3 for weekly, 10 for yearly)
@formula(sales = tbats(seasonal_periods=[7, 365.25], k=[3, 10]))

# Dual calendar (Gregorian + Hijri)
@formula(sales = tbats(seasonal_periods=[365.25, 354.37]))

# Force Box-Cox and damped trend
@formula(sales = tbats(seasonal_periods=52.18, use_box_cox=true, use_damped_trend=true))

# Full specification
@formula(sales = tbats(seasonal_periods=[7, 365.25], k=[3, 10],
                       use_box_cox=true, use_trend=true,
                       use_damped_trend=false, use_arma_errors=true))
```

# See Also
- [`bats`](@ref) - BATS model (integer seasonal periods only)
- [`TbatsTerm`](@ref) - Term type created by this function
"""
function tbats(; seasonal_periods::Union{Real, Vector{<:Real}, Nothing}=nothing,
                k::Union{Int, Vector{Int}, Nothing}=nothing,
                use_box_cox::Union{Bool, Nothing}=nothing,
                use_trend::Union{Bool, Nothing}=nothing,
                use_damped_trend::Union{Bool, Nothing}=nothing,
                use_arma_errors::Union{Bool, Nothing}=nothing)

    if !isnothing(seasonal_periods)
        if seasonal_periods isa Real
            seasonal_periods > 0 || throw(ArgumentError(
                "seasonal_periods must be positive, got $(seasonal_periods)"))
        else
            all(m -> m > 0, seasonal_periods) || throw(ArgumentError(
                "All seasonal_periods must be positive"))
        end
    end

    if !isnothing(k)
        if k isa Int
            k >= 1 || throw(ArgumentError("k must be >= 1, got $(k)"))
        else
            all(ki -> ki >= 1, k) || throw(ArgumentError("All k values must be >= 1"))
        end
    end

    if !isnothing(k) && !isnothing(seasonal_periods)
        k_len = k isa Int ? 1 : length(k)
        m_len = seasonal_periods isa Real ? 1 : length(seasonal_periods)
        k_len == m_len || throw(ArgumentError(
            "Length of k ($(k_len)) must match length of seasonal_periods ($(m_len))"))
    end

    return TbatsTerm(seasonal_periods, k, use_box_cox, use_trend, use_damped_trend, use_arma_errors)
end


const _THETA_VALID_MODELS = (:STM, :OTM, :DSTM, :DOTM)

"""
    theta(; model=nothing, alpha=nothing, theta_param=nothing,
          decomposition=nothing, nmse=nothing)

Specify a Theta forecasting model in a formula.

The Theta method decomposes a series into theta lines capturing long-term trend
and short-term dynamics, then combines their forecasts.

# Arguments
- `model::Union{Symbol, Nothing}=nothing` - Model variant:
  - `:STM` - Simple Theta (θ=2 fixed, α optimized)
  - `:OTM` - Optimized Theta (both θ and α optimized)
  - `:DSTM` - Dynamic Simple Theta (θ=2, dynamic trend estimation)
  - `:DOTM` - Dynamic Optimized Theta (dynamic + optimized)
  - `nothing` - Auto-select best variant by MSE
- `alpha::Union{Real, Nothing}=nothing` - Smoothing parameter (0 < α < 1).
  `nothing` = optimize automatically
- `theta_param::Union{Real, Nothing}=nothing` - Theta parameter (≥ 1).
  Ignored for STM/DSTM (fixed at 2). `nothing` = optimize for OTM/DOTM
- `decomposition::Union{String, Nothing}=nothing` - Seasonal decomposition:
  - `"multiplicative"` - Multiply seasonal factors (default for positive data)
  - `"additive"` - Add seasonal factors
  - `nothing` - Auto-detect based on data characteristics
- `nmse::Union{Int, Nothing}=nothing` - Steps for multi-step MSE (1-30, default: 3)

# Returns
`ThetaTerm` for use in formula specification.

# Examples
```julia
# Auto-select best model variant
@formula(sales = theta())

# Specific model variant
@formula(demand = theta(model=:OTM))
@formula(demand = theta(model=:STM))

# With fixed smoothing parameter
@formula(sales = theta(model=:OTM, alpha=0.2))

# Force seasonal decomposition type
@formula(sales = theta(decomposition="additive"))

# Full specification
@formula(sales = theta(model=:DOTM, decomposition="multiplicative", nmse=5))
```

# See Also
- [`ThetaTerm`](@ref) - Term type created by this function
"""
function theta(; model::Union{Symbol, Nothing}=nothing,
                alpha::Union{Real, Nothing}=nothing,
                theta_param::Union{Real, Nothing}=nothing,
                decomposition::Union{AbstractString, Symbol, Nothing}=nothing,
                nmse::Union{Int, Nothing}=nothing)

    
    if !isnothing(model)
        model ∈ _THETA_VALID_MODELS || throw(ArgumentError(
            "model must be one of $(_THETA_VALID_MODELS), got :$(model)"))
    end

    
    alpha_f = nothing
    if !isnothing(alpha)
        alpha_f = Float64(alpha)
        (0 < alpha_f < 1) || throw(ArgumentError(
            "alpha must be in (0, 1), got $(alpha)"))
    end

    
    theta_f = nothing
    if !isnothing(theta_param)
        theta_f = Float64(theta_param)
        theta_f >= 1 || throw(ArgumentError(
            "theta_param must be >= 1, got $(theta_param)"))

        
        if model ∈ (:STM, :DSTM)
            @warn "theta_param is ignored for $(model) (fixed at 2.0)"
        end
    end

    decomp_str = nothing
    if !isnothing(decomposition)
        decomp_str = lowercase(String(decomposition))
        decomp_str ∈ ("multiplicative", "additive") || throw(ArgumentError(
            "decomposition must be \"multiplicative\" or \"additive\", got \"$(decomposition)\""))
    end

    if !isnothing(nmse)
        (1 <= nmse <= 30) || throw(ArgumentError(
            "nmse must be between 1 and 30, got $(nmse)"))
    end

    return ThetaTerm(model, alpha_f, theta_f, decomp_str, nmse)
end

const _DIFFUSION_VALID_MODELS = (:Bass, :Gompertz, :GSGompertz, :Weibull)

"""
    diffusion(; model=nothing, m=nothing, p=nothing, q=nothing,
              a=nothing, b=nothing, c=nothing, loss=nothing, cumulative=nothing)

Specify a diffusion forecasting model in a formula.

Diffusion models capture technology adoption and market penetration patterns using
classical S-curve models.

# Arguments
- `model::Union{Symbol, Nothing}=nothing` - Model type:
  - `:Bass` - Bass diffusion (innovation/imitation) - default
  - `:Gompertz` - Gompertz growth curve
  - `:GSGompertz` - Gamma/Shifted Gompertz
  - `:Weibull` - Weibull distribution model
- `m::Union{Real, Nothing}=nothing` - Market potential (nothing = estimate)
- `p::Union{Real, Nothing}=nothing` - Innovation coefficient for Bass (nothing = estimate)
- `q::Union{Real, Nothing}=nothing` - Imitation coefficient for Bass (nothing = estimate)
- `a::Union{Real, Nothing}=nothing` - Parameter a for non-Bass models (nothing = estimate)
- `b::Union{Real, Nothing}=nothing` - Parameter b for non-Bass models (nothing = estimate)
- `c::Union{Real, Nothing}=nothing` - Parameter c for GSGompertz only (nothing = estimate)
- `loss::Union{Int, Nothing}=nothing` - Loss function power (1=MAE, 2=MSE), default 2
- `cumulative::Union{Bool, Nothing}=nothing` - Optimize on cumulative values, default true

# Returns
`DiffusionTerm` for use in formula specification.

# Examples
```julia
# Default Bass model
@formula(adoption = diffusion())

# Specific model type
@formula(adoption = diffusion(model=:Bass))
@formula(adoption = diffusion(model=:Gompertz))
@formula(adoption = diffusion(model=:GSGompertz))
@formula(adoption = diffusion(model=:Weibull))

# Bass with fixed market potential
@formula(adoption = diffusion(model=:Bass, m=10000))

# Bass with fixed innovation coefficient
@formula(adoption = diffusion(model=:Bass, p=0.03))

# Use L1 loss instead of L2
@formula(adoption = diffusion(loss=1))

# Optimize on adoption instead of cumulative
@formula(adoption = diffusion(cumulative=false))
```

# See Also
- [`DiffusionTerm`](@ref) - Term type created by this function
"""
function diffusion(; model::Union{Symbol, Nothing}=nothing,
                    m::Union{Real, Nothing}=nothing,
                    p::Union{Real, Nothing}=nothing,
                    q::Union{Real, Nothing}=nothing,
                    a::Union{Real, Nothing}=nothing,
                    b::Union{Real, Nothing}=nothing,
                    c::Union{Real, Nothing}=nothing,
                    loss::Union{Int, Nothing}=nothing,
                    cumulative::Union{Bool, Nothing}=nothing)

    # Validate model type
    if !isnothing(model)
        model ∈ _DIFFUSION_VALID_MODELS || throw(ArgumentError(
            "model must be one of $(_DIFFUSION_VALID_MODELS), got :$(model)"))
    end

    # Convert and validate parameters
    m_f = isnothing(m) ? nothing : Float64(m)
    if !isnothing(m_f) && m_f <= 0
        throw(ArgumentError("m (market potential) must be positive, got $(m)"))
    end

    p_f = isnothing(p) ? nothing : Float64(p)
    if !isnothing(p_f) && p_f <= 0
        throw(ArgumentError("p (innovation coefficient) must be positive, got $(p)"))
    end

    q_f = isnothing(q) ? nothing : Float64(q)
    if !isnothing(q_f) && q_f <= 0
        throw(ArgumentError("q (imitation coefficient) must be positive, got $(q)"))
    end

    a_f = isnothing(a) ? nothing : Float64(a)
    if !isnothing(a_f) && a_f <= 0
        throw(ArgumentError("a must be positive, got $(a)"))
    end

    b_f = isnothing(b) ? nothing : Float64(b)
    if !isnothing(b_f) && b_f <= 0
        throw(ArgumentError("b must be positive, got $(b)"))
    end

    c_f = isnothing(c) ? nothing : Float64(c)
    if !isnothing(c_f) && c_f <= 0
        throw(ArgumentError("c must be positive, got $(c)"))
    end

    # Validate loss
    if !isnothing(loss) && loss < 1
        throw(ArgumentError("loss must be >= 1, got $(loss)"))
    end

    # Warn about model-specific parameters
    if !isnothing(model)
        if model == :Bass && (!isnothing(a) || !isnothing(b) || !isnothing(c))
            @warn "Parameters a, b, c are ignored for Bass model. Use p and q instead."
        elseif model in (:Gompertz, :Weibull) && (!isnothing(p) || !isnothing(q))
            @warn "Parameters p, q are Bass-specific. Use a, b for $(model) model."
        elseif model in (:Gompertz, :Weibull) && !isnothing(c)
            @warn "Parameter c is only for GSGompertz model."
        elseif model == :GSGompertz && (!isnothing(p) || !isnothing(q))
            @warn "Parameters p, q are Bass-specific. Use a, b, c for GSGompertz model."
        end
    end

    return DiffusionTerm(model, m_f, p_f, q_f, a_f, b_f, c_f, loss, cumulative)
end

"""
    e(code::AbstractString = "Z")

Specify the error component in an ETS model.

- `"A"`: additive error
- `"M"`: multiplicative error
- `"Z"`: automatically select
"""
e(code::AbstractString = "Z") = EtsComponentTerm(:error, _validate_ets_code(:error, code))

"""
    t(code::AbstractString = "Z")

Specify the trend component in an ETS model.

- `"N"`: no trend
- `"A"`: additive trend
- `"M"`: multiplicative trend
- `"Z"`: automatically select
"""
t(code::AbstractString = "Z") = EtsComponentTerm(:trend, _validate_ets_code(:trend, code))

"""
    s(code::AbstractString = "Z")

Specify the seasonal component in an ETS model.

- `"N"`: no seasonality
- `"A"`: additive seasonality
- `"M"`: multiplicative seasonality
- `"Z"`: automatically select
"""
s(code::AbstractString = "Z") = EtsComponentTerm(:seasonal, _validate_ets_code(:seasonal, code))

"""
    drift(v=true)

Control the damping/drift behaviour of an ETS trend component.

- `true`: include a damped trend (φ estimated)
- `false`: forbid damping (standard trend)
- `nothing`/`drift(:auto)`: allow automatic selection
"""
drift() = EtsDriftTerm(true)
drift(flag::Bool) = EtsDriftTerm(flag)
drift(::Nothing) = EtsDriftTerm(nothing)
function drift(mode::Symbol)
    mode === :auto ||
        throw(ArgumentError("Unsupported drift mode ':$(mode)'. Use true, false, nothing, or :auto."))
    return EtsDriftTerm(nothing)
end
function drift(mode::AbstractString)
    upper = uppercase(mode)
    if upper == "AUTO"
        return EtsDriftTerm(nothing)
    elseif upper == "TRUE"
        return EtsDriftTerm(true)
    elseif upper == "FALSE"
        return EtsDriftTerm(false)
    else
        throw(ArgumentError("Unsupported drift mode \"$(mode)\". Use \"auto\", \"true\", or \"false\"."))
    end
end


"""
    ModelFormula

Represents a complete model formula in Durbyn's forecasting grammar.

# Fields
- `target::Symbol` - Target variable name (left-hand side)
- `terms::Vector{AbstractTerm}` - Model specification terms (right-hand side)

# Examples
```julia
# Typically created via the @formula macro:
# @formula(y = p(1,2) + q(2,3))
ModelFormula(:y, AbstractTerm[ArimaOrderTerm(:p, 1, 2), ArimaOrderTerm(:q, 2, 3)])
```
"""
struct ModelFormula
    target::Symbol
    terms::Vector{AbstractTerm}
end


"""
    +(term1::AbstractTerm, term2::AbstractTerm)
    +(term::AbstractTerm, var::Symbol)
    +(var::Symbol, term::AbstractTerm)

Combine grammar terms and/or variable symbols with the + operator.

# Returns
Vector of terms that can be further combined

# Examples
```julia
# ARIMA terms only
p(1,2) + q(2,3)                    # [ArimaOrderTerm(:p,...), ArimaOrderTerm(:q,...)]

# ARIMAX with exogenous variables
p(1,2) + q(2,3) + :temperature     # Adds VarTerm(:temperature)
p(1,2) + :temp + q(2,3)            # Variable can be anywhere
:promotion + p(1,2) + :temperature # Multiple variables

# Complex formulas
p(1,2) + d(1) + q(2,3) + P(0,1) + D(1) + Q(0,1) + :temperature + :promotion
```
"""
# Combine two terms
Base.:+(t1::AbstractTerm, t2::AbstractTerm) = AbstractTerm[t1, t2]

# Add term to existing vector (convert to Vector{AbstractTerm} to allow mixed types)
Base.:+(terms::Vector{<:AbstractTerm}, t::AbstractTerm) = push!(AbstractTerm[terms...], t)
Base.:+(t::AbstractTerm, terms::Vector{<:AbstractTerm}) = pushfirst!(AbstractTerm[terms...], t)

# Combine term with Symbol (create VarTerm)
Base.:+(t::AbstractTerm, s::Symbol) = AbstractTerm[t, VarTerm(s)]
Base.:+(s::Symbol, t::AbstractTerm) = AbstractTerm[VarTerm(s), t]

# Add Symbol to existing vector (convert to Vector{AbstractTerm} to allow mixed types)
Base.:+(terms::Vector{<:AbstractTerm}, s::Symbol) = push!(AbstractTerm[terms...], VarTerm(s))
Base.:+(s::Symbol, terms::Vector{<:AbstractTerm}) = pushfirst!(AbstractTerm[terms...], VarTerm(s))

# Combine two Symbols
Base.:+(s1::Symbol, s2::Symbol) = AbstractTerm[VarTerm(s1), VarTerm(s2)]

# Handle nothing (from d() and D() with no arguments)
# nothing + term → just the term
Base.:+(::Nothing, t::AbstractTerm) = t
Base.:+(t::AbstractTerm, ::Nothing) = t
Base.:+(::Nothing, terms::Vector{<:AbstractTerm}) = terms
Base.:+(terms::Vector{<:AbstractTerm}, ::Nothing) = terms
Base.:+(::Nothing, ::Nothing) = nothing
Base.:+(::Nothing, s::Symbol) = VarTerm(s)
Base.:+(s::Symbol, ::Nothing) = VarTerm(s)

"""
    @formula(expr)

Create a model formula from an expression using the `=` syntax.

# Syntax
```julia
@formula(target = terms)
```

where `terms` can be any combination of ARIMA order terms and variables joined with `+`.

# Examples
```julia
# Single term
formula = @formula(y = p(1, 2))

# Multiple terms
formula = @formula(y = p(1, 2) + q(2, 3))

# SARIMA specification
formula = @formula(sales = p(1, 2) + q(2, 3) + P(0, 1) + Q(0, 1))

# With exogenous variables
formula = @formula(sales = p(1, 2) + q(2, 3) + temperature + promotion)
```
"""
macro formula(expr)
    if expr.head != :(=)
        error("@formula requires an assignment expression: @formula(target = terms)")
    end

    target = expr.args[1]
    rhs = expr.args[2]

    # Validate LHS is a bare symbol
    if !(target isa Symbol)
        if target isa Expr && target.head == :.
            error("@formula LHS must be a bare symbol, not a field access. " *
                  "Use @formula(y = ...) instead of @formula(df.y = ...)")
        elseif target isa QuoteNode || (target isa Expr && target.head == :quote)
            error("@formula LHS must be a bare symbol, not a quoted symbol. " *
                  "Use @formula(y = ...) instead of @formula(:y = ...)")
        else
            error("@formula LHS must be a bare symbol (e.g., y), got $(typeof(target))")
        end
    end

    function process_rhs(ex)
        if ex isa Symbol
            if ex === Symbol(".")
                return :($(esc(:AutoVarTerm))())
            end
            return :($(esc(:VarTerm))($(QuoteNode(ex))))
        elseif ex isa QuoteNode || (ex isa Expr && ex.head == :quote)
            # User wrote :x instead of x - give helpful error
            sym = ex isa QuoteNode ? ex.value : ex.args[1]
            error("@formula RHS: use bare symbol `$(sym)` instead of `:$(sym)`. " *
                  "The macro handles symbol conversion automatically.")
        elseif ex isa Expr && ex.head == :call
            if ex.args[1] == :+
                return Expr(:call, :+, [process_rhs(arg) for arg in ex.args[2:end]]...)
            else
                return esc(ex)
            end
        else
            return esc(ex)
        end
    end

    processed_rhs = process_rhs(rhs)

    result_expr = quote
        local rhs_result = $processed_rhs
        local terms_vec = if rhs_result isa Vector
            rhs_result
        elseif rhs_result === nothing
            $(esc(:AbstractTerm))[]  # Empty vector if only nothing
        else
            [rhs_result]
        end
        $(esc(:ModelFormula))($(QuoteNode(target)), terms_vec)
    end

    return result_expr
end


"""
    compile_arima_formula(formula::ModelFormula) -> Dict{Symbol, Tuple{Int,Int}}

Compile a model formula into ARIMA-specific parameter ranges.

# Arguments
- `formula::ModelFormula` - Formula containing ARIMA order terms

# Returns
Dictionary mapping term symbols to (min, max) tuples:
- `:p => (min_p, max_p)` - Non-seasonal AR
- `:d => (d, d)` - Non-seasonal differencing (typically fixed)
- `:q => (min_q, max_q)` - Non-seasonal MA
- `:P => (min_P, max_P)` - Seasonal AR
- `:D => (D, D)` - Seasonal differencing (typically fixed)
- `:Q => (min_Q, max_Q)` - Seasonal MA

# Errors
- Throws `ArgumentError` if duplicate terms are found
- Throws `ArgumentError` if non-ARIMA terms are present

# Examples
```julia
formula = :y = p(1, 2) + d(1) + q(2, 3)
compiled = compile_arima_formula(formula)
# Returns: Dict(:p => (1, 2), :d => (1, 1), :q => (2, 3))
```
"""
function compile_arima_formula(formula::ModelFormula)
    result = Dict{Symbol, Tuple{Int, Int}}()

    for term in formula.terms
        if !isa(term, ArimaOrderTerm)
            throw(ArgumentError("Expected ArimaOrderTerm, got $(typeof(term)). " *
                              "Only p(), d(), q(), P(), D(), Q() terms are supported for ARIMA models."))
        end

        if haskey(result, term.term)
            throw(ArgumentError("Duplicate term '$(term.term)' in formula. " *
                              "Each of p, d, q, P, D, Q can appear only once."))
        end

        result[term.term] = (term.min, term.max)
    end

    return result
end

"""
    compile_ets_formula(formula::ModelFormula)

Extract ETS components (error, trend, seasonal, drift) from a model formula.

# Returns
Named tuple `(error, trend, seasonal, damped)` with component codes and
optional damping directive.
"""
function compile_ets_formula(formula::ModelFormula)
    error_code = "Z"
    trend_code = "Z"
    seasonal_code = "Z"
    damped::Union{Bool, Nothing} = nothing
    error_set = false
    trend_set = false
    seasonal_set = false

    for term in formula.terms
        if term isa EtsComponentTerm
            if term.component === :error
                error_set &&
                    throw(ArgumentError("Multiple error terms detected in ETS formula."))
                error_code = _validate_ets_code(:error, term.code)
                error_set = true
            elseif term.component === :trend
                trend_set &&
                    throw(ArgumentError("Multiple trend terms detected in ETS formula."))
                trend_code = _validate_ets_code(:trend, term.code)
                trend_set = true
            elseif term.component === :seasonal
                seasonal_set &&
                    throw(ArgumentError("Multiple seasonal terms detected in ETS formula."))
                seasonal_code = _validate_ets_code(:seasonal, term.code)
                seasonal_set = true
            else
                throw(ArgumentError("Unsupported ETS component :$(term.component)."))
            end
        elseif term isa EtsDriftTerm
            !isnothing(damped) &&
                throw(ArgumentError("Multiple drift() directives detected in ETS formula."))
            damped = term.damped
        elseif term isa VarTerm || term isa AutoVarTerm || term isa ArimaOrderTerm
            throw(ArgumentError("ETS formulas cannot include ARIMA terms or exogenous regressors."))
        elseif term === nothing
            continue
        else
            throw(ArgumentError("Unsupported term type $(typeof(term)) in ETS formula."))
        end
    end

    return (error = error_code,
            trend = trend_code,
            seasonal = seasonal_code,
            damped = damped)
end

function _extract_single_term(formula::ModelFormula, ::Type{T}) where {T<:AbstractTerm}
    selected = nothing
    for term in formula.terms
        if term isa T
            isnothing(selected) ||
                throw(ArgumentError("Formula may contain only one $(T) term."))
            selected = term
        elseif term isa AbstractTerm
            # Any other AbstractTerm is incompatible with the requested type
            throw(ArgumentError("Formula term $(term) is not compatible with $(T)."))
        elseif term !== nothing
            throw(ArgumentError("Unsupported term $(term) in formula for $(T)."))
        end
    end
    isnothing(selected) &&
        throw(ArgumentError("Formula must include $(T) to build this specification."))
    return selected
end


function Base.show(io::IO, term::ArimaOrderTerm)
    if term.min == term.max
        print(io, "$(term.term)($(term.min))")
    else
        print(io, "$(term.term)($(term.min), $(term.max))")
    end
end

function Base.show(io::IO, term::VarTerm)
    print(io, "$(term.name)")
end

function Base.show(io::IO, ::AutoVarTerm)
    print(io, "auto()")
end

function Base.show(io::IO, term::EtsComponentTerm)
    component_label = term.component === :error ? "e" :
                      term.component === :trend ? "t" : "s"
    print(io, "$(component_label)(\"$(term.code)\")")
end

function Base.show(io::IO, term::EtsDriftTerm)
    if term.damped === true
        print(io, "drift()")
    elseif term.damped === false
        print(io, "drift(false)")
    else
        print(io, "drift(:auto)")
    end
end

function Base.show(io::IO, ::SesTerm)
    print(io, "ses()")
end

function Base.show(io::IO, term::HoltTerm)
    args = String[]
    if !isnothing(term.damped)
        push!(args, "damped=$(term.damped)")
    end
    if term.exponential
        push!(args, "exponential=true")
    end
    if isempty(args)
        print(io, "holt()")
    else
        print(io, "holt(", join(args, ", "), ")")
    end
end

function Base.show(io::IO, term::HoltWintersTerm)
    args = ["seasonal=\"$(term.seasonal)\""]
    if !isnothing(term.damped)
        push!(args, "damped=$(term.damped)")
    end
    if term.exponential
        push!(args, "exponential=true")
    end
    print(io, "hw(", join(args, ", "), ")")
end

function Base.show(io::IO, term::CrostonTerm)
    args = String[]
    if term.method != "hyndman"
        push!(args, "method=\"$(term.method)\"")
    end
    if !isnothing(term.init_strategy)
        push!(args, "init_strategy=\"$(term.init_strategy)\"")
    end
    if !isnothing(term.number_of_params)
        push!(args, "number_of_params=$(term.number_of_params)")
    end
    if !isnothing(term.cost_metric)
        push!(args, "cost_metric=\"$(term.cost_metric)\"")
    end
    if !isnothing(term.optimize_init)
        push!(args, "optimize_init=$(term.optimize_init)")
    end
    if !isnothing(term.rm_missing)
        push!(args, "rm_missing=$(term.rm_missing)")
    end
    if isempty(args)
        print(io, "croston()")
    else
        print(io, "croston(", join(args, ", "), ")")
    end
end

function Base.show(io::IO, term::ArarTerm)
    args = String[]
    if !isnothing(term.max_ar_depth)
        push!(args, "max_ar_depth=$(term.max_ar_depth)")
    end
    if !isnothing(term.max_lag)
        push!(args, "max_lag=$(term.max_lag)")
    end
    if isempty(args)
        print(io, "arar()")
    else
        print(io, "arar(", join(args, ", "), ")")
    end
end

function Base.show(io::IO, term::BatsTerm)
    args = String[]
    if !isnothing(term.seasonal_periods)
        if term.seasonal_periods isa Vector
            push!(args, "seasonal_periods=[$(join(term.seasonal_periods, ", "))]")
        else
            push!(args, "seasonal_periods=$(term.seasonal_periods)")
        end
    end
    if !isnothing(term.use_box_cox)
        push!(args, "use_box_cox=$(term.use_box_cox)")
    end
    if !isnothing(term.use_trend)
        push!(args, "use_trend=$(term.use_trend)")
    end
    if !isnothing(term.use_damped_trend)
        push!(args, "use_damped_trend=$(term.use_damped_trend)")
    end
    if !isnothing(term.use_arma_errors)
        push!(args, "use_arma_errors=$(term.use_arma_errors)")
    end
    if isempty(args)
        print(io, "bats()")
    else
        print(io, "bats(", join(args, ", "), ")")
    end
end

function Base.show(io::IO, term::TbatsTerm)
    args = String[]
    if !isnothing(term.seasonal_periods)
        if term.seasonal_periods isa Vector
            push!(args, "seasonal_periods=[$(join(term.seasonal_periods, ", "))]")
        else
            push!(args, "seasonal_periods=$(term.seasonal_periods)")
        end
    end
    if !isnothing(term.k)
        if term.k isa Vector
            push!(args, "k=[$(join(term.k, ", "))]")
        else
            push!(args, "k=$(term.k)")
        end
    end
    if !isnothing(term.use_box_cox)
        push!(args, "use_box_cox=$(term.use_box_cox)")
    end
    if !isnothing(term.use_trend)
        push!(args, "use_trend=$(term.use_trend)")
    end
    if !isnothing(term.use_damped_trend)
        push!(args, "use_damped_trend=$(term.use_damped_trend)")
    end
    if !isnothing(term.use_arma_errors)
        push!(args, "use_arma_errors=$(term.use_arma_errors)")
    end
    if isempty(args)
        print(io, "tbats()")
    else
        print(io, "tbats(", join(args, ", "), ")")
    end
end

function Base.show(io::IO, term::ThetaTerm)
    args = String[]
    if !isnothing(term.model_type)
        push!(args, "model=:$(term.model_type)")
    end
    if !isnothing(term.alpha)
        push!(args, "alpha=$(term.alpha)")
    end
    if !isnothing(term.theta)
        push!(args, "theta_param=$(term.theta)")
    end
    if !isnothing(term.decomposition_type)
        push!(args, "decomposition=\"$(term.decomposition_type)\"")
    end
    if !isnothing(term.nmse)
        push!(args, "nmse=$(term.nmse)")
    end
    if isempty(args)
        print(io, "theta()")
    else
        print(io, "theta(", join(args, ", "), ")")
    end
end

function Base.show(io::IO, term::DiffusionTerm)
    args = String[]
    if !isnothing(term.model_type)
        push!(args, "model=:$(term.model_type)")
    end
    if !isnothing(term.m)
        push!(args, "m=$(term.m)")
    end
    if !isnothing(term.p)
        push!(args, "p=$(term.p)")
    end
    if !isnothing(term.q)
        push!(args, "q=$(term.q)")
    end
    if !isnothing(term.a)
        push!(args, "a=$(term.a)")
    end
    if !isnothing(term.b)
        push!(args, "b=$(term.b)")
    end
    if !isnothing(term.c)
        push!(args, "c=$(term.c)")
    end
    if !isnothing(term.loss)
        push!(args, "loss=$(term.loss)")
    end
    if !isnothing(term.cumulative)
        push!(args, "cumulative=$(term.cumulative)")
    end
    if isempty(args)
        print(io, "diffusion()")
    else
        print(io, "diffusion(", join(args, ", "), ")")
    end
end

function Base.show(io::IO, ::NaiveTerm)
    print(io, "naive_term()")
end

function Base.show(io::IO, ::SnaiveTerm)
    print(io, "snaive_term()")
end

function Base.show(io::IO, term::RwTerm)
    if term.drift
        print(io, "rw_term(drift=true)")
    else
        print(io, "rw_term()")
    end
end

function Base.show(io::IO, ::MeanfTerm)
    print(io, "meanf_term()")
end

function Base.show(io::IO, formula::ModelFormula)
    print(io, "$(formula.target) = ")
    for (i, term) in enumerate(formula.terms)
        print(io, term)
        if i < length(formula.terms)
            print(io, " + ")
        end
    end
end
