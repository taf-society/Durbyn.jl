export holt_winters, HoltWinters

struct HoltWinters
    fitted::AbstractArray
    residuals::AbstractArray
    components::Vector{Any}
    x::AbstractArray
    par::Any
    loglik::Union{Float64,Int}
    initstate::AbstractArray
    states::AbstractArray
    state_names::Any
    SSE::Union{Float64,Int}
    sigma2::Union{Float64,Int}
    m::Int
    lambda::Union{Float64,Bool,Nothing}
    biasadj::Bool
    aic::Union{Float64,Int}
    bic::Union{Float64,Int}
    aicc::Union{Float64,Int}
    mse::Union{Float64,Int}
    amse::Union{Float64,Int}
    fit::Any
    method::String
end

function holt_winters(
    y::AbstractArray,
    m::Int;
    seasonal::String = "additive",
    damped::Bool = false,
    initial::String = "optimal",
    exponential::Bool = false,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    gamma::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    lambda::Union{Float64,Bool,Nothing} = nothing,
    biasadj::Bool = false,
    options::NelderMeadOptions = NelderMeadOptions(),
)

    initial = match_arg(initial, ["optimal", "simple"])
    seasonal = match_arg(seasonal, ["additive", "multiplicative"])

    if m <= 1
        throw(ArgumentError("The time series should have frequency greater than 1."))
    end

    if length(y) <= m + 3
        throw(
            ArgumentError("I need at least $(m + 3) observations to estimate seasonality."),
        )
    end

    if seasonal == "additive" && exponential
        throw(
            ArgumentError(
                "Forbidden model combination: additive seasonality with exponential trend.",
            ),
        )
    end

    model_code = ""

    if initial == "optimal" || damped
        if seasonal == "additive"
            model_code = exponential ? "ANA" : "AAA"
        else
            model_code = exponential ? "MMM" : "MAM"
        end

        model = ets_base_model(
            y,
            m,
            model_code,
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            phi = phi,
            damped = damped,
            opt_crit = "mse",
            lambda = lambda,
            biasadj = biasadj,
            options = options,
        )
    else
        model = holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            phi = phi,
            seasonal = seasonal,
            exponential = exponential,
            lambda = lambda,
            biasadj = biasadj,
            options = options,
        )
    end

    method = damped ? "Damped Holt-Winters'" : "Holt-Winters'"
    method *= seasonal == "additive" ? " additive method" : " multiplicative method"
    if exponential
        method *= " with exponential trend"
    end

    if damped && initial == "simple"
        @warn "Damped Holt-Winters' method requires optimal initialization"
    end

    return HoltWinters(
        model.fitted,
        model.residuals,
        model.components,
        model.x,
        model.par,
        model.loglik,
        model.initstate,
        model.states,
        model.state_names,
        model.SSE,
        model.sigma2,
        model.m,
        model.lambda,
        model.biasadj,
        model.aic,
        model.bic,
        model.aicc,
        model.mse,
        model.amse,
        model.fit,
        method,
    )
end

"""
    holt_winters(table, m::Int; kwargs...)

Tables.jl interface for Holt-Winters exponential smoothing.

This function accepts any Tables.jl-compatible table as the time series data to be used 
for Holt-Winters forecasting.

# Arguments
- `table`: Any Tables.jl-compatible table (DataFrame, NamedTuple of columns, etc.)
- `m::Int`: The seasonal period (frequency) of the time series
- `col::Union{String,Symbol,Nothing}`: (optional) The name of the column to use as the time series. 
  If not provided, the first column of the table will be used.
- `kwargs...`: All other keyword arguments are passed to the main `holt_winters` function

# Returns
- `HoltWinters`: A fitted Holt-Winters model
"""
function holt_winters(table, m::Int; col::Union{String,Symbol,Nothing} = nothing, kwargs...)
    if !Tables.istable(table)
        throw(ArgumentError("Input must be a Tables.jl-compatible table"))
    end

    # extract columns from the table
    columns = Tables.columns(table)
    col_names = Tables.columnnames(columns)

    if isempty(col_names)
        throw(ArgumentError("Table must have at least one column"))
    end

    # if col is not specified, use the first column
    if isnothing(col)
        col = first(col_names)
    end
    y = Tables.getcolumn(columns, Symbol(col))

    return holt_winters(collect(y), m; kwargs...)
end
