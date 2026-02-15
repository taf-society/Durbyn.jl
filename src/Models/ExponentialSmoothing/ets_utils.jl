export EtsModel, ETS

abstract type ETS end

"""
ETS model output

# Fields
- `fitted::Vector{Float64}`: The fitted values from the ETS model, representing the predicted values at each time point.
- `residuals::Vector{Float64}`: The residuals, which are the differences between the observed values and the fitted values.
- `components::Vector{Any}`: A collection of the model components such as level, trend, and seasonality.
- `x::Vector{Float64}`: The original time series data on which the ETS model was fitted.
- `par::Dict{String, Any}`: A dictionary containing the parameters of the ETS model, where the keys are parameter names and the values are the parameter values.
- `initstate::DataFrame`: A DataFrame containing the initial state estimates of the model.
- `states::DataFrame`: A DataFrame containing the state estimates of the model over time.
- `SSE::Float64`: The sum of squared errors (SSE) of the model, a measure of the model's fit to the data.
- `sigma2::Float64`: The variance of the residuals, indicating the spread of the residuals around zero.
- `m::Int`: The frequency of the seasonal component, e.g., 12 for monthly data with yearly seasonality.
- `lambda::Float64`: The Box-Cox transformation parameter, used if the data were transformed before fitting the model.
- `biasadj::Bool`: A boolean flag indicating whether bias adjustment was applied to the model.
- `loglik::Float64`: Log-likelihood of the model.
- `aic::Float64`: Akaike Information Criterion (AIC) for model selection.
- bic::Float64: Bayesian Information Criterion (BIC) for model selection.
- aicc::Float64: Corrected Akaike Information Criterion (AICc) for small sample sizes.
- mse::Float64:  Mean Squared Error of the model fit.
- amse::Float64:  Average Mean Squared Error, typically used for forecasting accuracy.
- fit::Vector{Float64}: The fitted model.
- `method::String`: The method used for model fitting.

"""

struct EtsModel <: ETS
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

struct EtsRefit
    model::String
    alpha::Union{AbstractFloat,Nothing,Bool}
    beta::Union{AbstractFloat,Nothing,Bool}
    gamma::Union{AbstractFloat,Nothing,Bool}
    phi::Union{AbstractFloat,Nothing,Bool}
end

struct SimpleHoltWinters <: ETS
    SSE::Float64
    fitted::Vector{Float64}
    residuals::Vector{Float64}
    level::Vector{Float64}
    trend::Vector{Float64}
    season::Vector{Float64}
    phi::Float64
end

struct HoltWintersConventional <: ETS
    fitted::AbstractArray
    residuals::AbstractArray
    components::Any
    x::AbstractArray
    par::Dict{String,Any}
    initstate::AbstractArray
    states::AbstractArray
    state_names::Vector{String}
    SSE::Float64
    sigma2::Float64
    m::Int
    lambda::Union{Nothing,Float64}
    biasadj::Bool
    method::String
end

@inline function ets_model_type_code(x::AbstractString)
    if x == "N"
        return 0
    elseif x == "A"
        return 1
    elseif x == "M"
        return 2
    end
    throw(ArgumentError("Unknown ETS model type: $x"))
end

@inline function ets_model_type_code(x::Char)
    if x == 'N'
        return 0
    elseif x == 'A'
        return 1
    elseif x == 'M'
        return 2
    end
    throw(ArgumentError("Unknown ETS model type: $x"))
end

struct ETSWorkspace
    x::Vector{Float64}
    e::Vector{Float64}
    amse::Vector{Float64}
    olds::Vector{Float64}
    s::Vector{Float64}
    f::Vector{Float64}
    denom::Vector{Float64}
    init_state::Vector{Float64}
end

function ETSWorkspace(n::Int, m::Int, nmse::Int, max_state_len::Int)
    seasonal_len = max(24, m)
    amse_len = max(nmse, 30)
    return ETSWorkspace(
        zeros(Float64, max_state_len * (n + 1)),
        zeros(Float64, n),
        zeros(Float64, amse_len),
        zeros(Float64, seasonal_len),
        zeros(Float64, seasonal_len),
        zeros(Float64, 30),
        zeros(Float64, 30),
        zeros(Float64, max_state_len),
    )
end

# Integer codes for opt_crit (eliminates string comparison in hot path)
const OPT_CRIT_LIK = 0
const OPT_CRIT_MSE = 1
const OPT_CRIT_AMSE = 2
const OPT_CRIT_SIGMA = 3
const OPT_CRIT_MAE = 4

@inline function opt_crit_code(s::AbstractString)
    s == "lik" && return OPT_CRIT_LIK
    s == "mse" && return OPT_CRIT_MSE
    s == "amse" && return OPT_CRIT_AMSE
    s == "sigma" && return OPT_CRIT_SIGMA
    s == "mae" && return OPT_CRIT_MAE
    throw(ArgumentError("Unknown optimization criterion: $s"))
end

# Integer codes for bounds
const BOUNDS_BOTH = 0
const BOUNDS_USUAL = 1
const BOUNDS_ADMISSIBLE = 2

@inline function bounds_code(s::AbstractString)
    s == "both" && return BOUNDS_BOTH
    s == "usual" && return BOUNDS_USUAL
    s == "admissible" && return BOUNDS_ADMISSIBLE
    throw(ArgumentError("Unknown bounds type: $s"))
end

function normalize_parameter(param)
    if isnothing(param)
        return nothing
    elseif param isa Bool
        return param ? 1.0 : 0.0
    elseif isnan(param)
        return nothing
    else
        return param
    end
end

function construct_states(
    level::AbstractArray,
    trend::AbstractArray,
    season::AbstractArray,
    trendtype::String,
    seasontype::String,
    m::Int,
)

    states = hcat(level)
    state_names = ["l"]

    if trendtype != "N"
        states = hcat(states, trend)
        push!(state_names, "b")
    end

    if seasontype != "N"
        nr = size(states, 1)
        @inbounds for i = 1:m

            seasonal_column = season[(m-i).+(1:nr)]
            states = hcat(states, seasonal_column)
        end

        seas_names = ["s$i" for i = 1:m]
        append!(state_names, seas_names)
    end

    initstates = states[1, :]

    return states, state_names, initstates
end

function ets_base(y, n, x, m, error, trend, season, alpha, beta, gamma, phi, e, amse, nmse)
    olds = zeros(Float64, max(24, m))
    s = zeros(Float64, max(24, m))
    f = zeros(Float64, 30)
    denom = zeros(Float64, 30)
    return ets_base(
        y,
        n,
        x,
        m,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        e,
        amse,
        nmse,
        olds,
        s,
        f,
        denom,
    )
end

function ets_base(
    y,
    n,
    x,
    m,
    error,
    trend,
    season,
    alpha,
    beta,
    gamma,
    phi,
    e,
    amse,
    nmse,
    olds::AbstractVector{Float64},
    s::AbstractVector{Float64},
    f::AbstractVector{Float64},
    denom::AbstractVector{Float64},
)
    oldb = 0.0

    if m < 1
        m = 1
    end
    nmse_cap = min(nmse, 30)

    nstates = m * (season > 0) + 1 + (trend > 0)
    trend_offset = trend > 0 ? 1 : 0

    # Copy initial state components
    l = x[1]
    if trend > 0
        b = x[2]
    else
        b = 0.0
    end

    if season > 0
        @inbounds for j = 1:m
            s[j] = x[trend_offset+j+1]
        end
    end

    lik = 0.0
    lik2 = 0.0
    @inbounds for j = 1:nmse_cap
        amse[j] = 0.0
        denom[j] = 0.0
    end

    @inbounds for i = 1:n
        # Copy previous state
        oldl = l
        if trend > 0
            oldb = b
        end
        if season > 0
            for j = 1:m
                olds[j] = s[j]
            end
        end

        # One step forecast
        forecast_ets_base(oldl, oldb, olds, m, trend, season, phi, f, nmse_cap)

        f1 = f[1]
        if abs(f1 - -99999.0) < 1.0e-10
            lik = -99999.0
            return lik
        end

        if error == 1
            e[i] = y[i] - f1
        else
            if abs(f1) < 1.0e-10
                f_0 = f1 + 1.0e-10
            else
                f_0 = f1
            end
            e[i] = (y[i] - f1) / f_0
        end

        for j = 1:nmse_cap
            if (i + j - 1) <= n
                denom[j] += 1.0
                tmp = y[i+j-1] - f[j]
                amse[j] = (amse[j] * (denom[j] - 1.0) + (tmp * tmp)) / denom[j]
            end
        end

        # Update state
        l, b, s = update_ets_base(
            oldl,
            l,
            oldb,
            b,
            olds,
            s,
            m,
            trend,
            season,
            alpha,
            beta,
            gamma,
            phi,
            y[i],
        )

        # Store new state
        x[nstates*i+1] = l
        if trend > 0
            x[nstates*i+2] = b
        end
        if season > 0
            for j = 1:m
                x[nstates*i+trend_offset+j+1] = s[j]
            end
        end

        lik += e[i] * e[i]
        val = abs(f1)
        if val > 0.0
            lik2 += log(val)
        else
            lik2 += log(val + 1e-8)
        end
    end

    if lik > 0.0
        lik = n * log(lik)
    else
        lik = n * log(lik + 1e-8)
    end

    if error == 2
        lik += 2 * lik2
    end

    return lik
end

function forecast_ets_base(l, b, s, m, trend, season, phi, f, h)
    TOL = 1.0e-10
    phistar = phi
    @inbounds for i = 1:h
        if trend == 0
            f[i] = l
        elseif trend == 1
            f[i] = l + phistar * b
        elseif b < 0
            f[i] = -99999.0
        else
            f[i] = l * (b ^ phistar)
        end

        j = mod1(m - i + 1, m)

        if season == 1
            f[i] += s[j]
        elseif season == 2
            f[i] *= s[j]
        end

        if i < h
            if abs(phi - 1.0) < TOL
                phistar += 1.0
            else
                phistar += phi^(i + 1)
            end
        end
    end
end

function update_ets_base(
    oldl,
    l,
    oldb,
    b,
    olds,
    s,
    m,
    trend,
    season,
    alpha,
    beta,
    gamma,
    phi,
    y,
)
    # New Level
    if trend == 0
        q = oldl
        phib = 0
    elseif trend == 1
        phib = phi * oldb
        q = oldl + phib
    elseif abs(phi - 1.0) < 1.0e-10
        phib = oldb
        q = oldl * oldb
    else
        phib = oldb^phi
        q = oldl * phib
    end

    # Season
    if season == 0
        p = y
    elseif season == 1
        p = y - olds[m]
    else
        if abs(olds[m]) < 1.0e-10
            p = 1.0e10
        else
            p = y / olds[m]
        end
    end

    l = q + alpha * (p - q)

    # New Growth
    if trend > 0
        if trend == 1
            r = l - oldl
        else
            if abs(oldl) < 1.0e-10
                r = 1.0e10
            else
                r = l / oldl
            end
        end
        b = phib + (beta / alpha) * (r - phib)
    end

    # New Seasonal
    if season > 0
        if season == 1
            t = y - q
        else # if season == 2
            if abs(q) < 1.0e-10
                t = 1.0e10
            else
                t = y / q
            end
        end
        @inbounds s[1] = olds[m] + gamma * (t - olds[m]) # s[t] = s[t - m] + gamma * (t - s[t - m])
        @inbounds for j = 2:m
            s[j] = olds[j-1] # s[t] = s[t]
        end
    end

    return l, b, s
end

function simulate_ets_base(x, m, error, trend, season, alpha, beta, gamma, phi, h, y, e)
    oldb = 0.0
    olds = zeros(24)
    s = zeros(24)
    f = zeros(10)

    if m > 24 && season > 0
        return
    elseif m < 1
        m = 1
    end

    l = x[1]
    b = 0.0
    if trend > 0
        b = x[2]
    end

    if season > 0
        @inbounds for j = 1:m
            s[j] = x[(trend>0)+j+1]
        end
    end

    @inbounds for i = 1:h
        oldl = l
        if trend > 0
            oldb = b
        end
        if season > 0
            for j = 1:m
                olds[j] = s[j]
            end
        end

        forecast_ets_base(oldl, oldb, olds, m, trend, season, phi, f, 1)

        if abs(f[1] - -99999.0) < 1.0e-10
            y[1] = -99999.0
            return
        end

        if error == 1
            y[i] = f[1] + e[i]
        else
            y[i] = f[1] * (1.0 + e[i])
        end

        # Update state
        l, b, s = update_ets_base(
            oldl,
            l,
            oldb,
            b,
            olds,
            s,
            m,
            trend,
            season,
            alpha,
            beta,
            gamma,
            phi,
            y[i],
        )
    end
end

function forecast(
    x::AbstractVector,
    m::Int,
    trend::Int,
    season::Int,
    phi::Float64,
    h::Int,
    f::AbstractVector,
)

    if (m > 24) && (season > 0)
        return
    elseif m < 1
        m = 1
    end

    l = Float64(x[1])
    b = trend > 0 ? Float64(x[2]) : 0.0
    s = zeros(Float64, 24)

    if season > 0
        offset = trend > 0 ? 2 : 1
        @inbounds for j = 1:m
            s[j] = Float64(x[offset+j])
        end
    end

    forecast_ets_base(l, b, s, m, trend, season, phi, f, h)
end

function initparam(
    alpha::Union{Float64,Bool,Nothing},
    beta::Union{Float64,Bool,Nothing},
    gamma::Union{Float64,Bool,Nothing},
    phi::Union{Float64,Bool,Nothing},
    trendtype::String,
    seasontype::String,
    damped::Bool,
    lower::Vector{Float64},
    upper::Vector{Float64},
    m::Int,
    bounds::String;
    nothing_as_nan::Bool = false,)


    if bounds == "admissible"
        lower[1] = 0.0; lower[2] = 0.0; lower[3] = 0.0
        upper[1] = 1e-3; upper[2] = 1e-3; upper[3] = 1e-3
    elseif any(lower .> upper)
        throw(ArgumentError("Inconsistent parameter boundaries"))
    end

    # Select alpha
    if isnothing(alpha)
        m_eff = (seasontype == "N") ? 1 : m
        alpha = lower[1] + 0.2 * (upper[1] - lower[1]) / m_eff
        if alpha > 1 || alpha < 0
            alpha = lower[1] + 2e-3
        end
    end
    # Select beta
    if trendtype != "N" && (isnothing(beta))
        upper[2] = min(upper[2], alpha)
        beta = lower[2] + 0.1 * (upper[2] - lower[2])
        if beta < 0 || beta > alpha
            beta = alpha - 1e-3
        end
    end

    # Select gamma
    if seasontype != "N" && (isnothing(gamma))
        upper[3] = min(upper[3], 1 - alpha)
        gamma = lower[3] + 0.05 * (upper[3] - lower[3])
        if gamma < 0 || gamma > 1 - alpha
            gamma = 1 - alpha - 1e-3
        end
    end

    # Select phi
    if damped && isnothing(phi)
        phi = lower[4] + 0.99 * (upper[4] - lower[4])
        if phi < 0 || phi > 1
            phi = upper[4] - 1e-3
        end
    end

    if nothing_as_nan
        if isnothing(alpha)
            alpha = NaN
        end
        if isnothing(beta)
            beta = NaN
        end
        if isnothing(gamma)
            gamma = NaN
        end
        if isnothing(phi)
            phi = NaN
        end
    end

    return (alpha=alpha, beta=beta, gamma=gamma, phi=phi)
end

function admissible(
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    m::Int,
)

    alpha = normalize_parameter(alpha)
    beta = normalize_parameter(beta)
    gamma = normalize_parameter(gamma)
    phi = normalize_parameter(phi)

    if isnothing(phi)
        phi = 1.0
    elseif phi isa Bool
        phi = phi ? 1.0 : 0.0
    end

    if phi < 0 || phi > 1 + 1e-8
        return false
    end

    if isnothing(gamma)
        if alpha isa Bool
            alpha = alpha ? 1.0 : 0.0
        end
        if alpha < 1 - 1 / phi || alpha > 1 + 1 / phi
            return false
        end

        if !isnothing(beta)
            if beta isa Bool
                beta = beta ? 1.0 : 0.0
            end
            if beta < alpha * (phi - 1) || beta > (1 + phi) * (2 - alpha)
                return false
            end
        end
    elseif m > 1  # Seasonal model
        if isnothing(beta)
            beta = 0.0
        elseif beta isa Bool
            beta = beta ? 1.0 : 0.0
        end

        if gamma isa Bool
            gamma = gamma ? 1.0 : 0.0
        end
        if gamma < max(1 - 1 / phi - alpha, 0.0) || gamma > 1 + 1 / phi - alpha
            return false
        end

        if alpha isa Bool
            alpha = alpha ? 1.0 : 0.0
        end
        if alpha < 1 - 1 / phi - gamma * (1 - m + phi + phi * m) / (2 * phi * m)
            return false
        end

        if beta < -(1 - phi) * (gamma / m + alpha)
            return false
        end

        a = phi * (1 - alpha - gamma)
        b = alpha + beta - alpha * phi + gamma - 1
        c = repeat([alpha + beta - alpha * phi], m - 2)
        d = alpha + beta - phi
        P = vcat([a, b], c, [d, 1])

        poly = Polynomial(P)
        poly_roots = roots(poly)

        if maximum(abs.(poly_roots)) > 1 + 1e-10
            return false
        end
    end

    # Passed all tests
    return true
end

function check_param(
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    lower::Vector{Float64},
    upper::Vector{Float64},
    bounds::String,
    m::Int,
)

    alpha = normalize_parameter(alpha)
    beta = normalize_parameter(beta)
    gamma = normalize_parameter(gamma)
    phi = normalize_parameter(phi)

    if bounds != "admissible"
        if !isnothing(alpha) && !isnan(alpha)
            if alpha < lower[1] || alpha > upper[1]
                return false
            end
        end
        if !isnothing(beta) && !isnan(beta)
            if beta < lower[2] || beta > alpha || beta > upper[2]
                return false
            end
        end
        if !isnothing(phi) && !isnan(phi)
            if phi < lower[4] || phi > upper[4]
                return false
            end
        end
        if !isnothing(gamma) && !isnan(gamma)
            if gamma < lower[3] || gamma > 1 - alpha || gamma > upper[3]
                return false
            end
        end
    end

    if bounds != "usual"
        if !admissible(alpha, beta, gamma, phi, m)
            return false
        end
    end
    return true
end

# Hot-path overload using int bounds code (avoids string comparison per NM iteration)
function check_param(
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    lower::Vector{Float64},
    upper::Vector{Float64},
    bounds::Int,
    m::Int,
)
    alpha = normalize_parameter(alpha)
    beta = normalize_parameter(beta)
    gamma = normalize_parameter(gamma)
    phi = normalize_parameter(phi)

    if bounds != BOUNDS_ADMISSIBLE
        if !isnothing(alpha) && !isnan(alpha)
            if alpha < lower[1] || alpha > upper[1]
                return false
            end
        end
        if !isnothing(beta) && !isnan(beta)
            if beta < lower[2] || beta > alpha || beta > upper[2]
                return false
            end
        end
        if !isnothing(phi) && !isnan(phi)
            if phi < lower[4] || phi > upper[4]
                return false
            end
        end
        if !isnothing(gamma) && !isnan(gamma)
            if gamma < lower[3] || gamma > 1 - alpha || gamma > upper[3]
                return false
            end
        end
    end

    if bounds != BOUNDS_USUAL
        if !admissible(alpha, beta, gamma, phi, m)
            return false
        end
    end
    return true
end

# initialize_states Helpers
function handle_seasonality(y::AbstractVector, m::Integer, seasontype::String)
    n = length(y)
    n < 4 && throw(ArgumentError("y is too short!"))

    if n < 3 * m
        F = fourier(y; m = m, K = 1)
        t = collect(1:n)
        X = hcat(ones(n), t, F)

        β = X \ y
        α, φ = β[1], β[2]

        seasonal = if seasontype == "A"
            y .- (α .+ φ .* t)
        else
            y ./ (α .+ φ .* t)
        end

        return Dict(:seasonal => seasonal)
    else
        decomp = decompose(
            x = y,
            m = m,
            type = seasontype == "A" ? "additive" : "multiplicative",
        )
        return Dict(:seasonal => decomp.seasonal)
    end
end

function initialize_seasonal_components(y_d, m, seasontype)
    seasonal = y_d[:seasonal]
    init_seas = reverse(seasonal[2:m])
    if seasontype != "A"
        init_seas = [max(val, 0.01) for val in init_seas]
        if sum(init_seas) > m
            factor = sum(init_seas .+ 0.01)
            init_seas .= init_seas ./ factor
        end
    end

    return init_seas
end

function adjust_y_sa(y, y_d, seasontype)
    seasonal = y_d[:seasonal]
    if seasontype == "A"
        return y .- seasonal
    else
        return y ./ max.(seasonal, 0.01)
    end
end

function lsfit_ets(x::Matrix{Float64}, y::Vector{Float64})
    n, _ = size(x)
    X = hcat(ones(n), x)
    return X \ y
end

function calculate_initial_values(y_sa::Vector{Float64}, trendtype::String, maxn::Int)

    if trendtype == "N"
        l0 = mean(y_sa[1:maxn])
        b0 = nothing
    else

        x = reshape(collect(1.0:maxn), maxn, 1)
        β = lsfit_ets(x, y_sa[1:maxn])


        if trendtype == "A"
            l0, b0 = β[1], β[2]

            if abs(l0 + b0) < 1e-8
                l0 *= (1 + 1e-3)
                b0 *= (1 - 1e-3)
            end

        else
            l0 = β[1] + β[2]
            if abs(l0) < 1e-8
                l0 = 1e-7
            end

            b0 = (β[1] + 2 * β[2]) / l0
            l0 = l0 / b0

            if abs(b0) > 1e10
                b0 = sign(b0) * 1e10
            end

            if l0 < 1e-8 || b0 < 1e-8
                l0 = max(y_sa[1], 1e-3)
                denom = isapprox(y_sa[1], 0.0; atol = 1e-8) ? y_sa[1] + 1e-10 : y_sa[1]
                b0 = max(y_sa[2] / denom, 1e-3)
            end
        end
    end

    return Dict(:l0 => l0, :b0 => b0)
end

function initialize_states(y, m, trendtype, seasontype)
    if seasontype != "N"
        
        y_d = handle_seasonality(y, m, seasontype)
        init_seas = initialize_seasonal_components(y_d, m, seasontype)
        y_sa = adjust_y_sa(y, y_d, seasontype)
    else
        m = 1
        init_seas = nothing
        y_sa = y
    end

    maxn = min(max(10, 2 * m), length(y_sa))
    initial_values = calculate_initial_values(y_sa, trendtype, maxn)

    l0 = initial_values[:l0]
    b0 = initial_values[:b0]
    out = vcat([l0, b0], init_seas)
    out = [x for x in out if !isnothing(x)]
    return out
end

function calculate_residuals(
    y::AbstractArray,
    m::Int,
    init_state::AbstractArray,
    errortype::String,
    trendtype::String,
    seasontype::String,
    damped::Bool,
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    nmse::Int,
)
    err = ets_model_type_code(errortype)
    trend = ets_model_type_code(trendtype)
    season = ets_model_type_code(seasontype)
    workspace = ETSWorkspace(length(y), m, nmse, length(init_state))
    likelihood, amse, e, x = calculate_residuals(
        y,
        m,
        init_state,
        err,
        trend,
        season,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        nmse,
        workspace,
    )
    return likelihood, copy(amse), copy(e), copy(x)
end

function calculate_residuals!(
    workspace::ETSWorkspace,
    y::AbstractArray,
    m::Int,
    init_state::AbstractArray,
    errortype::Int,
    trendtype::Int,
    seasontype::Int,
    damped::Bool,
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    nmse::Int,
)
    n = length(y)
    p = length(init_state)
    x_len = p * (n + 1)
    if length(workspace.x) < x_len
        throw(ArgumentError("ETSWorkspace.x is too small for state dimension $p and series length $n"))
    end
    if length(workspace.e) < n
        throw(ArgumentError("ETSWorkspace.e is too small for series length $n"))
    end
    if length(workspace.amse) < nmse
        throw(ArgumentError("ETSWorkspace.amse is too small for nmse=$nmse"))
    end

    x = workspace.x
    e = workspace.e
    amse = workspace.amse
    @inbounds x[1:p] .= init_state
    if nmse > 0
        fill!(view(amse, 1:nmse), 0.0)
    end

    if !damped
        phi = 1.0
    end
    if trendtype == 0
        beta = 0.0
    end
    if seasontype == 0
        gamma = 0.0
    end

    alpha_f = Float64(alpha)
    beta_f = Float64(beta)
    gamma_f = Float64(gamma)
    phi_f = Float64(phi)

    likelihood = ets_base(
        y,
        n,
        x,
        m,
        errortype,
        trendtype,
        seasontype,
        alpha_f,
        beta_f,
        gamma_f,
        phi_f,
        e,
        amse,
        nmse,
        workspace.olds,
        workspace.s,
        workspace.f,
        workspace.denom,
    )

    if abs(likelihood + 99999.0) < 1e-7
        likelihood = NaN
    end

    return likelihood, p
end

function calculate_residuals(
    y::AbstractArray,
    m::Int,
    init_state::AbstractArray,
    errortype::Int,
    trendtype::Int,
    seasontype::Int,
    damped::Bool,
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    nmse::Int,
    workspace::ETSWorkspace,
)
    likelihood, p = calculate_residuals!(
        workspace,
        y,
        m,
        init_state,
        errortype,
        trendtype,
        seasontype,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        nmse,
    )
    n = length(y)
    x_len = p * (n + 1)
    x = reshape(view(workspace.x, 1:x_len), p, n + 1)'
    amse = view(workspace.amse, 1:nmse)
    e = view(workspace.e, 1:n)
    return likelihood, amse, e, x
end

function simple_holt_winters(
    x::AbstractArray,
    lenx::Int;
    alpha::Union{Nothing,Float64,Bool} = nothing,
    beta::Union{Nothing,Float64,Bool} = nothing,
    gamma::Union{Nothing,Float64,Bool} = nothing,
    phi::Union{Nothing,Float64,Bool} = nothing,
    seasonal::String = "additive",
    m::Int,
    dotrend::Bool = false,
    doseasonal::Bool = false,
    exponential::Union{Nothing,Bool} = nothing,
    l_start::Union{Nothing,AbstractArray} = nothing,
    b_start::Union{Nothing,AbstractArray} = nothing,
    s_start::Union{Nothing,AbstractArray} = nothing,
)

    if exponential != true || isnothing(exponential)
        exponential = false
    end

    if isnothing(phi) || !isa(phi, Number)
        phi = 1
    end

    # Initialise arrays
    level = zeros(Float64, lenx)
    trend = zeros(Float64, lenx)
    season = zeros(Float64, lenx)
    xfit = zeros(Float64, lenx)
    residuals = zeros(Float64, lenx)
    SSE = 0.0

    if !dotrend
        beta = 0.0
        b_start = 0.0
    end

    if !doseasonal
        gamma = 0.0
        s_start .= ifelse(seasonal == "additive", 0, 1)
    end

    lastlevel = copy(l_start)
    level0 = copy(l_start)
    lasttrend = copy(b_start)
    trend0 = copy(b_start)
    season0 = copy(s_start)

    @inbounds for i = 1:lenx
        if i > 1
            lastlevel = level[i-1]
        end

        if i > 1
            lasttrend = trend[i-1]
        end

        if i > m
            lastseason = season[i-m]
        else
            if i <= length(season0)
                lastseason = season0[i]
            else
                lastseason = nothing
            end
        end

        if isnothing(lastseason)
            lastseason = ifelse(seasonal == "additive", 0, 1)
        end

        if seasonal == "additive"
            if !exponential
                xhat = lastlevel .+ phi .* lasttrend .+ lastseason
            else
                xhat = lastlevel .* lasttrend .^ phi .+ lastseason
            end

        else
            if !exponential
                xhat = (lastlevel .+ phi .* lasttrend) .* lastseason
            else
                xhat = lastlevel .* lasttrend .^ phi .* lastseason
            end

        end

        xfit[i] = xhat[1]
        res = x[i] - xhat[1]
        residuals[i] = res
        SSE += res^2

        if seasonal == "additive"
            if !exponential
                level[i] =
                    (alpha*(x[i]-lastseason).+(1-alpha)*(lastlevel.+phi*lasttrend))[1]
            else
                level[i] =
                    (alpha*(x[i]-lastseason).+(1-alpha)*(lastlevel.*lasttrend .^ phi))[1]
            end
        else
            if !exponential
                level[i] =
                    (alpha*(x[i]/lastseason).+(1-alpha)*(lastlevel.+phi*lasttrend))[1]
            else
                level[i] =
                    (alpha.*(x[i]./lastseason).+(1-alpha).*(lastlevel.*lasttrend .^ phi))[1]
            end
        end

        if !exponential
            trend[i] = (beta.*(level[i].-lastlevel).+(1-beta).*phi.*lasttrend)[1]
        else
            trend[i] = (beta.*(level[i]./lastlevel).+(1-beta).*lasttrend .^ phi)[1]
        end

        if seasonal == "additive"
            if !exponential
                season[i] =
                    (gamma*(x[i].-lastlevel.-phi*lasttrend).+(1-gamma)*lastseason)[1]
            else
                season[i] =
                    (gamma*(x[i].-lastlevel.*lasttrend .^ phi).+(1-gamma)*lastseason)[1]
            end
        else
            if !exponential
                season[i] =
                    (gamma*(x[i]./(lastlevel.+phi*lasttrend)).+(1-gamma)*lastseason)[1]
            else
                season[i] =
                    (gamma.*(x[i]./(lastlevel.*lasttrend .^ phi)).+(1-gamma).*lastseason)[1]
            end
        end
    end

    return SimpleHoltWinters(
        SSE,
        xfit,
        residuals,
        [level0; level],
        [trend0; trend],
        [season0; season],
        phi,
    )
end

function calculate_opt_sse(
    p,
    select::AbstractArray,
    x::AbstractArray,
    lenx::Int,
    alpha::Union{Nothing,Float64,Bool},
    beta::Union{Nothing,Float64,Bool},
    gamma::Union{Nothing,Float64,Bool},
    seasonal::String,
    m::Int,
    exponential::Union{Nothing,Bool},
    phi::Union{Nothing,Float64},
    l_start::Union{Nothing,AbstractArray},
    b_start::Union{Nothing,AbstractArray},
    s_start::Union{Nothing,AbstractArray},
)

    if select[1] > 0
        alpha = p[1]
    end
    if select[2] > 0
        beta = p[1+select[1]]
    end
    if select[3] > 0
        gamma = p[1+select[1]+select[2]]
    end

    dotrend = (!isa(beta, Bool) || beta)
    doseasonal = (!isa(gamma, Bool) || gamma)

    out = simple_holt_winters(
        x,
        lenx,
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        phi = phi,
        seasonal = seasonal,
        m = m,
        dotrend = dotrend,
        doseasonal = doseasonal,
        l_start = l_start,
        exponential = exponential,
        b_start = b_start,
        s_start = s_start,
    )

    return out.SSE
end

function holt_winters_conventional(
    x::AbstractArray,
    m::Int;
    alpha::Union{Nothing,Float64,Bool} = nothing,
    beta::Union{Nothing,Float64,Bool} = nothing,
    gamma::Union{Nothing,Float64,Bool} = nothing,
    phi::Union{Nothing,Float64,Bool} = nothing,
    seasonal::String = "additive",
    exponential::Bool = false,
    lambda::Union{Nothing,Float64} = nothing,
    biasadj::Bool = false,
    warnings::Bool = true,
    options::NelderMeadOptions
)
    if !(seasonal in ["additive", "multiplicative"])
        throw(
            ArgumentError(
                "Invalid seasonal component: must be 'additive' or 'multiplicative'.",
            ),
        )
    end

    origx = copy(x)
    lenx = length(x)

    if (lambda == "auto") || (typeof(lambda) == Float64 && !isnothing(lambda))
        x, lambda = box_cox(x, m, lambda = lambda)
    end

    if isnothing(phi) || !(phi isa Number) || (phi isa Bool)
        phi = 1.0
    end

    if !isnothing(alpha) && !(alpha isa Number)
        throw(
            ArgumentError(
                "Cannot fit models without level ('alpha' must not be 0 or false).",
            ),
        )
    end


    if !all(isnothing.([alpha, beta, gamma])) &&
       any(x -> (!isnothing(x) && (x < 0 || x > 1)), [alpha, beta, gamma])
        throw(
            ArgumentError(
                "'alpha', 'beta', and 'gamma' must be within the unit interval (0, 1).",
            ),
        )
    end

    if (isnothing(gamma) || gamma > 0)
        if seasonal == "multiplicative" && any(x .<= 0)
            throw(ArgumentError("Data must be positive for multiplicative Holt-Winters."))
        end
    end

    if m <= 1
        gamma = false
    end

    # Initialize l0, b0, s0
    if !isnothing(gamma) && gamma isa Bool && !gamma
        seasonal = "none"
        l_start = x[1]
        s_start = 0.0
        if isnothing(beta) || !(beta isa Bool) || beta
            if !exponential
                b_start = x[2] - x[1]
            else
                b_start = x[2] / x[1]
            end
        end
    else
        l_start = mean(x[1:m])
        b_start = (mean(x[m+1:m+m]) - l_start) / m
        if seasonal == "additive"
            s_start = x[1:m] .- l_start
        else
            s_start = x[1:m] ./ l_start
        end
    end

    if !@isdefined(b_start)
        b_start = [nothing]
    end

    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [1.0, 1.0, 1.0, 1.0]

    if !isnothing(beta) && isa(beta, Bool) && !beta
        trendtype = "N"
    elseif exponential
        trendtype = "M"
    else
        trendtype = "A"
    end

    if seasonal == "none"
        seasontype = "N"
    elseif seasonal == "multiplicative"
        seasontype = "M"
    else
        seasontype = "A"
    end

    optim_start = initparam(
        alpha,
        beta,
        gamma,
        1.0,
        trendtype,
        seasontype,
        false,
        lower,
        upper,
        m,
        "usual",
    )

    select = Int.(isnothing.([alpha, beta, gamma]))

    if !isa(s_start, AbstractArray)
        s_start = [s_start]
    end

    if !isa(b_start, AbstractArray)
        b_start = [b_start]
    end

    if !isa(l_start, AbstractArray)
        l_start = [l_start]
    end


    if sum(select) > 0
        starting_points = Float64[]

        if select[1] > 0
            push!(starting_points, optim_start.alpha)
        end

        if select[2] > 0
            push!(starting_points, optim_start.beta)
        end

        if select[3] > 0
            push!(starting_points, optim_start.gamma)
        end

        parscale = max.(abs.(starting_points), 0.1)
        
        cal_opt_sse_closure =
            p -> calculate_opt_sse(
                descaler(p, parscale),
                select,
                x,
                lenx,
                alpha,
                beta,
                gamma,
                seasonal,
                m,
                exponential,
                phi,
                l_start,
                b_start,
                s_start,
            )

        sol = nmmin(cal_opt_sse_closure, scaler(starting_points, parscale), options)

        is_convergence = sol.fail == 0
        minimizers = descaler(sol.x_opt, parscale)

        if (!is_convergence || any((minimizers .< 0) .| (minimizers .> 1))) && warnings
            if sol.fail in [1, 10]
                @warn "Optimization difficulties: convergence code $(sol.fail)"
            elseif sol.fail ∉ [0, 1, 10]
                @warn "Optimization failure: convergence code $(sol.fail), using best parameters found"
            end
        end

        if select[1] > 0
            alpha = minimizers[1]
        end
        if select[2] > 0
            beta = minimizers[1+select[1]]
        end
        if select[3] > 0
            gamma = minimizers[1+select[1]+select[2]]
        end
    end

    final_fit = simple_holt_winters(
        x,
        lenx,
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        phi = phi,
        seasonal = seasonal,
        m = m,
        dotrend = (!isa(beta, Bool) || beta),
        doseasonal = (!isa(gamma, Bool) || gamma),
        l_start = l_start,
        exponential = exponential,
        b_start = b_start,
        s_start = s_start,
    )

    res = final_fit.residuals

    fitted = final_fit.fitted
    if !isnothing(lambda)
        fitted = inv_box_cox(fitted, lambda = lambda, biasadj = biasadj, fvar = var(res))
    end

    states, state_names, initstates = construct_states(
        final_fit.level,
        final_fit.trend,
        final_fit.season,
        trendtype,
        seasontype,
        m,
    )

    damped = phi < 1.0

    if seasonal == "additive"
        components = ["A", trendtype, seasontype, damped]
    elseif seasonal == "multiplicative"
        components = ["M", trendtype, seasontype, damped]
    elseif seasonal == "none" && exponential
        components = ["M", trendtype, seasontype, damped]
    else
        components = ["A", trendtype, seasontype, damped]
    end

    param = Dict("alpha" => alpha)
    param["phi"] = NaN

    if trendtype != "N"
        param["beta"] = beta
    end

    if seasontype != "N"
        param["gamma"] = gamma
    end
    if damped
        param["phi"] = phi
    end

    # Calculate sigma2 based on components
    if components[1] == "A"
        sigma2 = mean(res .^ 2)
    else
        sigma2 = mean((res ./ fitted) .^ 2)
    end

    method_parts = []
    if exponential && trendtype != "N"
        push!(method_parts, "Holt's method with exponential trend")
    elseif trendtype != "N"
        push!(method_parts, "Holt's method")
    else
        push!(method_parts, "Simple exponential smoothing")
    end

    if damped
        push!(method_parts, "Damped")
    end

    if seasonal == "additive"
        push!(method_parts, "additive seasonality")
    elseif seasonal == "multiplicative"
        push!(method_parts, "multiplicative seasonality")
    end

    method = join(method_parts, " with ")

    out = HoltWintersConventional(
        fitted,
        res,
        components,
        origx,
        merge(param, Dict("initstate" => initstates)),
        initstates,
        states,
        state_names,
        final_fit.SSE,
        sigma2,
        m,
        lambda,
        biasadj,
        method,
    )

    return (out)
end

function create_params(
    optimized_params::AbstractArray,
    opt_alpha::Bool,
    opt_beta::Bool,
    opt_gamma::Bool,
    opt_phi::Bool,
)
    j = 1
    pars = Dict()

    if opt_alpha
        pars["alpha"] = optimized_params[j]
        j += 1
    end

    if opt_beta
        pars["beta"] = optimized_params[j]
        j += 1
    end

    if opt_gamma
        pars["gamma"] = optimized_params[j]
        j += 1
    end

    if opt_phi
        pars["phi"] = optimized_params[j]
        j += 1
    end

    result = optimized_params[j:end]
    return merge(pars, Dict("initstate" => result))
end


function objective_fun(
    par,
    y,
    nstate,
    errortype::Int,
    trendtype::Int,
    seasontype::Int,
    damped,
    lower,
    upper,
    opt_crit::Int,
    nmse,
    bounds::Int,
    m,
    init_alpha,
    init_beta,
    init_gamma,
    init_phi,
    opt_alpha,
    opt_beta,
    opt_gamma,
    opt_phi,
    workspace::ETSWorkspace,
)

    j = 1

    alpha = opt_alpha ? par[j] : init_alpha
    j += opt_alpha

    beta = nothing
    if trendtype != 0
        beta = opt_beta ? par[j] : init_beta
        j += opt_beta
    end

    gamma = nothing
    if seasontype != 0
        gamma = opt_gamma ? par[j] : init_gamma
        j += opt_gamma
    end

    phi = nothing
    if damped
        phi = opt_phi ? par[j] : init_phi
        j += opt_phi
    end

    if isnan(alpha)
        throw(ArgumentError("alpha must be numeric"))
    elseif !isnothing(beta) && isnan(beta)
        throw(ArgumentError("beta must be numeric"))
    elseif !isnothing(gamma) && isnan(gamma)
        throw(ArgumentError("gamma must be numeric"))
    elseif !isnothing(phi) && isnan(phi)
        throw(ArgumentError("phi must be numeric"))
    end

    if !check_param(alpha, beta, gamma, phi, lower, upper, bounds, m)
        return Inf
    end

    init_state = view(par, (length(par)-nstate+1):length(par))
    init_state_eval = init_state
    trend_slots = trendtype == 0 ? 0 : 1

    if seasontype != 0
        workspace_init = workspace.init_state
        @inbounds copyto!(workspace_init, 1, init_state, 1, nstate)
        seasonal_start = 2 + trend_slots
        seasonal_sum = sum(view(workspace_init, seasonal_start:nstate))
        workspace_init[nstate+1] = (seasontype == 2 ? m : 0.0) - seasonal_sum
        init_state_eval = view(workspace_init, 1:(nstate+1))
    end

    if seasontype == 2
        if minimum(view(init_state_eval, (2+trend_slots):length(init_state_eval))) < 0.0
            return Inf
        end
    end

    lik, _ = calculate_residuals!(
        workspace,
        y,
        m,
        init_state_eval,
        errortype,
        trendtype,
        seasontype,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        nmse,
    )

    if isnan(lik) || abs(lik + 99999) < 1e-7
        lik = Inf
    elseif lik < -1e10
        # Match forecast::ets behavior: keep very good (near-perfect) fits finite.
        lik = -1e10
    end

    if opt_crit == OPT_CRIT_LIK
        return lik
    elseif opt_crit == OPT_CRIT_MSE
        return workspace.amse[1]
    elseif opt_crit == OPT_CRIT_AMSE
        return sum(view(workspace.amse, 1:nmse)) / nmse
    elseif opt_crit == OPT_CRIT_SIGMA
        return sum(abs2, view(workspace.e, 1:length(y))) / length(y)
    elseif opt_crit == OPT_CRIT_MAE
        return sum(abs, view(workspace.e, 1:length(y))) / length(y)
    else
        error("Unknown optimization criterion")
    end
end

function optim_ets_base(
    opt_params,
    y,
    nstate,
    errortype,
    trendtype,
    seasontype,
    damped,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    m,
    initial_params,
    options)

    init_alpha = initial_params.alpha
    init_beta = initial_params.beta
    init_gamma = initial_params.gamma
    init_phi = initial_params.phi
    opt_alpha = !isnan(init_alpha)
    opt_beta = !isnan(init_beta)
    opt_gamma = !isnan(init_gamma)
    opt_phi = !isnan(init_phi)
    errortype_code = ets_model_type_code(errortype)
    trendtype_code = ets_model_type_code(trendtype)
    seasontype_code = ets_model_type_code(seasontype)
    opt_crit_int = opt_crit_code(opt_crit)
    bounds_int = bounds_code(bounds)
    max_state_len = nstate + (seasontype_code != 0 ? 1 : 0)
    workspace = ETSWorkspace(length(y), m, nmse, max_state_len)

    result = nmmin(par -> objective_fun(
            par,
            y,
            nstate,
            errortype_code,
            trendtype_code,
            seasontype_code,
            damped,
            lower,
            upper,
            opt_crit_int,
            nmse,
            bounds_int,
            m,
            init_alpha,
            init_beta,
            init_gamma,
            init_phi,
            opt_alpha,
            opt_beta,
            opt_gamma,
            opt_phi,
            workspace,
        ), opt_params,
        options)

    optimized_params = result.x_opt
    optimized_value = result.f_opt
    number_of_iterations = result.fncount

    optimized_params =
        create_params(optimized_params, opt_alpha, opt_beta, opt_gamma, opt_phi)

    return Dict(
        "optimized_params" => optimized_params,
        "optimized_value" => optimized_value,
        "number_of_iterations" => number_of_iterations,
    )
end

function etsmodel(
    y::Vector{Float64},
    m::Int,
    errortype::String,
    trendtype::String,
    seasontype::String,
    damped::Bool,
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    lower::Vector{Float64},
    upper::Vector{Float64},
    opt_crit::String,
    nmse::Int,
    bounds::String,
    options::NelderMeadOptions)

    if seasontype == "N"
        m = 1
    end

    if isa(alpha, Bool)
        if alpha
            alpha = 1.0 - 1e-10
        else
            alpha = 0.0
        end
    end

    if isa(beta, Bool)
        if beta
            beta = 1.0
        else
            beta = 0.0
        end
    end

    if isa(gamma, Bool)
        if gamma
            gamma = 1.0
        else
            gamma = 0.0
        end
    end

    if isa(phi, Bool)
        if phi
            phi = 1.0
        else
            phi = 0.0
        end
    end

    if !(isnothing(alpha) || (!isnothing(alpha) && isnan(alpha)))
        upper[2] = min(alpha, upper[2])
        upper[3] = min(1 - alpha, upper[3])
    end


    if !(isnothing(beta) || (!isnothing(beta) && isnan(beta)))
        lower[1] = max(beta, lower[1])
    end

    if !(isnothing(gamma) || (!isnothing(gamma) && isnan(gamma)))
        upper[1] = min(1 - gamma, upper[1])
    end

    par = initparam(
        alpha,
        beta,
        gamma,
        phi,
        trendtype,
        seasontype,
        damped,
        lower,
        upper,
        m,
        bounds,
        nothing_as_nan = true,
    )

    if !isnan(par.alpha)
        alpha = par.alpha
    end

    if !isnan(par.beta)
        beta = par.beta
    end

    if !isnan(par.gamma)
        gamma = par.gamma
    end

    if !isnan(par.phi)
        phi = par.phi
    end

    if !check_param(alpha, beta, gamma, phi, lower, upper, bounds, m)
        damped_str = damped ? "d" : ""
        throw(ArgumentError("For model `ETS($(errortype),\
         $(trendtype)$(damped_str), $(seasontype))` \
         parameters are out of range!"))
    end

    init_state = initialize_states(y, m, trendtype, seasontype)
    nstate = length(init_state)
    initial_params = par
    par = [par.alpha, par.beta, par.gamma, par.phi]
    par = na_omit(par)
    par = vcat(par, init_state)

    lower = vcat(lower, fill(-Inf, nstate))
    upper = vcat(upper, fill(Inf, nstate))

    np = length(par)
    if np >= length(y) - 1
        return Dict(
            :aic => Inf,
            :bic => Inf,
            :aicc => Inf,
            :mse => Inf,
            :amse => Inf,
            :fit => nothing,
            :par => par,
            :states => init_state,
        )
    end

    init_state = nothing

    optimized_fit = optim_ets_base(
        par,
        y,
        nstate,
        errortype,
        trendtype,
        seasontype,
        damped,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        m,
        initial_params,
        options)

    fit_par = optimized_fit["optimized_params"]

    states = fit_par["initstate"]

    if seasontype != "N"
        states =
            vcat(states, m * (seasontype == "M") - sum(states[(2+(trendtype!="N")):nstate]))
    end

    if !isnan(initial_params.alpha)
        alpha = fit_par["alpha"]
    end
    if !isnan(initial_params.beta)
        beta = fit_par["beta"]
    end
    if !isnan(initial_params.gamma)
        gamma = fit_par["gamma"]
    end
    if !isnan(initial_params.phi)
        phi = fit_par["phi"]
    end

    lik, amse, e, states = calculate_residuals(
        y,
        m,
        states,
        errortype,
        trendtype,
        seasontype,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        nmse,
    )

    np += 1
    ny = length(y)
    aic = lik + 2 * np
    bic = lik + log(ny) * np
    aicc = aic + 2 * np * (np + 1) / (ny - np - 1)
    mse = amse[1]
    amse = mean(amse)

    if errortype == "A"
        fits = y .- e
    else
        fits = y ./ (1 .+ e)
    end

    out = Dict(
        "loglik" => -0.5 * lik,
        "aic" => aic,
        "bic" => bic,
        "aicc" => aicc,
        "mse" => mse,
        "amse" => amse,
        "fit" => optimized_fit,
        "residuals" => e,
        "fitted" => fits,
        "states" => states,
        "par" => fit_par,
    )
    return out
end

function process_parameters(
    y,
    m,
    model,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    additive_only,
    lambda,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    ic,
    na_action_type,
)

    opt_crit = match_arg(opt_crit, ["lik", "amse", "mse", "sigma", "mae"])
    bounds = match_arg(bounds, ["both", "usual", "admissible"])
    ic = match_arg(ic, ["aicc", "aic", "bic"])
    na_action_type = match_arg(na_action_type, ["na_contiguous", "na_interp", "na_fail"])

    ny = length(y)
    y = na_action(y, na_action_type)

    if ny != length(y) && na_action_type == "na_contiguous"
        @warn "Missing values encountered. Using longest contiguous portion of time series"
        ny = length(y)
    end

    orig_y = y

    if typeof(model) == ETS && isnothing(lambda)
        lambda = model.lambda
    end

    if !isnothing(lambda)
        y, lambda = box_cox(y, m, lambda = lambda)
        additive_only = true
    end

    if nmse < 1 || nmse > 30
        throw(ArgumentError("nmse out of range"))
    end

    if any(x -> x < 0, upper .- lower)
        throw(ArgumentError("Lower limits must be less than upper limits"))
    end

    return orig_y,
    y,
    lambda,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    additive_only,
    opt_crit,
    bounds,
    ic,
    na_action_type,
    ny
end

function ets_refit(
    y::AbstractArray,
    m::Int,
    model::ETS;
    biasadj::Bool = false,
    use_initial_values::Bool = false,
    nmse::Int = 3,
    kwargs...,
)
    alpha = max(model.par["alpha"], 1e-10)
    beta = get(model.par, "beta", nothing)
    gamma = get(model.par, "gamma", nothing)
    phi = get(model.par, "phi", nothing)

    modelcomponents = string(model.components[1], model.components[2], model.components[3])
    damped = model.components[4]

    if use_initial_values
        errortype = string(modelcomponents[1])
        trendtype = string(modelcomponents[2])
        seasontype = string(modelcomponents[3])

        # Handle both 1D and 2D initstate arrays
        if ndims(model.initstate) == 1
            initstates = Vector(model.initstate)
        else
            initstates = Vector(model.initstate[1, :])
        end

        lik, amse, e, states = calculate_residuals(
            y,
            m,
            initstates,
            errortype,
            trendtype,
            seasontype,
            damped,
            alpha,
            beta,
            gamma,
            phi,
            nmse,
        )

        np = length(model.par) + 1
        loglik = -0.5 * lik
        aic = lik + 2 * np
        bic = lik + log(length(y)) * np
        aicc = model.aic + 2 * np * (np + 1) / (length(y) - np - 1)
        mse = amse[1]
        amse = mean(amse)

        if errortype == "A"
            fitted = y .- e
        else
            fitted = y ./ (1 .+ e)
        end

        sigma2 = sum(e .^ 2) / (length(y) - np)
        x = y

        if biasadj
            model.fitted = inv_box_cox(model.fitted, lambda=model.lambda, biasadj=biasadj, fvar=var(model.residuals))
        end
        model = EtsModel(
            fitted,
            e,
            model.components,
            x,
            model.par,
            loglik,
            model.initstate,
            states,
            nothing,
            model.SSE,
            sigma2,
            m,
            model.lambda,
            biasadj,
            aic,
            bic,
            aicc,
            mse,
            amse,
            model.fit,
            model.method,
        )

        return model
    else
        model = modelcomponents
        @info "Model is being refit with current smoothing parameters but initial states are being re-estimated.\nSet 'use_initial_values=true' if you want to re-use existing initial values."
        return EtsRefit(modelcomponents, alpha, beta, gamma, phi)
    end
end

function validate_and_set_model_params(model, y, m, damped, restrict, additive_only)

    errortype = string(model[1])
    trendtype = string(model[2])
    seasontype = string(model[3])

    if !(errortype in ["M", "A", "Z"])
        throw(ArgumentError("Invalid error type"))
    end
    if !(trendtype in ["N", "A", "M", "Z"])
        throw(ArgumentError("Invalid trend type"))
    end
    if !(seasontype in ["N", "A", "M", "Z"])
        throw(ArgumentError("Invalid season type"))
    end

    if m < 1 || length(y) <= m
        seasontype = "N"
    end
    if m == 1
        if seasontype in ["A", "M"]
            throw(ArgumentError("Nonseasonal data"))
        else
            seasontype = "N"
        end
    end
    if m > 24
        if seasontype in ["A", "M"]
            throw(ArgumentError("Frequency too high"))
        elseif seasontype == "Z"
            @warn "I can't handle data with frequency greater than 24. Seasonality will be ignored. Try stlf() if you need seasonal forecasts."
            seasontype = "N"
        end
    end

    if restrict
        if (errortype == "A" && (trendtype == "M" || seasontype == "M")) ||
           (errortype == "M" && trendtype == "M" && seasontype == "A") ||
           (additive_only && (errortype == "M" || trendtype == "M" || seasontype == "M"))
            throw(ArgumentError("Forbidden model combination"))
        end
    end

    data_positive = minimum(y) > 0
    if !data_positive
        if errortype == "M"
            throw(ArgumentError(
                "Multiplicative error models require strictly positive data. " *
                "Data contains zero or negative values."
            ))
        end
        if trendtype == "M"
            throw(ArgumentError(
                "Multiplicative trend models require strictly positive data. " *
                "Data contains zero or negative values."
            ))
        end
        if seasontype == "M"
            throw(ArgumentError(
                "Multiplicative seasonal models require strictly positive data. " *
                "Data contains zero or negative values."
            ))
        end
    end

    if !isnothing(damped)
        if damped && trendtype == "N"
            throw(
                ArgumentError(
                    "Forbidden model combination: Damped trend with no trend component",
                ),
            )
        end
    end

    n = length(y)
    npars = 2

    if trendtype in ["A", "M"]
        npars += 2
    end
    if seasontype in ["A", "M"]
        npars += m
    end
    if !isnothing(damped)
        npars += damped ? 1 : 0
    end

    return errortype, trendtype, seasontype, npars, data_positive
end

function fit_small_dataset(
    y,
    m,
    alpha,
    beta,
    gamma,
    phi,
    trendtype,
    seasontype,
    lambda,
    biasadj,
    options
)

    if seasontype in ["A", "M"]
        try
            fit = holt_winters_conventional(
                y,
                m,
                alpha = alpha,
                beta = beta,
                gamma = gamma,
                seasonal = (seasontype == "M" ? "multiplicative" : "additive"),
                exponential = (trendtype == "M"),
                phi = phi,
                lambda = lambda,
                biasadj = biasadj,
                warnings = false,
                options = options
            )
            return fit
        catch e
            @warn "Seasonal component could not be estimated: $e"
        end
    end

    if trendtype in ["A", "M"]
        try
            fit = holt_winters_conventional(
                y,
                m,
                alpha = alpha,
                beta = beta,
                gamma = false,
                seasonal = "additive",
                exponential = (trendtype == "M"),
                phi = phi,
                lambda = lambda,
                biasadj = biasadj,
                warnings = false,
                options = options
            )
            return fit
        catch e
            @warn "Trend component could not be estimated: $e"
            return nothing
        end
    end

    if trendtype == "N" && seasontype == "N"
        try
            fit = holt_winters_conventional(
                y,
                m,
                alpha = alpha,
                beta = false,
                gamma = false,
                seasonal = "additive",
                exponential = false,
                phi = nothing,
                lambda = lambda,
                biasadj = biasadj,
                warnings = false,
                options = options
            )
            return fit
        catch e
            @warn "Model without trend and seasonality could not be estimated: $e"
            return nothing
        end
    end

    fit1 = try
        holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = beta,
            gamma = false,
            seasonal = "additive",
            exponential = (trendtype == "M"),
            phi = phi,
            lambda = lambda,
            biasadj = biasadj,
            warnings = false,
            options = options
        )
    catch e
        nothing
    end

    fit2 = try
        holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = false,
            gamma = false,
            seasonal = "additive",
            exponential = (trendtype == "M"),
            phi = phi,
            lambda = lambda,
            biasadj = biasadj,
            warnings = false,
            options = options
        )
    catch e
        nothing
    end

    fit =
        isnothing(fit1) ? fit2 :
        (isnothing(fit2) ? fit1 : (fit1.sigma2 < fit2.sigma2 ? fit1 : fit2))

    if isnothing(fit)
        error("Unable to estimate a model.")
    end

    return fit
end

function get_ic(fit, ic)
    if ic == "aic"
        return fit["aic"]
    elseif ic == "bic"
        return fit["bic"]
    elseif ic == "aicc"
        return fit["aicc"]
    else
        return Inf
    end
end

function generate_ets_grid_fixed(
    errortype::String,
    trendtype::String,
    seasontype::String,
    allow_multiplicative_trend::Bool,
    restrict::Bool,
    additive_only::Bool,
    data_positive::Bool,
    damped::Union{Bool,Nothing},
)
    errortype_vals = (errortype == "Z") ? ["A", "M"] : [errortype]
    trendtype_vals = if trendtype == "Z"
        allow_multiplicative_trend ? ["N", "A", "M"] : ["N", "A"]
    else
        [trendtype]
    end
    seasontype_vals = (seasontype == "Z") ? ["N", "A", "M"] : [seasontype]
    damped_vals = isnothing(damped) ? [true, false] : [damped]

    grid = []

    for e in errortype_vals
        for t in trendtype_vals
            for s in seasontype_vals
                for d in damped_vals
                    if t == "N" && d
                        continue
                    end
                    if restrict
                        if e == "A" && (t == "M" || s == "M")
                            continue
                        end
                        if e == "M" && t == "M" && s == "A"
                            continue
                        end
                        if additive_only && (e == "M" || t == "M" || s == "M")
                            continue
                        end
                    end
                    if !data_positive && e == "M"
                        continue
                    end
                    push!(grid, (e, t, s, d))
                end
            end
        end
    end

    return grid
end

function fit_ets_models(
    grid,
    y,
    m,
    alpha,
    beta,
    gamma,
    phi,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    ic,
    options)

    best_ic = Inf
    best_model = nothing
    best_params = ()

    for combo in grid
        et, t, s, d = combo
        try
            the_fit_model = etsmodel(
                y,
                m,
                et,
                t,
                s,
                d,
                alpha,
                beta,
                gamma,
                phi,
                copy(lower),
                copy(upper),
                opt_crit,
                nmse,
                bounds,
                options)
            fit_ic = get_ic(the_fit_model, ic)
            if fit_ic < best_ic
                best_ic = fit_ic
                best_model = the_fit_model
                best_params = combo
            end
        catch e
            @warn "Error fitting model with combination: $combo" exception=(e, catch_backtrace())
            continue
        end
    end
    return Dict(
        "best_model" => best_model,
        "best_params" => best_params,
        "best_ic" => best_ic,
    )
end

function fit_best_ets_model(
    y,
    m,
    errortype,
    trendtype,
    seasontype,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    ic,
    data_positive;
    restrict = true,
    additive_only = false,
    allow_multiplicative_trend = true,
    options)

    grid = generate_ets_grid_fixed(
        errortype,
        trendtype,
        seasontype,
        allow_multiplicative_trend,
        restrict,
        additive_only,
        data_positive,
        damped,
    )

    result = fit_ets_models(
        grid,
        y,
        m,
        alpha,
        beta,
        gamma,
        phi,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        ic,
        options)

    best_model = result["best_model"]
    best_params = result["best_params"]
    best_ic = result["best_ic"]
    result = nothing

    best_e, best_t, best_s, best_d = best_params

    if best_ic == Inf
        throw(ModelFitError("No model able to be fitted"))
    end

    method = "ETS($(best_e),$(best_t)$(best_d ? "d" : ""),$(best_s))"
    components = [best_e, best_t, best_s, best_d]
    return Dict("model" => best_model, "method" => method, "components" => components)
end


function ets_base_model(
    y::AbstractArray,
    m::Int,
    model;
    damped::Union{Bool,Nothing} = nothing,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    gamma::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    additive_only::Bool = false,
    lambda::Union{Float64,Bool,Nothing,String} = nothing,
    biasadj::Bool = false,
    lower::AbstractArray = [0.0001, 0.0001, 0.0001, 0.8],
    upper::AbstractArray = [0.9999, 0.9999, 0.9999, 0.98],
    opt_crit::String = "lik",
    nmse::Int = 3,
    bounds::String = "both",
    ic::String = "aicc",
    restrict::Bool = true,
    allow_multiplicative_trend::Bool = false,
    use_initial_values::Bool = false,
    na_action_type::String = "na_contiguous",
    options::NelderMeadOptions,
)

    orig_y,
    y,
    lambda,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    additive_only,
    opt_crit,
    bounds,
    ic,
    na_action_type,
    ny = process_parameters(
        y,
        m,
        model,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        additive_only,
        lambda,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        ic,
        na_action_type,
    )

    if model isa ETS
        refit_result = ets_refit(
            y,
            m,
            model,
            biasadj = biasadj,
            use_initial_values = use_initial_values,
            nmse = nmse,
        )
        if refit_result isa ETS
            return refit_result
        end
        # EtsRefit was returned, extract model string for further processing
        model = refit_result.model
    end

    errortype, trendtype, seasontype, npars, data_positive =
        validate_and_set_model_params(model, y, m, damped, restrict, additive_only)

    if ny <= npars + 4

        if !isnothing(damped) && damped
            @warn "Not enough data to use damping"
        end

        return fit_small_dataset(
            orig_y,
            m,
            alpha,
            beta,
            gamma,
            phi,
            trendtype,
            seasontype,
            lambda,
            biasadj,
            options
        )
    end

    model = fit_best_ets_model(
        Float64.(y),
        m,
        errortype,
        trendtype,
        seasontype,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        ic,
        data_positive,
        restrict = restrict,
        additive_only = additive_only,
        allow_multiplicative_trend = allow_multiplicative_trend,
        options = options)

    method = model["method"]
    components = model["components"]
    model = model["model"]
    np = length(model["par"])
    sigma2 = sum(skipmissing(model["residuals"] .^ 2)) / (ny - np)
    SSE = NaN

    if !isnothing(lambda)
        model["fitted"] =
            inv_box_cox(model["fitted"], lambda = lambda, biasadj = biasadj, fvar = sigma2)
    end

    initstates = model["states"][1, :]

    model = EtsModel(
        model["fitted"],
        model["residuals"],
        components,
        orig_y,
        model["par"],
        model["loglik"],
        initstates,
        model["states"],
        ["cff"],
        SSE,
        sigma2,
        m,
        lambda,
        biasadj,
        model["aic"],
        model["bic"],
        model["aicc"],
        model["mse"],
        model["amse"],
        model["fit"],
        method,
    )
    return model
end

function simulate_ets(
    object::ETS,
    nsim::Union{Int,Nothing} = nothing;
    future::Bool = true,
    bootstrap::Bool = false,
    innov::Union{Vector{Float64},Nothing} = nothing,
)

    x = object.x
    m = object.m
    states = object.states
    residuals = object.residuals
    sigma2 = object.sigma2
    components = object.components
    par = object.par
    lambda = object.lambda
    biasadj = object.biasadj

    nsim = isnothing(nsim) ? length(x) : nsim

    if !isnothing(innov)
        nsim = length(innov)
    end

    if !all(ismissing.(x))
        if isnothing(m)
            m = 1
        end
    else
        if nsim == 0
            nsim = 100
        end
        x = [10]
        future = false
    end

    initstate = future ? states[end, :] : states[rand(1:size(states, 1)), :]

    if bootstrap
        res = filter(!ismissing, residuals) .- mean(filter(!ismissing, residuals))
        e = rand(res, nsim)
    elseif isnothing(innov)
        e = rand(Normal(0, sqrt(sigma2)), nsim)
    elseif length(innov) == nsim
        e = innov
    else
        error("Length of innov must be equal to nsim")
    end

    if components[1] == "M"
        e = max.(-1, e)
    end

    y = zeros(nsim)
    errors = ets_model_type_code(components[1])
    trend = ets_model_type_code(components[2])
    season = ets_model_type_code(components[3])
    alpha = check_component(par, "alpha")
    # beta = ifelse(trend == "N", 0.0, check_component(par, "beta"))
    # gamma = ifelse(season == "N", 0.0, check_component(par, "gamma"))
    # phi = ifelse(!components[4], 1.0, check_component(par, "phi"))
    beta = (trend == 0) ? 0.0 : check_component(par, "beta")
    gamma = (season == 0) ? 0.0 : check_component(par, "gamma")
    phi = components[4] ? check_component(par, "phi") : 1.0
    simulate_ets_base(
        initstate,
        m,
        errors,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        nsim,
        y,
        e,
    )

    if abs(y[1] - (-99999.0)) < 1e-7
        error("Problem with multiplicative damped trend")
    end

    if !isnothing(lambda)
        y = inv_box_cox(y, lambda=lambda)
    end

    return y
end
