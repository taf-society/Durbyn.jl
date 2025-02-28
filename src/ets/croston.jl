export Croston, croston
struct Croston
    model::Any
    mean::Any
    fitted::Any
    method::Any
    m::Int
end

function croston_fit_fc(y::AbstractArray, m::Int; h::Int, alpha::Union{Float64,Bool,Nothing}=nothing)

    x = copy(y)
    y = [val for val in x if val > 0]

    if isempty(y)
        fc = fill(0.0, h)
        return fc
    end
    
    positions = findall(>(0), x)
    tt = diff(vcat(0, positions))

    if length(y) == 1 && length(tt) == 1
        y_f = fill(y[1], h)
        p_f = fill(tt[1], h)
    elseif length(y) <= 1 || length(tt) <= 1
        return fill(Float64(NaN), h)
    else
        y_f_struct = forecast_ets_base(ses(y, m, initial="simple", alpha=alpha), h=h)
        p_f_struct = forecast_ets_base(ses(tt, m, initial="simple", alpha=alpha), h=h)
        y_f = y_f_struct.mean
        p_f = p_f_struct.mean
    end

    ratio = y_f ./ p_f

    return(ratio, y_f, p_f)
end

function croston_base(y::AbstractArray, m::Int; h::Int, alpha::Union{Float64,Bool,Nothing}=nothing)
    
    n = length(y)
    ratio, y_f, p_f = croston_fit_fc(y, m; h=h, alpha=alpha)

        n = length(y)
        fits = fill(Number(NaN), n)
        if n > 1
            for i in 1:(n-1)
                tmp, y_f1, p_f1 = croston_fit_fc(y[1:i], m; h=1, alpha=alpha)
                fits[i+1] = tmp[1]
            end
        end

    Croston(Dict("demand" => y_f, "period" => p_f), ratio, fits, "Croston Method", m)
end



function croston(y::AbstractArray, m::Int; h::Int, alpha::Union{Float64,Bool,Nothing}=nothing)
    if any(y .< 0)
        @warn "Some negative values found; converting them to 0. 
               This inflates the count of zero-demand periods, which may affect Croston forecasts."
        y[y .< 0] .= 0
    end

    out = croston_base(y, m, h=h, alpha=alpha)

    return out
end