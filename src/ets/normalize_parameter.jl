function normalize_parameter(param)
    if param === nothing
        return nothing
    elseif param isa Bool
        return param ? 1.0 : 0.0
    elseif isnan(param)
        return nothing
    else
        return param
    end
end