function ets_model_type_code(x::String)
    ets_model_type = Dict("N" => 0, "A" => 1, "M" => 2)
    return ets_model_type[x]
end
