function show(io::IO, r::KWFilterResult)
    T = length(r.y)
    println(io, "KWFilterResult")
    println(io, "  Filter type: ", r.filter_type)
    println(io, "  Output:      ", r.output)
    println(io, "  Series length: ", T)
    println(io, "  Integration order (d): ", r.d)
    maxcoef = div(length(r.ideal_coefs) - 1, 2)
    println(io, "  Ideal filter truncation (Q): ", maxcoef)
    if !isempty(r.params)
        println(io, "  Parameters:")
        for (k, v) in r.params
            println(io, "    ", k, " = ", v)
        end
    end
end

