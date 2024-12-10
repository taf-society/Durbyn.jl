function na_interp(x, m; lambda=nothing, linear=nothing)
    if isnothing(linear)
        linear = (m <= 1 || sum(!ismissing.(x)) <= 2 * m)
    end

    println("just a test place holder:", lambda)
end