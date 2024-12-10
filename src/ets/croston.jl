function croston_base(y::AbstractArray, m::Int, alpha::Union{Float64,Bool,Nothing}=nothing)
    y_original = copy(y)
    y = y[y .> 0]

    if length(y) == 0
        # Return non fit model
    end

    tt = diff([0; findall(x .> 0)])

    if length(y) == 1 && length(tt) == 1
        
    end

end

function croston(y::AbstractArray, m::Int, alpha::Union{Float64,Bool,Nothing}=nothing)
    
end