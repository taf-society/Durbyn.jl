function _difference_pass!(
    destination::Vector{Float64},
    source::Vector{Float64},
    lag_steps::Int,
)
    source_length = length(source)
    @inbounds begin
        for row_index in 1:lag_steps
            destination[row_index] = NaN
        end
        for row_index in (lag_steps + 1):source_length
            destination[row_index] = source[row_index] - source[row_index - lag_steps]
        end
    end
    return destination
end

function _difference_pass!(
    destination::Matrix{Float64},
    source::Matrix{Float64},
    lag_steps::Int,
)
    row_count, column_count = size(source)
    @inbounds begin
        for column_index in 1:column_count
            for row_index in 1:lag_steps
                destination[row_index, column_index] = NaN
            end
            for row_index in (lag_steps + 1):row_count
                destination[row_index, column_index] =
                    source[row_index, column_index] - source[row_index - lag_steps, column_index]
            end
        end
    end
    return destination
end

"""
    diff(series_values::AbstractVector; lag_steps::Int=1, difference_order::Int=1)

Compute lagged differences of a vector.

Each differencing pass applies:
`x_t - x_{t-lag_steps}` and inserts `NaN` where the lagged value is unavailable.
The operation is repeated `difference_order` times.

# Arguments
- `series_values::AbstractVector`: Input data vector.
- `lag_steps::Int=1`: Lag used in each differencing pass.
- `difference_order::Int=1`: Number of recursive differencing passes.

# Returns
A `Vector{Float64}` of the same length as the input when differencing is feasible.
If `lag_steps * difference_order >= length(series_values)`, returns an empty vector.
"""
function diff(series_values::AbstractVector; lag_steps::Int=1, difference_order::Int=1)
    if lag_steps < 1 || difference_order < 1
        throw(ArgumentError("Bad value for 'lag_steps' or 'difference_order'"))
    end
    if lag_steps * difference_order >= length(series_values)
        return series_values[1:0]
    end

    differenced_series = collect(Float64, series_values)
    difference_workspace = similar(differenced_series)
    for _ in 1:difference_order
        _difference_pass!(difference_workspace, differenced_series, lag_steps)
        differenced_series, difference_workspace = difference_workspace, differenced_series
    end

    return differenced_series
end


"""
    diff(series_matrix::AbstractMatrix; lag_steps::Int=1, difference_order::Int=1)

Compute lagged differences column-wise for a matrix.

Each column is differenced independently as a separate series. Undefined leading
positions are filled with `NaN`.

# Arguments
- `series_matrix::AbstractMatrix`: Input matrix (rows = time, columns = variables).
- `lag_steps::Int=1`: Lag used in each differencing pass.
- `difference_order::Int=1`: Number of recursive differencing passes.

# Returns
A `Matrix{Float64}` with the same size as input when differencing is feasible.
If `lag_steps * difference_order >= size(series_matrix, 1)`, returns an empty matrix.
"""
function diff(series_matrix::AbstractMatrix; lag_steps::Int=1, difference_order::Int=1)
    row_count = size(series_matrix, 1)
    if lag_steps < 1 || difference_order < 1
        throw(ArgumentError("Bad value for 'lag_steps' or 'difference_order'"))
    end
    if lag_steps * difference_order >= row_count
        return series_matrix[1:0, :]
    end

    differenced_matrix = Matrix{Float64}(series_matrix)
    workspace_matrix = similar(differenced_matrix)
    for _ in 1:difference_order
        _difference_pass!(workspace_matrix, differenced_matrix, lag_steps)
        differenced_matrix, workspace_matrix = workspace_matrix, differenced_matrix
    end

    return differenced_matrix
end

"""
    diff(named_matrix::NamedMatrix; lag_steps::Int=1, difference_order::Int=1) -> NamedMatrix

Apply lagged differencing to each column of a `NamedMatrix`, preserving row/column names.
"""
function diff(named_matrix::NamedMatrix; lag_steps::Int=1, difference_order::Int=1)
    differenced_data = diff(named_matrix.data; lag_steps=lag_steps, difference_order=difference_order)
    copied_rownames = isnothing(named_matrix.rownames) ? nothing : copy(named_matrix.rownames)
    return NamedMatrix{eltype(differenced_data)}(differenced_data, copied_rownames, copy(named_matrix.colnames))
end
