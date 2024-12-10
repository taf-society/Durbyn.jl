export na_action

"""
    na_action(x::AbstractArray, type="na_contiguous")

    Handle missing data in a vector `x` based on the specified `type` of action.

    # Arguments
    - `x::AbstractArray`: The input vector containing data which may have missing values.
    - `type::Strin`: The type of action to take on the missing data. This should be a `String` and can be one of:
        - `"na_contiguous"`: Handle missing data by treating contiguous blocks of missing values.
        - `"na_interp"`: Handle missing data by interpolation.
       - `"na_fail"`: Handle missing data by failing (raising an error).

    # Returns
    The vector `x` after applying the specified missing data handling action.

    # Example
    ```julia
    x = [1, 2, missing, 4, 5]
    result = na_action(x, "na_interp")
    ```
    """
function na_action(x::AbstractArray, type::String="na_contiguous")
    if type == "na_contiguous"
        return na_contiguous(x)
    elseif type == "na_interp"
        return na_interp(x)
    elseif type == "na_fail"
        return na_fail(x)
    else
        error("Invalid type")
    end
end