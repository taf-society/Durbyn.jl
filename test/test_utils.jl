@testset "as_integer" begin
    # Test 1: x is a single floating-point number
    x = 1.0
    @test as_integer(x) == 1

    # Test 2: x is a single integer
    x = 1
    @test as_integer(x) == 1

    # Test 3: x is an array of integers
    x = [1, 2]
    @test as_integer(x) == [1, 2]

    # Test 4: x is an array of floating-point numbers
    x = [1.0, 2.0]
    @test as_integer(x) == [1, 2]
end


@testset "is_constant Tests" begin

    # Test 1: All integers are the same
    @test is_constant([1, 1, 1, 1]) == true

    # Test 2: Integers are not the same
    @test is_constant([1, 2, 1, 1]) == false

    # Test 3: All strings are the same
    @test is_constant(["a", "a", "a"]) == true

    # Test 4: Floats are the same
    @test is_constant([1.0, 1.0, 1.0]) == true

    # Test 5: Floats are not the same
    @test is_constant([1.0, 1.1, 1.0]) == false

    # Test 6: Empty array (edge case)
    @test is_constant([]) == true

    # Test 7: Single-element array (edge case)
    @test is_constant([42]) == true

end

@testset "na_omit Tests" begin

    # Test 1: x contains missing values
    x = [1, 2, missing]
    @test na_omit(x) == [1, 2]

    # Test 2: x has no missing values
    x = [1, 2, 3]
    @test na_omit(x) == [1, 2, 3]

    # Test 3: x is all missing values
    x = [missing, missing]
    @test na_omit(x) == []

    # Test 4: x is an empty array
    x = []
    @test na_omit(x) == []

end

@testset "duplicated Tests" begin

    # Test 1: Array with duplicates
    arr = [1, 2, 3, 2, 4, 5, 1]
    @test duplicated(arr) == [false, false, false, true, false, false, true]

    # Test 2: Array without duplicates
    arr = [1, 2, 3, 4, 5]
    @test duplicated(arr) == [false, false, false, false, false]

    # Test 3: Array where all elements are duplicates
    arr = [1, 1, 1, 1]
    @test duplicated(arr) == [false, true, true, true]

    # Test 4: Empty array
    arr = []
    @test duplicated(arr) == []
end


@testset "match_arg Tests" begin

    # Test 1: Matching an integer
    @test match_arg(2, [1, 2, 3]) == 2

    # Test 2: Matching a string
    @test match_arg("apple", ["banana", "apple", "cherry"]) == "apple"

    # Test 5: Matching a float
    @test match_arg(2.5, [1.0, 2.5, 3.0]) == 2.5
end


@testset "complete_cases Tests" begin

    # Test 1: Array with no missing values
    data = [1, 2, 3, 4]
    @test complete_cases(data) == [true, true, true, true]

    # Test 2: Array with missing values
    data_with_missing = [1, 2, missing, 4]
    @test complete_cases(data_with_missing) == [true, true, false, true]

    # Test 3: DataFrame with no missing values
    df = DataFrame(A = [1, 2, 3], B = [4, 5, 6])
    @test complete_cases(df) == [true, true, true]

    # Test 4: DataFrame with missing values
    df_with_missing = DataFrame(A = [1, missing, 3], B = [4, 5, 6])
    @test complete_cases(df_with_missing) == [true, false, true]

end

@testset "mean2 Tests" begin
    # Test 1: Vector with no missing values
    data = [1.0, 2.0, 3.0, 4.0]
    @test mean2(data, omit_na=true) == 2.5

    # Test 2: Vector with missing values, omit_na=true
    data_with_missing = [1.0, 2.0, missing, 4.0]
    @test mean2(data_with_missing, omit_na=true) ≈ 2.3333333333333335

    # Test 3: Vector with missing values, omit_na=false (should throw an error)
    @test ismissing(mean2(data_with_missing, omit_na=false))

    # Test 4: Empty vector
    data_empty = Float64[]
    @test isnan(mean2(data_empty, omit_na=true))

    # Test 5: All missing values
    data_all_missing = [missing, missing]
    @test_throws ArgumentError mean2(data_all_missing, omit_na=true)
end

@testset "na_contiguous Tests" begin

    # Test 1: Typical case with multiple missing values
    y = [1.0, 2.0, 2.5, missing, 3.0, 4.0, missing, 4.5, missing, 5.0, missing]
    @test na_contiguous(y) == [1.0, 2.0, 2.5]

    # Test 2: Single contiguous stretch
    y = [1.0, 2.0, 3.0, 4.0]
    @test na_contiguous(y) == [1.0, 2.0, 3.0, 4.0]

    # Test 3: Contiguous non-missing values at the end
    y = [missing, missing, 3.0, 4.0]
    @test na_contiguous(y) == [3.0, 4.0]

    # Test 4: Tie for longest contiguous stretch, first one should be returned
    y = [1.0, 2.0, missing, 3.0, 4.0]
    @test na_contiguous(y) == [3.0, 4.0]

end

@testset "na_fail Tests" begin

    # Test 1: Array with no missing values
    data = [1, 2, 3, 4]
    @test na_fail(data) == [1, 2, 3, 4]

    # Test 2: Array with missing values (should throw an error)
    data_with_missing = [1, 2, missing, 4]
    @test_throws ArgumentError na_fail(data_with_missing)

    # Test 3: DataFrame with no missing values
    df = DataFrame(A = [1, 2, 3], B = [4, 5, 6])
    @test na_fail(df) == df

    # Test 4: DataFrame with missing values (should throw an error)
    df_with_missing = DataFrame(A = [1, missing, 3], B = [4, 5, 6])
    @test_throws ArgumentError na_fail(df_with_missing)

end