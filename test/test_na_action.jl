@testset "na_action Tests" begin
    # Test 1: Call with na_contiguous on an array
    arr = [1.0, 2.0, 3.0, missing, 4.0]
    @test na_action(arr, "na_contiguous") == [1.0, 2.0, 3.0]

    # Test 2: Call with na_interp on an array
    #@test na_action(arr, "na_interp") == "na_interp called"

    # Test 3: Call with na_fail on an array (with missing values)
    @test_throws ArgumentError na_action(arr, "na_fail")

    # Test 4: Call with na_fail on an array (no missing values)
    arr_no_missing = [1.0, 2.0, 3.0, 4.0]
    @test na_action(arr_no_missing, "na_fail") == [1.0, 2.0, 3.0, 4.0]

    # Test 5: Call with invalid type argument
    @test_throws ErrorException na_action(arr, "invalid_type")

    # Test 6: Call with na_contiguous on a DataFrame
    df = DataFrame(A = [1.0, 2.0, missing, 4.0], B = [4.0, 5.0, 6.0, missing])
    @test_throws MethodError na_action(df, "na_contiguous")
end