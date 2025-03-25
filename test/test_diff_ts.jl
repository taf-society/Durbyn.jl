using Test

@testset "diff_ts tests" begin
    # Example 1: First-order difference, lag=1 (vector)
    x1 = [1, 2, 4, 7, 11, 16]
    r1 = diff_ts(x1, lag=1, differences=1)
    @test r1 == [missing, 1, 2, 3, 4, 5]

    # Example 2: Second-order difference, lag=1 (vector)
    r2 = diff_ts(x1, lag=1, differences=2)
    @test r2 == [missing, missing, 1, 1, 1, 1]

    # Example 3: First-order difference, lag=2 (vector)
    r3 = diff_ts(x1, lag=2, differences=1)
    @test r3 == [missing, missing, 3, 5, 7, 9]

    # Example 4: Matrix input
    xmat = [
        1  10;
        2  20;
        4  30;
        7  40;
        11 50;
        16 60
    ]
    r4 = diff_ts(xmat, lag=1, differences=1)
    expected4 = [
        missing  missing;
        1        10;
        2        10;
        3        10;
        4        10;
        5        10
    ]
    @test r4 == expected4
end
