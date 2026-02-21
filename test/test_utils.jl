using Test
using Durbyn
using Statistics

import Durbyn.Utils: is_constant, is_constant_all, dropmissing, ismissingish
import Durbyn.Utils: duplicated, completecases, as_integer, mean2
import Durbyn.Utils: as_vector, ModelFitError, evaluation_metrics
import Durbyn.Utils: NamedMatrix, air_passengers, ausbeer, lynx, sunspots
import Durbyn.Stats: longest_contiguous, handle_missing, check_missing, interpolate_missing
import Durbyn.Stats: MissingMethod, Contiguous, Interpolate, FailMissing

const EPS_SCALAR = 1e-10


@testset "Durbyn.Utils Module Tests" begin

    @testset "is_constant (Vector)" begin

        @testset "Constant vectors" begin
            @test is_constant([1.0, 1.0, 1.0]) == true
            @test is_constant([5, 5, 5, 5, 5]) == true
            @test is_constant([0.0, 0.0]) == true
        end

        @testset "Non-constant vectors" begin
            @test is_constant([1.0, 2.0, 3.0]) == false
            @test is_constant([1.0, 1.0, 1.0, 2.0]) == false
            @test is_constant([0.0, 0.0, 0.001]) == false
        end

        @testset "Empty and single element" begin
            @test is_constant(Float64[]) == true
            @test is_constant([42.0]) == true
        end

        @testset "With missing values" begin
            @test is_constant([missing, missing]) == true
            @test is_constant([5.0, 5.0, missing, 5.0]) == true
            @test is_constant([5.0, 6.0, missing]) == false
        end

        @testset "With NaN values (treated as equal)" begin
            @test is_constant([NaN, NaN, missing]) == true
            @test is_constant([NaN, NaN, NaN]) == true
            @test is_constant([NaN, 1.0, NaN]) == false
        end
    end

    @testset "is_constant (Matrix)" begin

        @testset "Per-column check" begin
            X = [1.0 2.0 3.0;
                 1.0 2.0 4.0;
                 1.0 2.0 5.0]

            result = is_constant(X)

            @test result[1] == true
            @test result[2] == true
            @test result[3] == false
        end

        @testset "All columns constant" begin
            X = [1.0 2.0;
                 1.0 2.0;
                 1.0 2.0]

            @test all(is_constant(X)) == true
        end

        @testset "With NaN columns" begin
            X = [1.0 NaN;
                 1.0 NaN;
                 1.0 NaN]

            result = is_constant(X)
            @test result[1] == true
            @test result[2] == true
        end
    end

    @testset "is_constant_all" begin
        @testset "Vector" begin
            @test is_constant_all([5.0, 5.0, 5.0]) == true
            @test is_constant_all([5.0, 6.0, 5.0]) == false
        end

        @testset "Matrix - all columns constant" begin
            X = [1.0 2.0;
                 1.0 2.0;
                 1.0 2.0]
            @test is_constant_all(X) == true
        end

        @testset "Matrix - not all columns constant" begin
            X = [1.0 2.0;
                 1.0 3.0;
                 1.0 2.0]
            @test is_constant_all(X) == false
        end
    end

    @testset "dropmissing (Vector)" begin

        @testset "No missing values" begin
            x = [1.0, 2.0, 3.0]
            result = dropmissing(x)
            @test result == x
        end

        @testset "With missing values" begin
            x = [1.0, missing, 3.0, missing, 5.0]
            result = dropmissing(x)
            @test result == [1.0, 3.0, 5.0]
        end

        @testset "With NaN values" begin
            x = [1.0, NaN, 3.0, NaN, 5.0]
            result = dropmissing(x)
            @test result == [1.0, 3.0, 5.0]
        end

        @testset "With both missing and NaN" begin
            x = [1.0, missing, NaN, 4.0]
            result = dropmissing(x)
            @test result == [1.0, 4.0]
        end

        @testset "All missing/NaN" begin
            x = [missing, NaN, missing]
            result = dropmissing(x)
            @test isempty(result)
        end
    end

    @testset "dropmissing (Vector, Matrix)" begin

        @testset "Basic removal" begin
            x = [1.0, NaN, 3.0, 4.0, 5.0]
            X = [1.0 2.0;
                 3.0 4.0;
                 5.0 6.0;
                 7.0 8.0;
                 9.0 10.0]

            x_clean, X_clean = dropmissing(x, X)

            @test length(x_clean) == 4
            @test size(X_clean) == (4, 2)
            @test x_clean == [1.0, 3.0, 4.0, 5.0]
        end

        @testset "NaN in matrix" begin
            x = [1.0, 2.0, 3.0]
            X = [1.0 2.0;
                 3.0 NaN;
                 5.0 6.0]

            x_clean, X_clean = dropmissing(x, X)

            @test length(x_clean) == 2
            @test size(X_clean) == (2, 2)
        end

        @testset "All clean" begin
            x = [1.0, 2.0, 3.0]
            X = [1.0 2.0;
                 3.0 4.0;
                 5.0 6.0]

            x_clean, X_clean = dropmissing(x, X)

            @test x_clean == x
            @test X_clean == X
        end
    end

    @testset "ismissingish" begin
        @test ismissingish(missing) == true
        @test ismissingish(NaN) == true
        @test ismissingish(1.0) == false
        @test ismissingish(0.0) == false
        @test ismissingish(Inf) == false
    end

    @testset "duplicated" begin

        @testset "No duplicates" begin
            arr = [1, 2, 3, 4, 5]
            result = duplicated(arr)
            @test result == [false, false, false, false, false]
        end

        @testset "With duplicates" begin
            arr = [1, 2, 1, 3, 2]
            result = duplicated(arr)
            @test result == [false, false, true, false, true]
        end

        @testset "All duplicates" begin
            arr = [1, 1, 1, 1]
            result = duplicated(arr)
            @test result == [false, true, true, true]
        end

        @testset "String vector" begin
            arr = ["a", "b", "a", "c"]
            result = duplicated(arr)
            @test result == [false, false, true, false]
        end
    end

    @testset "completecases" begin
        @testset "No missing" begin
            x = [1.0, 2.0, 3.0]
            result = completecases(x)
            @test all(result)
        end

        @testset "With missing" begin
            x = [1.0, missing, 3.0]
            result = completecases(x)
            @test result == [true, false, true]
        end

        @testset "With NaN" begin
            x = [1.0, NaN, 3.0]
            result = completecases(x)
            @test result == [true, false, true]
        end
    end

    @testset "as_integer" begin
        @testset "Float vector" begin
            x = [1.5, 2.7, 3.9]
            result = as_integer(x)
            @test result == [1, 2, 3]
        end

        @testset "Int vector (passthrough)" begin
            x = [1, 2, 3]
            result = as_integer(x)
            @test result === x
        end

        @testset "Single float" begin
            @test as_integer(3.7) == 3
            @test as_integer(3.2) == 3
        end

        @testset "Single int (passthrough)" begin
            @test as_integer(5) === 5
        end
    end

    @testset "mean2 (with missing handling)" begin
        @testset "No missing" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = mean2(x)
            @test abs(result - 3.0) <= EPS_SCALAR
        end

        @testset "With missing, skipmissing=true" begin
            x = [1.0, 2.0, missing, 4.0, 5.0]
            result = mean2(x; skipmissing=true)
            @test abs(result - 3.0) <= EPS_SCALAR
        end

        @testset "With missing, skipmissing=true (repeated)" begin
            x = [1.0, 2.0, missing, 4.0, 5.0]
            result = mean2(x; skipmissing=true)
            @test abs(result - 3.0) <= EPS_SCALAR
        end
    end

    @testset "longest_contiguous" begin
        @testset "No missing" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = longest_contiguous(x)
            @test result == x
        end

        @testset "Missing at start" begin
            x = [missing, missing, 1.0, 2.0, 3.0]
            result = longest_contiguous(x)
            @test length(result) == 3
            @test result == [1.0, 2.0, 3.0]
        end

        @testset "Missing at end" begin
            x = [1.0, 2.0, 3.0, missing, missing]
            result = longest_contiguous(x)
            @test length(result) == 3
        end

        @testset "All missing should error" begin
            x = [missing, missing, missing]
            @test_throws ErrorException longest_contiguous(x)
        end
    end

    @testset "handle_missing dispatcher" begin
        x = [missing, 1.0, 2.0, 3.0, missing]

        @testset "Contiguous" begin
            result = handle_missing(x, Contiguous())
            @test !any(ismissing.(result))
        end
    end

    @testset "as_vector" begin
        @testset "Single row matrix" begin
            X = reshape([1.0, 2.0, 3.0], 1, 3)
            result = as_vector(X)
            @test result == [1.0, 2.0, 3.0]
            @test result isa Vector
        end

        @testset "Single column matrix" begin
            X = reshape([1.0, 2.0, 3.0], 3, 1)
            result = as_vector(X)
            @test result == [1.0, 2.0, 3.0]
            @test result isa Vector
        end

        @testset "Multi-dimensional should error" begin
            X = [1.0 2.0; 3.0 4.0]
            @test_throws ErrorException as_vector(X)
        end
    end

    @testset "ModelFitError" begin
        @testset "Error creation and message" begin
            err = ModelFitError("Test error message")
            @test err.msg == "Test error message"
        end

        @testset "Error throwing" begin
            @test_throws ModelFitError throw(ModelFitError("Test"))
        end
    end

    @testset "NamedMatrix" begin
        @testset "Basic construction" begin
            nm = NamedMatrix(2, ["col1", "col2"]; T=Float64, rownames=["row1", "row2"])

            @test size(nm.data) == (2, 2)
            @test nm.colnames == ["col1", "col2"]
            @test nm.rownames == ["row1", "row2"]
        end

        @testset "Data access" begin
            nm = NamedMatrix(2, ["a", "b"]; T=Float64, rownames=["x", "y"])
            nm.data[1, 1] = 10.0
            nm.data[1, 2] = 20.0
            nm.data[2, 1] = 30.0
            nm.data[2, 2] = 40.0

            @test nm.data[1, 1] == 10.0
            @test nm.data[2, 1] == 30.0
        end
    end

    @testset "Sample Datasets" begin
        @testset "air_passengers" begin
            ap = air_passengers()
            @test length(ap) == 144
            @test ap[1] == 112.0
            @test ap[end] == 432.0
            @test all(ap .> 0)
        end

        @testset "ausbeer" begin
            ab = ausbeer()
            @test length(ab) > 0
            @test all(ab .> 0)
        end

        @testset "lynx" begin
            ly = lynx()
            @test length(ly) > 0
            @test all(ly .>= 0)
        end

        @testset "sunspots" begin
            ss = sunspots()
            @test length(ss) > 0
        end
    end

    @testset "evaluation_metrics" begin
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        pred = [1.1, 2.2, 2.9, 4.1, 4.8]

        result = evaluation_metrics(actual, pred)

        @test haskey(result, "mse")
        @test haskey(result, "mae")
        @test haskey(result, "mar")
        @test haskey(result, "msr")

        @test result["mse"] >= 0
        @test result["mae"] >= 0
    end

    @testset "check_missing via handle_missing" begin
        clean_data = [1.0, 2.0, 3.0]
        @test handle_missing(clean_data, FailMissing()) == clean_data

        missing_data = [1.0, missing, 3.0]
        @test_throws ArgumentError handle_missing(missing_data, FailMissing())
    end

    @testset "interpolate_missing via handle_missing" begin
        x = Union{Float64,Missing}[1.0, missing, 3.0]
        result = handle_missing(x, Interpolate())
        @test !any(ismissing.(result))
        @test result[2] ≈ 2.0  # Interpolated value
    end

end
