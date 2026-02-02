using Test
using Durbyn
using Statistics

# Access internal functions through the module path
const _compute_accuracy_metrics = Durbyn.Generics._compute_accuracy_metrics
const _index_actual_by_groups = Durbyn.Generics._index_actual_by_groups
const _build_group_index = Durbyn.Generics._build_group_index
const _has_grouping_columns = Durbyn.Generics._has_grouping_columns
const _identify_value_column = Durbyn.Generics._identify_value_column
const _detect_time_column = Durbyn.Generics._detect_time_column
const _detect_shared_time_column = Durbyn.Generics._detect_shared_time_column

const EPS = 1e-6

@testset "Accuracy Module Tests" begin

    @testset "Basic accuracy metrics (vector inputs)" begin
        forecast_vec = [10.0, 20.0, 30.0, 40.0, 50.0]
        actual_vec = [12.0, 18.0, 33.0, 38.0, 52.0]

        acc = accuracy(forecast_vec, actual_vec)

        @test haskey(acc, :ME)
        @test haskey(acc, :RMSE)
        @test haskey(acc, :MAE)
        @test haskey(acc, :MPE)
        @test haskey(acc, :MAPE)
        @test haskey(acc, :ACF1)

        # ME = mean([2, -2, 3, -2, 2]) = 0.6
        errors = actual_vec .- forecast_vec
        @test acc.ME ≈ mean(errors) atol=EPS

        # MAE = mean([2, 2, 3, 2, 2]) = 2.2
        @test acc.MAE ≈ mean(abs.(errors)) atol=EPS

        # RMSE = sqrt(mean([4, 4, 9, 4, 4])) = sqrt(5) ≈ 2.236
        @test acc.RMSE ≈ sqrt(mean(errors.^2)) atol=EPS
    end

    @testset "MASE with training data" begin
        training = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        forecast_vec = [11.0, 12.0, 13.0]
        actual_vec = [11.5, 11.0, 14.0]

        acc = accuracy(forecast_vec, actual_vec; training_data=training)

        @test haskey(acc, :MASE)
        @test !isnothing(acc.MASE)

        # Naive errors = diff(training) = [1,1,1,1,1,1,1,1,1]
        # Scale = mean(abs.(naive_errors)) = 1.0
        # errors = [0.5, -1.0, 1.0]
        # MAE = mean([0.5, 1.0, 1.0]) = 0.833...
        # MASE = MAE / scale = 0.833...
        @test acc.MASE ≈ 0.8333333333333334 atol=EPS
    end

    @testset "Grouped accuracy with multi-column keys" begin
        # Create data with (product, region) grouping
        # This tests Bug Fix #1

        # Create a mock grouped forecast structure
        forecasts = Dict{NamedTuple, Any}()

        # Group 1: product=A, region=North
        key1 = (product = "A", region = "North")
        forecasts[key1] = (mean = [10.0, 20.0, 30.0], method = "test")

        # Group 2: product=A, region=South
        key2 = (product = "A", region = "South")
        forecasts[key2] = (mean = [15.0, 25.0, 35.0], method = "test")

        # Group 3: product=B, region=North
        key3 = (product = "B", region = "North")
        forecasts[key3] = (mean = [100.0, 200.0, 300.0], method = "test")

        fc = (
            forecasts = forecasts,
            groups = [key1, key2, key3]
        )

        # Actual data table with matching multi-column keys
        actual = (
            product = ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
            region = ["North", "North", "North", "South", "South", "South", "North", "North", "North"],
            date = [1, 2, 3, 1, 2, 3, 1, 2, 3],
            value = [12.0, 18.0, 33.0, 14.0, 26.0, 34.0, 95.0, 205.0, 295.0]
        )

        # This should work without "No actual data for group" warnings
        acc = accuracy(fc, actual)

        @test haskey(acc, :product)
        @test haskey(acc, :region)
        @test haskey(acc, :ME)
        @test haskey(acc, :MAE)

        # Should have 3 groups
        @test length(acc.product) == 3
        @test length(acc.region) == 3
    end

    @testset "Grouped accuracy includes MASE" begin
        # This tests Bug Fix #2

        # Create mock grouped forecast
        forecasts = Dict{NamedTuple, Any}()
        key1 = (series = "A",)
        # Include training data in the forecast object
        forecasts[key1] = (
            mean = [10.0, 20.0, 30.0],
            method = "test",
            x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]  # training data
        )

        fc = (
            forecasts = forecasts,
            groups = [key1]
        )

        actual = (
            series = ["A", "A", "A"],
            date = [1, 2, 3],
            value = [12.0, 18.0, 33.0]
        )

        # Pass training data
        training = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        acc = accuracy(fc, actual; training_data=training)

        # MASE should be included when training_data is provided
        @test haskey(acc, :MASE)
    end

    @testset "Grouped accuracy aligns by date" begin
        # This tests Bug Fix #3

        # Create mock grouped forecast
        forecasts = Dict{NamedTuple, Any}()
        key1 = (series = "A",)
        forecasts[key1] = (mean = [10.0, 20.0, 30.0], method = "test")

        fc = (
            forecasts = forecasts,
            groups = [key1]
        )

        # Actual data with REVERSED date order
        actual_reversed = (
            series = ["A", "A", "A"],
            date = [3, 2, 1],  # Reversed order
            value = [33.0, 18.0, 12.0]  # Values match reversed dates
        )

        acc = accuracy(fc, actual_reversed)

        # After alignment by date:
        # date=1 -> actual=12.0, forecast=10.0, error=2.0
        # date=2 -> actual=18.0, forecast=20.0, error=-2.0
        # date=3 -> actual=33.0, forecast=30.0, error=3.0
        # ME = mean([2, -2, 3]) = 1.0

        @test acc.ME[1] ≈ 1.0 atol=EPS
    end

    @testset "Table with :group column treated as grouped" begin
        # This tests Bug Fix #4 and #5

        # Create forecast table with only :group as grouping column
        fc_table = (
            group = ["G1", "G1", "G2", "G2"],
            date = [1, 2, 1, 2],
            mean = [10.0, 20.0, 100.0, 200.0]
        )

        actual_table = (
            group = ["G1", "G1", "G2", "G2"],
            date = [1, 2, 1, 2],
            value = [12.0, 18.0, 105.0, 195.0]
        )

        # _has_grouping_columns should detect :group
        ct = (group = ["G1"], value = [1.0])
        @test _has_grouping_columns(ct) == true

        # accuracy should use grouped calculation
        acc = accuracy(fc_table, actual_table)

        @test haskey(acc, :group)
        @test length(acc.group) == 2
        @test "G1" in acc.group
        @test "G2" in acc.group
    end

    @testset "by parameter warning" begin
        # This tests Bug Fix #5
        forecast_vec = [10.0, 20.0, 30.0]
        actual_vec = [12.0, 18.0, 33.0]

        # Should emit a warning when by parameter is used
        @test_logs (:warn, "`by` parameter is not yet implemented and will be ignored") begin
            accuracy(forecast_vec, actual_vec; by=:series)
        end
    end

    @testset "_build_group_index sorts by date" begin
        # Test the sorting behavior directly
        ct = (
            series = ["A", "A", "A", "B", "B"],
            date = [3, 1, 2, 2, 1],  # Out of order
            value = [30.0, 10.0, 20.0, 200.0, 100.0]
        )

        groups = _build_group_index(ct, [:series])

        # Group A should have indices sorted by date: [2, 3, 1] (dates 1, 2, 3)
        key_a = (series = "A",)
        @test groups[key_a] == [2, 3, 1]

        # Group B should have indices sorted by date: [5, 4] (dates 1, 2)
        key_b = (series = "B",)
        @test groups[key_b] == [5, 4]
    end

    @testset "_index_actual_by_groups with multi-column keys" begin
        ct = (
            product = ["A", "A", "B", "B"],
            region = ["N", "N", "N", "N"],
            date = [2, 1, 2, 1],  # Out of order
            value = [20.0, 10.0, 200.0, 100.0]
        )

        groups = _index_actual_by_groups(ct, :value, [:product, :region])

        # Should have 2 groups
        @test length(groups) == 2

        # Values should be sorted by date within each group
        key_an = (product = "A", region = "N")
        @test groups[key_an] == [10.0, 20.0]  # Sorted by date

        key_bn = (product = "B", region = "N")
        @test groups[key_bn] == [100.0, 200.0]  # Sorted by date
    end

    @testset "Error handling" begin
        @testset "Length mismatch" begin
            forecast_vec = [10.0, 20.0, 30.0]
            actual_vec = [12.0, 18.0]

            @test_throws ErrorException accuracy(forecast_vec, actual_vec)
        end

        @testset "GroupedForecasts requires table" begin
            fc = (
                forecasts = Dict{NamedTuple, Any}(),
                groups = []
            )
            actual_vec = [1.0, 2.0, 3.0]

            @test_throws ErrorException accuracy(fc, actual_vec)
        end
    end

    @testset "Value column identification" begin
        # Test that various column names are recognized
        @test _identify_value_column((value = [1.0],)) == :value
        @test _identify_value_column((y = [1.0],)) == :y
        @test _identify_value_column((actual = [1.0],)) == :actual
        @test _identify_value_column((observed = [1.0],)) == :observed

        # Fallback to first numeric column not in reserved list
        @test _identify_value_column((series = ["A"], sales = [1.0])) == :sales
    end

    @testset "Missing group columns error" begin
        # Test that an explicit error is thrown when required group columns are missing
        ct = (
            series = ["A", "A"],
            date = [1, 2],
            value = [10.0, 20.0]
        )

        # Should error when asking for columns that don't exist
        @test_throws ErrorException _index_actual_by_groups(ct, :value, [:series, :region])
    end

    @testset "Time column mismatch warning" begin
        # Forecast table uses :date, actual table uses :time (no shared column)
        fc_table = (
            group = ["G1", "G1"],
            date = [1, 2],
            mean = [10.0, 20.0]
        )

        actual_table = (
            group = ["G1", "G1"],
            time = [1, 2],  # Different time column name
            value = [12.0, 18.0]
        )

        # Should emit a warning about mismatched time columns
        @test_logs (:warn, r"Forecast table uses :date.*actual table uses :time") begin
            accuracy(fc_table, actual_table)
        end
    end

    @testset "Shared time column detection" begin
        # Both tables have :date and :time, should prefer shared :date
        ct1 = (series = ["A"], date = [1], time = [100])
        ct2 = (series = ["A"], time = [100], step = [1])

        shared, ct1_col, ct2_col = _detect_shared_time_column(ct1, ct2)
        @test shared == :time  # :time is the shared column
        @test ct1_col == :date  # ct1 would use :date on its own
        @test ct2_col == :time  # ct2 would use :time on its own
    end

    @testset "Shared time column aligns tables correctly" begin
        # Tables have different first time columns but share :time
        fc_table = (
            group = ["G1", "G1"],
            date = [1, 2],      # fc would use :date alone
            time = [10, 20],    # shared column
            mean = [10.0, 20.0]
        )

        actual_table = (
            group = ["G1", "G1"],
            time = [20, 10],    # reversed order, shared column
            step = [2, 1],
            value = [18.0, 12.0]  # values correspond to time order
        )

        # Should align on shared :time column, not warn
        # time=10 -> fc mean=10, actual value=12 -> error=2
        # time=20 -> fc mean=20, actual value=18 -> error=-2
        # ME = 0
        acc = accuracy(fc_table, actual_table)
        @test acc.ME[1] ≈ 0.0 atol=EPS
    end

    @testset "_index_actual_by_groups uses time/step columns" begin
        # Test that :time is used when :date is not available
        ct = (
            series = ["A", "A", "A"],
            time = [3, 1, 2],  # Out of order
            value = [30.0, 10.0, 20.0]
        )

        groups = _index_actual_by_groups(ct, :value, [:series])
        key = (series = "A",)
        @test groups[key] == [10.0, 20.0, 30.0]  # Sorted by time

        # Test :step fallback
        ct_step = (
            series = ["A", "A"],
            step = [2, 1],
            value = [20.0, 10.0]
        )

        groups_step = _index_actual_by_groups(ct_step, :value, [:series])
        @test groups_step[key] == [10.0, 20.0]  # Sorted by step
    end

end
