using Test
using Durbyn.TableOps
using Statistics

# ============================================================================
# select() tests
# ============================================================================
@testset "select" begin
    tbl = (a = [1, 2, 3], b = [4, 5, 6], c = [7, 8, 9])

    @testset "basic selection" begin
        result = select(tbl, :a)
        @test result == (a = [1, 2, 3],)

        result = select(tbl, :a, :c)
        @test result == (a = [1, 2, 3], c = [7, 8, 9])

        result = select(tbl, :c, :a)
        @test result == (c = [7, 8, 9], a = [1, 2, 3])
    end

    @testset "renaming columns" begin
        result = select(tbl, :x => :a)
        @test result == (x = [1, 2, 3],)

        result = select(tbl, :x => :a, :y => :b)
        @test result == (x = [1, 2, 3], y = [4, 5, 6])

        result = select(tbl, :new_a => :a, :b, :new_c => :c)
        @test result == (new_a = [1, 2, 3], b = [4, 5, 6], new_c = [7, 8, 9])
    end

    @testset "empty specs returns all columns" begin
        result = select(tbl)
        @test result == tbl
    end

    @testset "error handling" begin
        @test_throws ArgumentError select(tbl, :nonexistent)
        @test_throws ArgumentError select(tbl, :x => :nonexistent)
    end

    @testset "duplicate output column error" begin
        @test_throws ArgumentError select(tbl, :x => :a, :x => :b)
    end

    @testset "single row table" begin
        single = (a = [1], b = [2])
        result = select(single, :b)
        @test result == (b = [2],)
    end

    @testset "empty table" begin
        empty_tbl = (a = Int[], b = Int[])
        result = select(empty_tbl, :a)
        @test result == (a = Int[],)
    end
end

# ============================================================================
# rename() tests
# ============================================================================
@testset "rename" begin
    tbl = (a = [1, 2, 3], b = [4, 5, 6], c = [7, 8, 9])

    @testset "basic rename" begin
        result = rename(tbl, :x => :a)
        @test result == (x = [1, 2, 3], b = [4, 5, 6], c = [7, 8, 9])
    end

    @testset "multiple renames" begin
        result = rename(tbl, :x => :a, :y => :b)
        @test result == (x = [1, 2, 3], y = [4, 5, 6], c = [7, 8, 9])
    end

    @testset "error on missing column" begin
        @test_throws ArgumentError rename(tbl, :x => :nonexistent)
    end

    @testset "duplicate output column in specs" begin
        @test_throws ArgumentError rename(tbl, :x => :a, :x => :b)
    end

    @testset "duplicate source column in specs" begin
        # Trying to rename same column twice
        @test_throws ArgumentError rename(tbl, :x => :a, :y => :a)
    end

    @testset "rename conflicts with existing column" begin
        # Renaming :a to :b when :b already exists and isn't renamed
        @test_throws ArgumentError rename(tbl, :b => :a)
    end
end

# ============================================================================
# query() tests
# ============================================================================
@testset "query" begin
    tbl = (id = [1, 2, 3, 4, 5], value = [10, 20, 15, 30, 25])

    @testset "basic filtering" begin
        result = query(tbl, row -> row.value > 15)
        @test result.id == [2, 4, 5]
        @test result.value == [20, 30, 25]
    end

    @testset "combined conditions" begin
        result = query(tbl, row -> row.id > 1 && row.value < 25)
        @test result.id == [2, 3]
        @test result.value == [20, 15]
    end

    @testset "filter all rows" begin
        result = query(tbl, row -> row.value > 100)
        @test isempty(result.id)
        @test isempty(result.value)
    end

    @testset "filter no rows (keep all)" begin
        result = query(tbl, row -> true)
        @test result == tbl
    end

    @testset "single matching row" begin
        result = query(tbl, row -> row.id == 3)
        @test result.id == [3]
        @test result.value == [15]
    end

    @testset "string column filtering" begin
        str_tbl = (name = ["Alice", "Bob", "Charlie"], age = [25, 30, 20])
        result = query(str_tbl, row -> startswith(row.name, "A") || startswith(row.name, "C"))
        @test result.name == ["Alice", "Charlie"]
        @test result.age == [25, 20]
    end

    @testset "empty table" begin
        empty_tbl = (id = Int[], value = Int[])
        result = query(empty_tbl, row -> row.value > 0)
        @test isempty(result.id)
    end

    @testset "error on non-table input" begin
        @test_throws ArgumentError query([1, 2, 3], x -> x > 1)
    end
end

# ============================================================================
# arrange() tests
# ============================================================================
@testset "arrange" begin
    tbl = (name = ["Charlie", "Alice", "Bob"], age = [20, 25, 30], score = [85, 90, 85])

    @testset "single column ascending" begin
        result = arrange(tbl, :name)
        @test result.name == ["Alice", "Bob", "Charlie"]
        @test result.age == [25, 30, 20]
    end

    @testset "single column descending" begin
        result = arrange(tbl, :age => :desc)
        @test result.name == ["Bob", "Alice", "Charlie"]
        @test result.age == [30, 25, 20]
    end

    @testset "multiple columns" begin
        tbl2 = (group = ["A", "B", "A", "B"], value = [3, 1, 2, 4])
        result = arrange(tbl2, :group, :value)
        @test result.group == ["A", "A", "B", "B"]
        @test result.value == [2, 3, 1, 4]
    end

    @testset "multiple columns with mixed order" begin
        tbl2 = (group = ["A", "B", "A", "B"], value = [3, 1, 2, 4])
        result = arrange(tbl2, :group, :value => :desc)
        @test result.group == ["A", "A", "B", "B"]
        @test result.value == [3, 2, 4, 1]
    end

    @testset "reverse option" begin
        result = arrange(tbl, :age; rev=true)
        @test result.age == [30, 25, 20]
    end

    @testset "descending with different keywords" begin
        result1 = arrange(tbl, :age => :desc)
        result2 = arrange(tbl, :age => :descending)
        result3 = arrange(tbl, :age => :reverse)
        @test result1.age == result2.age == result3.age == [30, 25, 20]
    end

    @testset "empty specs returns original order" begin
        result = arrange(tbl)
        @test result == tbl
    end

    @testset "single row table" begin
        single = (a = [1], b = [2])
        result = arrange(single, :a)
        @test result == single
    end

    @testset "empty table" begin
        empty_tbl = (a = Int[], b = Int[])
        result = arrange(empty_tbl, :a)
        @test result == empty_tbl
    end

    @testset "error on nonexistent column" begin
        @test_throws ArgumentError arrange(tbl, :nonexistent)
    end

    @testset "stable sort with equal values" begin
        tbl3 = (id = [1, 2, 3], value = [10, 10, 10])
        result = arrange(tbl3, :value)
        @test result.id == [1, 2, 3]
    end
end

# ============================================================================
# groupby() tests
# ============================================================================
@testset "groupby" begin
    tbl = (category = ["A", "B", "A", "B", "A"], value = [1, 2, 3, 4, 5])

    @testset "basic grouping" begin
        gt = groupby(tbl, :category)
        @test gt isa TableOps.GroupedTable
        @test length(gt) == 2
    end

    @testset "group key columns" begin
        gt = groupby(tbl, :category)
        @test gt.keycols == [:category]
    end

    @testset "multiple grouping columns" begin
        tbl2 = (a = ["X", "X", "Y", "Y"], b = [1, 2, 1, 2], val = [10, 20, 30, 40])
        gt = groupby(tbl2, :a, :b)
        @test length(gt) == 4
        @test gt.keycols == [:a, :b]
    end

    @testset "groupby with vector of symbols" begin
        gt = groupby(tbl, [:category])
        @test length(gt) == 2
    end

    @testset "error on empty grouping columns" begin
        @test_throws ArgumentError groupby(tbl, Symbol[])
    end

    @testset "error on nonexistent column" begin
        @test_throws ArgumentError groupby(tbl, :nonexistent)
    end

    @testset "show methods" begin
        gt = groupby(tbl, :category)
        io = IOBuffer()
        show(io, gt)
        str = String(take!(io))
        @test contains(str, "GroupedTable")
        @test contains(str, "2 groups")

        io = IOBuffer()
        show(io, MIME"text/plain"(), gt)
        str = String(take!(io))
        @test contains(str, "GroupedTable")
        @test contains(str, "Groups: 2")
    end

    @testset "single group" begin
        single_group = (cat = ["A", "A", "A"], val = [1, 2, 3])
        gt = groupby(single_group, :cat)
        @test length(gt) == 1
    end

    @testset "all unique groups" begin
        unique_groups = (id = [1, 2, 3], val = [10, 20, 30])
        gt = groupby(unique_groups, :id)
        @test length(gt) == 3
    end
end

# ============================================================================
# mutate() tests
# ============================================================================
@testset "mutate" begin
    tbl = (a = [1, 2, 3], b = [4, 5, 6])

    @testset "add new column with function" begin
        result = mutate(tbl, c = data -> data.a .+ data.b)
        @test result.a == [1, 2, 3]
        @test result.b == [4, 5, 6]
        @test result.c == [5, 7, 9]
    end

    @testset "add multiple columns" begin
        result = mutate(tbl,
            sum_col = data -> data.a .+ data.b,
            prod_col = data -> data.a .* data.b)
        @test result.sum_col == [5, 7, 9]
        @test result.prod_col == [4, 10, 18]
    end

    @testset "modify existing column" begin
        result = mutate(tbl, a = data -> data.a .* 2)
        @test result.a == [2, 4, 6]
        @test result.b == [4, 5, 6]
    end

    @testset "add constant column" begin
        result = mutate(tbl, constant = data -> fill(100, length(data.a)))
        @test result.constant == [100, 100, 100]
    end

    @testset "column based on other new column" begin
        result = mutate(tbl, c = data -> data.a .* 2)
        result2 = mutate(result, d = data -> data.c .+ 1)
        @test result2.d == [3, 5, 7]
    end

    @testset "error on wrong length" begin
        @test_throws ArgumentError mutate(tbl, bad = data -> [1, 2])
    end

    @testset "error on non-vector result" begin
        @test_throws ArgumentError mutate(tbl, bad = data -> 42)
    end

    @testset "empty table" begin
        empty_tbl = (a = Int[], b = Int[])
        result = mutate(empty_tbl, c = data -> data.a .+ data.b)
        @test isempty(result.c)
    end

    @testset "direct vector assignment" begin
        result = mutate(tbl, c = [10, 20, 30])
        @test result.c == [10, 20, 30]
    end
end

# ============================================================================
# summarise() / summarize() tests
# ============================================================================
@testset "summarise" begin
    tbl = (category = ["A", "B", "A", "B", "A"], value = [10, 20, 30, 40, 50])
    gt = groupby(tbl, :category)

    @testset "basic mean" begin
        result = summarise(gt, mean_val = :value => mean)
        @test result.category == ["A", "B"]
        @test result.mean_val ≈ [30.0, 30.0]
    end

    @testset "multiple summaries" begin
        result = summarise(gt,
            mean_val = :value => mean,
            sum_val = :value => sum,
            count = data -> length(data.value))
        @test result.category == ["A", "B"]
        @test result.mean_val ≈ [30.0, 30.0]
        @test result.sum_val == [90, 60]
        @test result.count == [3, 2]
    end

    @testset "min/max" begin
        result = summarise(gt,
            min_val = :value => minimum,
            max_val = :value => maximum)
        @test result.min_val == [10, 20]
        @test result.max_val == [50, 40]
    end

    @testset "custom function on entire group" begin
        result = summarise(gt, range = data -> maximum(data.value) - minimum(data.value))
        @test result.range == [40, 20]
    end

    @testset "summarize alias" begin
        result1 = summarise(gt, m = :value => mean)
        result2 = summarize(gt, m = :value => mean)
        @test result1 == result2
    end

    @testset "multiple grouping columns" begin
        tbl2 = (a = ["X", "X", "Y", "Y"], b = [1, 1, 2, 2], val = [10, 20, 30, 40])
        gt2 = groupby(tbl2, :a, :b)
        result = summarise(gt2, total = :val => sum)
        @test length(result.a) == 2
        @test result.total == [30, 70]
    end

    @testset "std and var" begin
        result = summarise(gt, std_val = :value => std)
        @test length(result.std_val) == 2
        @test all(x -> x >= 0, result.std_val)
    end

    @testset "single row groups" begin
        tbl3 = (id = [1, 2, 3], val = [10, 20, 30])
        gt3 = groupby(tbl3, :id)
        result = summarise(gt3, total = :val => sum)
        @test result.total == [10, 20, 30]
    end
end

# ============================================================================
# pivot_longer() tests
# ============================================================================
@testset "pivot_longer" begin
    wide = (date = ["2024-01", "2024-02"], A = [100, 110], B = [200, 220], C = [300, 330])

    @testset "basic pivot" begin
        long = pivot_longer(wide, id_cols=:date, names_to=:series, values_to=:value)
        @test length(long.date) == 6
        @test Set(long.series) == Set(["A", "B", "C"])
        @test sum(long.value) == 100 + 110 + 200 + 220 + 300 + 330
    end

    @testset "default names_to and values_to" begin
        long = pivot_longer(wide, id_cols=:date)
        @test haskey(long, :variable)
        @test haskey(long, :value)
    end

    @testset "specific value columns" begin
        long = pivot_longer(wide, id_cols=:date, value_cols=[:A, :B])
        @test Set(long.variable) == Set(["A", "B"])
        @test !("C" in long.variable)
    end

    @testset "multiple id columns" begin
        wide2 = (year = [2024, 2024], month = [1, 2], X = [10, 20], Y = [30, 40])
        long = pivot_longer(wide2, id_cols=[:year, :month])
        @test length(long.year) == 4
        @test length(long.month) == 4
    end

    @testset "infer id_cols from value_cols" begin
        long = pivot_longer(wide, value_cols=[:A, :B, :C])
        @test haskey(long, :date)
    end

    @testset "single value column" begin
        simple = (id = [1, 2], val = [10, 20])
        long = pivot_longer(simple, id_cols=:id, value_cols=:val)
        @test length(long.id) == 2
    end

    @testset "error on missing id column" begin
        @test_throws ArgumentError pivot_longer(wide, id_cols=:nonexistent)
    end

    @testset "error on missing value column" begin
        @test_throws ArgumentError pivot_longer(wide, id_cols=:date, value_cols=[:nonexistent])
    end

    @testset "empty table" begin
        empty_wide = (id = String[], A = Int[], B = Int[])
        long = pivot_longer(empty_wide, id_cols=:id)
        @test isempty(long.id)
    end

    @testset "zero-column table" begin
        empty_tbl = NamedTuple()
        result = pivot_longer(empty_tbl)
        @test haskey(result, :variable)
        @test haskey(result, :value)
        @test length(result.variable) == 0
    end

    @testset "zero-column table with invalid id_cols" begin
        empty_tbl = NamedTuple()
        @test_throws ArgumentError pivot_longer(empty_tbl, id_cols=:nonexistent)
    end

    @testset "zero-column table with invalid value_cols" begin
        empty_tbl = NamedTuple()
        @test_throws ArgumentError pivot_longer(empty_tbl, value_cols=[:nonexistent])
    end

    @testset "names_to == values_to collision" begin
        tbl = (id = [1, 2], A = [10, 20], B = [30, 40])
        @test_throws ArgumentError pivot_longer(tbl, id_cols=:id, names_to=:x, values_to=:x)
    end

    @testset "names_to == values_to collision on empty table" begin
        empty_tbl = NamedTuple()
        @test_throws ArgumentError pivot_longer(empty_tbl, names_to=:x, values_to=:x)
    end

    @testset "names_to conflicts with id_cols" begin
        tbl = (id = [1, 2], A = [10, 20], B = [30, 40])
        @test_throws ArgumentError pivot_longer(tbl, id_cols=:id, names_to=:id, values_to=:value)
    end

    @testset "values_to conflicts with id_cols" begin
        tbl = (id = [1, 2], A = [10, 20], B = [30, 40])
        @test_throws ArgumentError pivot_longer(tbl, id_cols=:id, names_to=:series, values_to=:id)
    end

    @testset "duplicate id_cols error" begin
        tbl = (id = [1, 2], A = [10, 20], B = [30, 40])
        @test_throws ArgumentError pivot_longer(tbl, id_cols=[:id, :id])
    end

    @testset "duplicate value_cols error" begin
        tbl = (id = [1, 2], A = [10, 20], B = [30, 40])
        @test_throws ArgumentError pivot_longer(tbl, id_cols=:id, value_cols=[:A, :A])
    end
end

# ============================================================================
# pivot_wider() tests
# ============================================================================
@testset "pivot_wider" begin
    long = (date = ["2024-01", "2024-01", "2024-02", "2024-02"],
            series = ["A", "B", "A", "B"],
            value = [100, 200, 110, 220])

    @testset "basic pivot" begin
        wide = pivot_wider(long, names_from=:series, values_from=:value, id_cols=:date)
        @test haskey(wide, :A)
        @test haskey(wide, :B)
        @test wide.date == ["2024-01", "2024-02"]
        @test wide.A == [100, 110]
        @test wide.B == [200, 220]
    end

    @testset "sort names" begin
        long2 = (id = [1, 1], cat = ["Z", "A"], val = [10, 20])
        wide = pivot_wider(long2, names_from=:cat, values_from=:val, sort_names=true)
        @test collect(keys(wide))[2:end] == [:A, :Z]
    end

    @testset "fill missing values" begin
        incomplete = (id = [1, 1, 2], category = ["A", "B", "A"], val = [10, 20, 30])
        wide = pivot_wider(incomplete, names_from=:category, values_from=:val, fill=0)
        @test wide.A == [10, 30]
        @test wide.B == [20, 0]
    end

    @testset "fill with missing (default)" begin
        incomplete = (id = [1, 1, 2], category = ["A", "B", "A"], val = [10, 20, 30])
        wide = pivot_wider(incomplete, names_from=:category, values_from=:val)
        @test wide.A == [10, 30]
        @test ismissing(wide.B[2])
    end

    @testset "infer id_cols" begin
        wide = pivot_wider(long, names_from=:series, values_from=:value)
        @test haskey(wide, :date)
    end

    @testset "multiple id columns" begin
        long2 = (year = [2024, 2024, 2024, 2024],
                 month = [1, 1, 2, 2],
                 cat = ["X", "Y", "X", "Y"],
                 val = [1, 2, 3, 4])
        wide = pivot_wider(long2, names_from=:cat, values_from=:val, id_cols=[:year, :month])
        @test wide.X == [1, 3]
        @test wide.Y == [2, 4]
    end

    @testset "error on same names_from and values_from" begin
        @test_throws ArgumentError pivot_wider(long, names_from=:series, values_from=:series)
    end

    @testset "error on missing column" begin
        @test_throws ArgumentError pivot_wider(long, names_from=:nonexistent, values_from=:value)
        @test_throws ArgumentError pivot_wider(long, names_from=:series, values_from=:nonexistent)
    end

    @testset "error on names_from in id_cols" begin
        @test_throws ArgumentError pivot_wider(long, names_from=:series, values_from=:value, id_cols=[:date, :series])
    end

    @testset "error on duplicate entries" begin
        dup = (id = [1, 1], cat = ["A", "A"], val = [10, 20])
        @test_throws ArgumentError pivot_wider(dup, names_from=:cat, values_from=:val)
    end

    @testset "error on missing in names_from" begin
        with_missing = (id = [1, 2], cat = ["A", missing], val = [10, 20])
        @test_throws ArgumentError pivot_wider(with_missing, names_from=:cat, values_from=:val)
    end

    @testset "error on names_from value conflicts with id_cols" begin
        # names_from contains value "id" which would conflict with id column
        conflict_data = (id = [1, 1], cat = ["id", "x"], val = [10, 20])
        @test_throws ArgumentError pivot_wider(conflict_data, names_from=:cat, values_from=:val, id_cols=:id)
    end

    @testset "empty table" begin
        empty_long = (id = Int[], series = String[], value = Int[])
        wide = pivot_wider(empty_long, names_from=:series, values_from=:value, id_cols=:id)
        @test isempty(wide.id)
    end
end

# ============================================================================
# pivot_longer / pivot_wider round-trip tests
# ============================================================================
@testset "pivot round-trip" begin
    original = (id = [1, 2, 3], A = [10, 20, 30], B = [40, 50, 60])

    long = pivot_longer(original, id_cols=:id, names_to=:var, values_to=:val)
    wide = pivot_wider(long, names_from=:var, values_from=:val, id_cols=:id)

    @test Set(keys(wide)) == Set([:id, :A, :B])
    @test sort(wide.id) == [1, 2, 3]
end

# ============================================================================
# glimpse() tests
# ============================================================================
@testset "glimpse" begin
    tbl = (name = ["Alice", "Bob", "Charlie", "Diana", "Eve"],
           age = [25, 30, 35, 40, 45],
           score = [85.5, 90.0, 78.5, 92.0, 88.0])

    @testset "basic glimpse" begin
        io = IOBuffer()
        glimpse(tbl; io=io)
        output = String(take!(io))
        @test contains(output, "Table glimpse")
        @test contains(output, "Rows: 5")
        @test contains(output, "Columns: 3")
        @test contains(output, "name")
        @test contains(output, "age")
        @test contains(output, "score")
    end

    @testset "maxrows parameter" begin
        io = IOBuffer()
        glimpse(tbl; maxrows=2, io=io)
        output = String(take!(io))
        @test contains(output, "…")
    end

    @testset "empty table" begin
        empty_tbl = (a = Int[], b = String[])
        io = IOBuffer()
        glimpse(empty_tbl; io=io)
        output = String(take!(io))
        @test contains(output, "Rows: 0")
        @test contains(output, "[]")
    end

    @testset "grouped table glimpse" begin
        gt = groupby(tbl, :age)
        io = IOBuffer()
        glimpse(gt; io=io)
        output = String(take!(io))
        @test contains(output, "GroupedTable glimpse")
        @test contains(output, "Groups:")
    end

    @testset "grouped table with maxgroups" begin
        big_tbl = (cat = repeat(["A", "B", "C", "D", "E"], 2), val = 1:10)
        gt = groupby(big_tbl, :cat)
        io = IOBuffer()
        glimpse(gt; maxgroups=2, io=io)
        output = String(take!(io))
        @test contains(output, "more groups")
    end
end

# ============================================================================
# GroupedTable tests
# ============================================================================
@testset "GroupedTable" begin
    tbl = (cat = ["A", "B", "A", "B", "C"], val = [1, 2, 3, 4, 5])
    gt = groupby(tbl, :cat)

    @testset "length" begin
        @test length(gt) == 3
    end

    @testset "show compact" begin
        io = IOBuffer()
        show(io, gt)
        str = String(take!(io))
        @test contains(str, "3 groups")
        @test contains(str, "cat")
    end

    @testset "show detailed" begin
        io = IOBuffer()
        show(io, MIME"text/plain"(), gt)
        str = String(take!(io))
        @test contains(str, "GroupedTable")
        @test contains(str, "Groups: 3")
        @test contains(str, "Key columns: cat")
        @test contains(str, "Rows:")
        @test contains(str, "Sample groups")
    end

    @testset "empty grouped table display" begin
        empty_tbl = (cat = String[], val = Int[])
        @test_throws ArgumentError groupby(empty_tbl, Symbol[])
    end
end

# ============================================================================
# Edge cases and integration tests
# ============================================================================
@testset "edge cases" begin
    @testset "special characters in column names" begin
        tbl = (; Symbol("col-1") => [1, 2], Symbol("col_2") => [3, 4])
        result = select(tbl, Symbol("col-1"))
        @test result[Symbol("col-1")] == [1, 2]
    end

    @testset "numeric column values becoming names" begin
        long = (id = [1, 1], year = [2020, 2021], val = [100, 200])
        wide = pivot_wider(long, names_from=:year, values_from=:val)
        @test haskey(wide, Symbol("2020"))
        @test haskey(wide, Symbol("2021"))
    end

    @testset "chain operations" begin
        tbl = (category = ["A", "B", "A", "B", "A"],
               value = [10, 20, 30, 40, 50],
               extra = [1, 2, 3, 4, 5])

        result = tbl |>
            x -> select(x, :category, :value) |>
            x -> query(x, row -> row.value > 15) |>
            x -> arrange(x, :value => :desc)

        # After filtering value > 15: [20, 30, 40, 50] with categories [B, A, B, A]
        # After sorting descending: [50, 40, 30, 20] with categories [A, B, A, B]
        @test result.category == ["A", "B", "A", "B"]
        @test result.value == [50, 40, 30, 20]
    end

    @testset "large table performance" begin
        n = 10000
        large_tbl = (id = 1:n, value = rand(n), category = rand(["A", "B", "C"], n))

        # These should complete without error
        result = query(large_tbl, row -> row.value > 0.5)
        @test length(result.id) > 0

        sorted = arrange(large_tbl, :value)
        @test issorted(sorted.value)

        gt = groupby(large_tbl, :category)
        @test length(gt) == 3

        summary = summarise(gt, mean_val = :value => mean)
        @test length(summary.category) == 3
    end

    @testset "missing values in data" begin
        tbl_with_missing = (id = [1, 2, 3], value = [10, missing, 30])

        # query should handle missing
        result = query(tbl_with_missing, row -> !ismissing(row.value) && row.value > 15)
        @test result.id == [3]

        # mutate with missing
        result = mutate(tbl_with_missing, doubled = data -> coalesce.(data.value .* 2, 0))
        @test result.doubled == [20, 0, 60]
    end

    @testset "different numeric types" begin
        tbl = (a = Int32[1, 2, 3], b = Float32[1.0, 2.0, 3.0], c = [1, 2, 3])
        result = mutate(tbl, sum = data -> data.a .+ data.b .+ data.c)
        @test result.sum == [3.0, 6.0, 9.0]
    end

    @testset "boolean columns" begin
        tbl = (id = [1, 2, 3], flag = [true, false, true])
        result = query(tbl, row -> row.flag)
        @test result.id == [1, 3]
    end
end

# ============================================================================
# complete() edge case tests
# ============================================================================
@testset "complete edge cases" begin
    @testset "single column" begin
        tbl = (x = [1, 2, 1], y = [10, 20, 30])
        result = complete(tbl, :x; fill_value=0)
        # All unique x values already present, so no rows added
        @test length(result.x) == 3
    end

    @testset "single column with missing combo" begin
        # More meaningful single-column test would need explicit unique values
        # complete works on existing unique combinations
        tbl = (cat = ["A", "B", "A"], val = [1, 2, 3])
        result = complete(tbl, :cat)
        @test length(result.cat) == 3  # A, B already present
    end

    @testset "multi-column with missing combo" begin
        tbl = (a = [1, 1, 2], b = ["x", "y", "x"], val = [10, 20, 30])
        result = complete(tbl, :a, :b; fill_value=0)
        @test length(result.a) == 4  # Adds (2, "y") combo
        # Find the added row
        added_idx = findfirst(i -> result.a[i] == 2 && result.b[i] == "y", 1:length(result.a))
        @test added_idx !== nothing
        @test result.val[added_idx] == 0
    end
end

# ============================================================================
# arrange() edge case tests
# ============================================================================
@testset "arrange edge cases" begin
    @testset "missing values sort to end" begin
        tbl = (x = [3, 1, missing, 2, missing], y = ["c", "a", "e", "b", "d"])
        result = arrange(tbl, :x)
        # Non-missing values should be sorted first
        @test result.x[1] == 1
        @test result.x[2] == 2
        @test result.x[3] == 3
        # Missing values at end
        @test ismissing(result.x[4])
        @test ismissing(result.x[5])
    end

    @testset "missing values with descending" begin
        tbl = (x = [3, 1, missing, 2], y = ["c", "a", "e", "b"])
        result = arrange(tbl, :x => :desc)
        # Descending: 3, 2, 1, then missing
        @test result.x[1] == 3
        @test result.x[2] == 2
        @test result.x[3] == 1
        @test ismissing(result.x[4])
    end

    @testset "stable sort preserves order for equal keys" begin
        tbl = (key = [1, 1, 1], order = [1, 2, 3])
        result = arrange(tbl, :key)
        # Equal keys should maintain original order (stable sort)
        @test result.order == [1, 2, 3]
    end
end

# ============================================================================
# Internal helper function tests
# ============================================================================
@testset "internal helpers" begin
    @testset "_to_columns" begin
        nt = (a = [1, 2], b = [3, 4])
        @test TableOps._to_columns(nt) == nt

        @test_throws ArgumentError TableOps._to_columns([1, 2, 3])
    end

    @testset "_check_lengths" begin
        valid = (a = [1, 2, 3], b = [4, 5, 6])
        @test TableOps._check_lengths(valid) == 3

        invalid = (a = [1, 2, 3], b = [4, 5])
        @test_throws ArgumentError TableOps._check_lengths(invalid)
    end

    @testset "_nrows" begin
        tbl = (a = [1, 2, 3], b = [4, 5, 6])
        @test TableOps._nrows(tbl) == 3

        empty_tbl = NamedTuple()
        @test TableOps._nrows(empty_tbl) == 0
    end

    @testset "_subset_mask" begin
        tbl = (a = [1, 2, 3], b = [4, 5, 6])
        mask = [true, false, true]
        result = TableOps._subset_mask(tbl, mask)
        @test result.a == [1, 3]
        @test result.b == [4, 6]
    end

    @testset "_subset_indices" begin
        tbl = (a = [1, 2, 3], b = [4, 5, 6])
        idxs = [3, 1]
        result = TableOps._subset_indices(tbl, idxs)
        @test result.a == [3, 1]
        @test result.b == [6, 4]
    end
end

# ============================================================================
# PanelData grouped order tests
# ============================================================================
@testset "PanelData grouped order" begin
    using Durbyn.ModelSpecs

    @testset "PanelData sorts by groups and date at construction" begin
        # Input data in arbitrary order
        data = (series = ["B", "A", "B", "A", "B", "A"],
                date = [1, 1, 2, 2, 3, 3],
                value = [100, 10, 200, 20, 300, 30])
        panel = PanelData(data; groupby=:series, date=:date)

        # Data should be sorted by series (alphabetically), then by date
        @test panel.data.series == ["A", "A", "A", "B", "B", "B"]
        @test panel.data.date == [1, 2, 3, 1, 2, 3]
        @test panel.data.value == [10, 20, 30, 100, 200, 300]
    end

    @testset "mutate preserves grouped order" begin
        data = (series = ["B", "A", "B", "A", "B", "A"],
                date = [1, 1, 2, 2, 3, 3],
                value = [100, 10, 200, 20, 300, 30])
        panel = PanelData(data; groupby=:series, date=:date)

        result = mutate(panel, doubled = d -> d.value .* 2)
        # Data remains in grouped order (A's then B's, sorted by date)
        @test result.data.series == ["A", "A", "A", "B", "B", "B"]
        @test result.data.value == [10, 20, 30, 100, 200, 300]
        @test result.data.doubled == [20, 40, 60, 200, 400, 600]
    end

    @testset "query preserves grouped order" begin
        data = (series = ["B", "A", "B", "A", "B", "A"],
                date = [1, 1, 2, 2, 3, 3],
                value = [100, 10, 200, 20, 300, 30])
        panel = PanelData(data; groupby=:series, date=:date)

        result = query(panel, r -> r.value > 50)
        # Filtered data maintains grouped order
        @test result.data.series == ["B", "B", "B"]
        @test result.data.value == [100, 200, 300]
    end

    @testset "arrange reorders within groups" begin
        data = (series = ["B", "A", "B", "A"],
                date = [2, 1, 1, 2],
                value = [200, 10, 100, 20])
        panel = PanelData(data; groupby=:series, date=:date)

        # Arrange by value (descending within each group)
        result = arrange(panel, :value; rev=true)
        # Within each group, values should be descending
        a_mask = result.data.series .== "A"
        b_mask = result.data.series .== "B"
        @test issorted(result.data.value[a_mask], rev=true)
        @test issorted(result.data.value[b_mask], rev=true)
    end
end

# ============================================================================
# Join duplicate name validation tests
# ============================================================================
@testset "join duplicate name validation" begin
    @testset "inner_join detects suffix collision" begin
        # Left has x_y and x, right has x
        # Conflict on x: left x -> x_x, right x -> x_y, but x_y already exists in left!
        left = (id = [1, 2], x_y = [10, 20], x = [1, 2])
        right = (id = [1, 2], x = [100, 200])
        @test_throws ArgumentError inner_join(left, right, by=:id)
    end

    @testset "left_join detects suffix collision" begin
        left = (id = [1, 2], x_y = [10, 20], x = [1, 2])
        right = (id = [1, 2], x = [100, 200])
        @test_throws ArgumentError left_join(left, right, by=:id)
    end

    @testset "right_join detects suffix collision" begin
        # Right has x_x and x, left has x
        # Conflict on x: left x -> x_x, but x_x already exists in right!
        left = (id = [1, 2], x = [10, 20])
        right = (id = [1, 2], x_x = [100, 200], x = [1, 2])
        @test_throws ArgumentError right_join(left, right, by=:id)
    end

    @testset "full_join detects suffix collision" begin
        left = (id = [1, 2], x_y = [10, 20], x = [1, 2])
        right = (id = [1, 2], x = [100, 200])
        @test_throws ArgumentError full_join(left, right, by=:id)
    end

    @testset "joins work with custom suffixes avoiding collision" begin
        left = (id = [1, 2], x_y = [10, 20], x = [1, 2])
        right = (id = [1, 2], x = [100, 200])
        # Using different suffixes that don't collide
        result = inner_join(left, right, by=:id, suffix=("_left", "_right"))
        @test haskey(result, :x_y)
        @test haskey(result, :x_left)
        @test haskey(result, :x_right)
    end
end
