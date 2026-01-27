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

    @testset "numeric group ordering" begin
        # Test that numeric groups are sorted numerically, not lexicographically
        # Lexicographic: 1, 10, 2, 3 vs Numeric: 1, 2, 3, 10
        tbl = (id = [10, 1, 3, 2], val = [100, 10, 30, 20])
        gt = groupby(tbl, :id)
        @test length(gt) == 4
        # Keys should be in numeric order: 1, 2, 3, 10
        key_ids = [k.id for k in gt.keys]
        @test key_ids == [1, 2, 3, 10]
    end

    @testset "mixed group ordering with missing" begin
        # Missing values should sort last
        tbl = (id = [3, missing, 1, 2], val = [30, 0, 10, 20])
        gt = groupby(tbl, :id)
        @test length(gt) == 4
        key_ids = [k.id for k in gt.keys]
        @test key_ids[1:3] == [1, 2, 3]
        @test ismissing(key_ids[4])
    end

    @testset "mixed types throw error" begin
        # Mixed types (Int and String) that can't be compared with isless
        # should throw an error - no silent string fallback
        tbl = (id = Any[1, "2", 3], val = [10, 20, 30])
        @test_throws ArgumentError groupby(tbl, :id)
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

    @testset "missing values with rev=true stay last" begin
        tbl = (x = [1, missing, 2], y = ["a", "b", "c"])
        result = arrange(tbl, :x; rev=true)
        @test result.x[1] == 2
        @test result.x[2] == 1
        @test ismissing(result.x[3])
    end

    @testset "stable sort preserves order for equal keys" begin
        tbl = (key = [1, 1, 1], order = [1, 2, 3])
        result = arrange(tbl, :key)
        # Equal keys should maintain original order (stable sort)
        @test result.order == [1, 2, 3]
    end

    @testset "mixed types throw error" begin
        # Column with mixed Int and String types - should error, not silently fallback
        tbl = (x = Any[1, "b", 2, "a"], y = [1, 2, 3, 4])
        @test_throws ArgumentError arrange(tbl, :x)
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

# ============================================================================
# fill_missing tests
# ============================================================================
@testset "fill_missing" begin
    @testset "basic fill down" begin
        tbl = (a = [1, missing, missing, 4], b = [missing, 2, missing, missing])
        result = fill_missing(tbl, :a, direction=:down)
        @test isequal(result.a, [1, 1, 1, 4])
        @test isequal(result.b, [missing, 2, missing, missing])  # unchanged
    end

    @testset "basic fill up" begin
        tbl = (a = [missing, missing, 3, 4],)
        result = fill_missing(tbl, :a, direction=:up)
        @test isequal(result.a, [3, 3, 3, 4])
    end

    @testset "fill with nothing values - sentinel bug fix" begin
        # nothing is a valid value distinct from missing
        tbl = (a = Union{Int, Nothing, Missing}[nothing, missing, 3, missing],)
        result = fill_missing(tbl, :a, direction=:down)
        # nothing should propagate like any other value
        @test result.a[1] === nothing
        @test result.a[2] === nothing  # filled from nothing above
        @test result.a[3] == 3
        @test result.a[4] == 3  # filled from 3 above
    end

    @testset "fill up with nothing values" begin
        tbl = (a = Union{Int, Nothing, Missing}[missing, 2, missing, nothing],)
        result = fill_missing(tbl, :a, direction=:up)
        @test result.a[1] == 2  # filled from 2 below
        @test result.a[2] == 2
        @test result.a[3] === nothing  # filled from nothing below
        @test result.a[4] === nothing
    end
end

# ============================================================================
# bind_rows column length validation tests
# ============================================================================
@testset "bind_rows validation" begin
    @testset "valid tables bind correctly" begin
        tbl1 = (a = [1, 2], b = [3, 4])
        tbl2 = (a = [5, 6], b = [7, 8])
        result = bind_rows(tbl1, tbl2)
        @test result.a == [1, 2, 5, 6]
        @test result.b == [3, 4, 7, 8]
    end

    @testset "mismatched column lengths in input table" begin
        bad_tbl = (a = [1, 2, 3], b = [4, 5])  # mismatched lengths
        good_tbl = (a = [6, 7], b = [8, 9])
        @test_throws ArgumentError bind_rows(bad_tbl, good_tbl)
        @test_throws ArgumentError bind_rows(good_tbl, bad_tbl)
    end
end

# ============================================================================
# join column length validation tests
# ============================================================================
@testset "join column length validation" begin
    @testset "inner_join validates column lengths" begin
        bad_left = (id = [1, 2], x = [10, 20, 30])  # mismatched
        good_right = (id = [1, 2], y = [100, 200])
        @test_throws ArgumentError inner_join(bad_left, good_right, by=:id)

        good_left = (id = [1, 2], x = [10, 20])
        bad_right = (id = [1, 2, 3], y = [100, 200])  # mismatched
        @test_throws ArgumentError inner_join(good_left, bad_right, by=:id)
    end

    @testset "left_join validates column lengths" begin
        bad_tbl = (id = [1, 2], x = [10, 20, 30])
        good_tbl = (id = [1, 2], y = [100, 200])
        @test_throws ArgumentError left_join(bad_tbl, good_tbl, by=:id)
    end

    @testset "right_join validates column lengths" begin
        bad_tbl = (id = [1, 2], x = [10, 20, 30])
        good_tbl = (id = [1, 2], y = [100, 200])
        @test_throws ArgumentError right_join(good_tbl, bad_tbl, by=:id)
    end

    @testset "full_join validates column lengths" begin
        bad_tbl = (id = [1, 2], x = [10, 20, 30])
        good_tbl = (id = [1, 2], y = [100, 200])
        @test_throws ArgumentError full_join(bad_tbl, good_tbl, by=:id)
    end
end

# ============================================================================
# separate name collision validation tests
# ============================================================================
@testset "separate name validation" begin
    @testset "basic separate works" begin
        tbl = (id = [1, 2], full_name = ["John_Doe", "Jane_Smith"])
        result = separate(tbl, :full_name; into=[:first, :last], sep="_")
        @test result.first == ["John", "Jane"]
        @test result.last == ["Doe", "Smith"]
    end

    @testset "duplicate names in into" begin
        tbl = (id = [1, 2], name = ["a_b", "c_d"])
        @test_throws ArgumentError separate(tbl, :name; into=[:x, :x], sep="_")
    end

    @testset "into name conflicts with existing column" begin
        tbl = (id = [1, 2], name = ["a_b", "c_d"])
        # :id already exists
        @test_throws ArgumentError separate(tbl, :name; into=[:id, :last], sep="_")
    end

    @testset "into can use separated column name with remove=true" begin
        # When remove=true, the original column is removed, so its name can be reused
        tbl = (id = [1, 2], name = ["a_b", "c_d"])
        result = separate(tbl, :name; into=[:name, :suffix], sep="_", remove=true)
        @test haskey(result, :name)
        @test result.name == ["a", "c"]
    end
end

# ============================================================================
# unite name collision validation tests
# ============================================================================
@testset "unite name validation" begin
    @testset "basic unite works" begin
        tbl = (id = [1, 2], year = [2020, 2021], month = [1, 6])
        result = unite(tbl, :date, :year, :month; sep="-")
        @test result.date == ["2020-1", "2021-6"]
        @test !haskey(result, :year)  # removed by default
        @test !haskey(result, :month)
    end

    @testset "new_col conflicts with existing column" begin
        tbl = (id = [1, 2], year = [2020, 2021], month = [1, 6])
        # :id already exists and won't be removed
        @test_throws ArgumentError unite(tbl, :id, :year, :month)
    end

    @testset "new_col can match source column with remove=true" begin
        tbl = (id = [1, 2], year = [2020, 2021], month = [1, 6])
        # :year will be removed, so it can be the output name
        result = unite(tbl, :year, :year, :month; sep="-", remove=true)
        @test result.year == ["2020-1", "2021-6"]
    end

    @testset "new_col conflicts with source column when remove=false" begin
        tbl = (id = [1, 2], year = [2020, 2021], month = [1, 6])
        # With remove=false, :year stays, so can't use it as output name
        @test_throws ArgumentError unite(tbl, :year, :year, :month; remove=false)
    end
end

# ============================================================================
# Across output name collision tests
# ============================================================================
@testset "Across output name collision" begin
    @testset "summarise Across collision with group key" begin
        tbl = (a_mean = ["x", "x", "y"], a = [1, 2, 3], b = [4, 5, 6])
        gt = groupby(tbl, :a_mean)
        ac = Across([:a], [:mean => mean])
        # Output would be :a_mean which collides with group key
        @test_throws ArgumentError summarise(gt, ac)
    end

    @testset "summarise Across collision between generated outputs" begin
        # When multiple columns would produce the same output name
        tbl = (group = ["x", "x", "y"], a = [1, 2, 3], a_sum = [10, 20, 30])
        gt = groupby(tbl, :group)
        # Both :a and :a_sum columns with :sum function would produce :a_sum twice
        # But :a_sum_sum and :a_sum are different names, so we need a different test
        # Actually - Across generates names like "col_fn", so :a with :sum => :a_sum
        # Let's test that if we have two functions with same name, it collides
        # across([:a, :b], [:sum => sum]) would create :a_sum and :b_sum (different names)
        # But across([:a], [:sum => sum, :sum => x -> sum(x)]) would create :a_sum twice
        ac = Across([:a], [:sum => sum, :sum => x -> sum(x)])
        @test_throws ArgumentError summarise(gt, ac)
    end

    @testset "mutate Across collision with existing column" begin
        tbl = (a = [1, 2, 3], a_abs = [10, 20, 30])
        ac = Across([:a], [:abs => abs])
        # Output :a_abs collides with existing column
        @test_throws ArgumentError mutate(tbl, ac)
    end

    @testset "valid Across operations work" begin
        tbl = (group = ["x", "x", "y"], a = [1, 2, 3], b = [4, 5, 6])
        ac = Across([:a, :b], [:sum => sum, :mean => mean])
        gt = groupby(tbl, :group)
        result = summarise(gt, ac)
        @test haskey(result, :a_sum)
        @test haskey(result, :b_sum)
        @test haskey(result, :a_mean)
        @test haskey(result, :b_mean)
    end
end

# ============================================================================
# select duplicate source column tests
# ============================================================================
@testset "select duplicate source handling" begin
    @testset "explicit spec errors on duplicate source" begin
        tbl = (a = [1, 2], b = [3, 4], c = [5, 6])
        # Selecting :a twice explicitly
        @test_throws ArgumentError select(tbl, :a, :a)
        @test_throws ArgumentError select(tbl, :x => :a, :y => :a)
        @test_throws ArgumentError select(tbl, :a, :x => :a)
    end

    @testset "selector silently skips already selected" begin
        tbl = (a = [1, 2], b = [3, 4], c = [5, 6])
        # everything() after explicit :a should skip :a
        result = select(tbl, :a, everything())
        @test collect(keys(result)) == [:a, :b, :c]
        @test result.a == [1, 2]
    end
end

# ============================================================================
# fill_missing edge cases
# ============================================================================
@testset "fill_missing edge cases" begin
    @testset "empty table" begin
        tbl = (a = Int[], b = Int[])
        # Should not throw on empty tables
        result = fill_missing(tbl, :a, direction=:down)
        @test isempty(result.a)
        result = fill_missing(tbl, :a, direction=:up)
        @test isempty(result.a)
    end
end

# ============================================================================
# pivot_wider type collision tests
# ============================================================================
@testset "pivot_wider type collision" begin
    @testset "distinct values stringifying to same Symbol" begin
        # 1 (Int) and "1" (String) both become :1
        tbl = (id = [1, 1], name = [1, "1"], value = [10, 20])
        @test_throws ArgumentError pivot_wider(tbl; id_cols=:id, names_from=:name, values_from=:value)
    end

    @testset "same type values work" begin
        tbl = (id = [1, 1], name = ["a", "b"], value = [10, 20])
        result = pivot_wider(tbl; id_cols=:id, names_from=:name, values_from=:value)
        @test haskey(result, :a)
        @test haskey(result, :b)
    end
end

# ============================================================================
# pivot_longer overlap and defaults tests
# ============================================================================
@testset "pivot_longer overlap and defaults" begin
    @testset "id_cols and value_cols overlap" begin
        tbl = (a = [1, 2], b = [3, 4], c = [5, 6])
        @test_throws ArgumentError pivot_longer(tbl; id_cols=[:a], value_cols=[:a, :b])
    end

    @testset "both empty means all columns are values" begin
        tbl = (a = [1, 2], b = [3, 4])
        result = pivot_longer(tbl)
        # With no id_cols, all columns become values
        @test !haskey(result, :a)
        @test !haskey(result, :b)
        @test haskey(result, :variable)
        @test haskey(result, :value)
        @test length(result.value) == 4  # 2 rows * 2 columns
    end
end

# ============================================================================
# arrange direction validation tests
# ============================================================================
@testset "arrange direction validation" begin
    tbl = (a = [3, 1, 2], b = [1, 2, 3])

    @testset "explicit ascending" begin
        result = arrange(tbl, :a => :asc)
        @test result.a == [1, 2, 3]
        result = arrange(tbl, :a => :ascending)
        @test result.a == [1, 2, 3]
    end

    @testset "explicit descending" begin
        result = arrange(tbl, :a => :desc)
        @test result.a == [3, 2, 1]
        result = arrange(tbl, :a => :descending)
        @test result.a == [3, 2, 1]
        result = arrange(tbl, :a => :reverse)
        @test result.a == [3, 2, 1]
    end

    @testset "invalid direction throws" begin
        @test_throws ArgumentError arrange(tbl, :a => :invalid)
        @test_throws ArgumentError arrange(tbl, :a => false)
        @test_throws ArgumentError arrange(tbl, :a => true)
    end
end

# ============================================================================
# join by=nothing column order preservation tests
# ============================================================================
@testset "join by=nothing preserves left column order" begin
    # Left table has columns in specific order
    left = (x = [1, 2], y = [10, 20], z = [100, 200])
    # Right table has columns in different order
    right = (z = [100, 200], x = [1, 2], w = [1000, 2000])
    # Common columns are :x and :z, left table order should be preserved
    result = inner_join(left, right)
    # Join should use keys in left table order: [:x, :z] not [:z, :x]
    @test result.x == [1, 2]
    @test result.z == [100, 200]
end

# ============================================================================
# right_join column order preservation tests
# ============================================================================
@testset "right_join preserves right column order" begin
    left = (id = [1, 2], a = [10, 20], b = [100, 200])
    right = (b = [100, 200], id = [1, 2], c = [1000, 2000])

    result = right_join(left, right, by=:id)
    @test propertynames(result) == (:b, :id, :c, :a, :b_x)
end

# ============================================================================
# distinct with missing values tests
# ============================================================================
@testset "distinct with missing values" begin
    @testset "missing values are deduplicated correctly" begin
        tbl = (a = [1, missing, 2, missing, 1], b = [10, 20, 30, 40, 50])
        result = distinct(tbl, :a; keep_all=true)
        # Should have 3 unique values: 1, missing, 2
        @test length(result.a) == 3
        @test 1 in result.a
        @test 2 in result.a
        @test any(ismissing, result.a)
    end

    @testset "missing in multi-column key" begin
        tbl = (a = [1, 1, missing, missing], b = [missing, missing, 1, 1], c = [10, 20, 30, 40])
        result = distinct(tbl, :a, :b; keep_all=true)
        # (1, missing) appears twice, (missing, 1) appears twice
        # Should deduplicate to 2 rows
        @test length(result.a) == 2
    end
end

# ============================================================================
# complete with missing values tests
# ============================================================================
@testset "complete with missing values" begin
    @testset "missing in key column doesn't duplicate" begin
        tbl = (a = [1, missing, 2], b = ["x", "y", "z"], val = [10, 20, 30])
        result = complete(tbl, :a)
        # unique values of :a are [1, missing, 2] - 3 values
        # Should not create duplicate rows for missing
        @test length(result.a) == 3
    end
end

# ============================================================================
# pivot_longer row order tests
# ============================================================================
@testset "pivot_longer row-major order" begin
    @testset "output is row-major (matches tidyr)" begin
        wide = (id = [1, 2], A = [10, 20], B = [100, 200])
        result = pivot_longer(wide; id_cols=:id, names_to=:name, values_to=:value)

        # Row-major order: for each input row, output all value columns
        # Row 1: (1, A, 10), (1, B, 100)
        # Row 2: (2, A, 20), (2, B, 200)
        @test result.id == [1, 1, 2, 2]
        @test result.name == ["A", "B", "A", "B"]
        @test result.value == [10, 100, 20, 200]
    end

    @testset "docstring example produces correct output" begin
        wide = (date = ["2024-01", "2024-02"],
                A = [100, 110],
                B = [200, 220],
                C = [300, 330])
        result = pivot_longer(wide; id_cols=:date, names_to=:series, values_to=:value)

        # As documented in the docstring
        @test result.date == ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02"]
        @test result.series == ["A", "B", "C", "A", "B", "C"]
        @test result.value == [100, 200, 300, 110, 220, 330]
    end
end

# ============================================================================
# PanelData select structural column protection tests
# ============================================================================
@testset "PanelData select rejects renaming structural columns" begin
    using Durbyn.ModelSpecs

    @testset "rejects renaming group column" begin
        data = (series = ["A", "A", "B", "B"],
                date = [1, 2, 1, 2],
                value = [10, 20, 100, 200])
        panel = PanelData(data; groupby=:series, date=:date)

        # Attempting to rename group column via select should error
        @test_throws ArgumentError select(panel, :new_series => :series, :value)
    end

    @testset "rejects renaming date column" begin
        data = (series = ["A", "A", "B", "B"],
                date = [1, 2, 1, 2],
                value = [10, 20, 100, 200])
        panel = PanelData(data; groupby=:series, date=:date)

        # Attempting to rename date column via select should error
        @test_throws ArgumentError select(panel, :new_date => :date, :value)
    end

    @testset "allows non-structural column renaming" begin
        data = (series = ["A", "A", "B", "B"],
                date = [1, 2, 1, 2],
                value = [10, 20, 100, 200])
        panel = PanelData(data; groupby=:series, date=:date)

        # Renaming non-structural columns should work
        result = select(panel, :renamed_value => :value)
        @test haskey(result.data, :series)  # structural kept
        @test haskey(result.data, :renamed_value)
        @test !haskey(result.data, :value)  # original renamed away
    end

    @testset "allows selecting structural columns without rename" begin
        data = (series = ["A", "A", "B", "B"],
                date = [1, 2, 1, 2],
                value = [10, 20, 100, 200],
                extra = [1, 2, 3, 4])
        panel = PanelData(data; groupby=:series, date=:date)

        # Just selecting (not renaming) structural columns is fine
        result = select(panel, :series, :date, :value)
        @test haskey(result.data, :series)
        @test haskey(result.data, :date)
        @test haskey(result.data, :value)
        @test !haskey(result.data, :extra)
    end
end

# ============================================================================
# Cross join tests (by=[])
# ============================================================================
@testset "cross join with by=[]" begin
    @testset "inner_join cross join" begin
        left = (a = [1, 2], x = [10, 20])
        right = (b = ["A", "B", "C"], y = [100, 200, 300])

        # by=[] produces Cartesian product
        result = inner_join(left, right, by=[])
        @test length(result.a) == 6  # 2 * 3
        @test result.a == [1, 1, 1, 2, 2, 2]
        @test result.b == ["A", "B", "C", "A", "B", "C"]
        @test result.x == [10, 10, 10, 20, 20, 20]
        @test result.y == [100, 200, 300, 100, 200, 300]
    end

    @testset "left_join cross join" begin
        left = (a = [1, 2], x = [10, 20])
        right = (b = ["A", "B"], y = [100, 200])

        result = left_join(left, right, by=[])
        @test length(result.a) == 4  # 2 * 2
        @test result.a == [1, 1, 2, 2]
        @test result.b == ["A", "B", "A", "B"]
    end

    @testset "right_join cross join" begin
        left = (a = [1, 2], x = [10, 20])
        right = (b = ["A", "B"], y = [100, 200])

        result = right_join(left, right, by=[])
        @test length(result.a) == 4
        @test result.a == [1, 2, 1, 2]
        @test result.b == ["A", "A", "B", "B"]
    end

    @testset "full_join cross join" begin
        left = (a = [1, 2], x = [10, 20])
        right = (b = ["A", "B"], y = [100, 200])

        # For cross join, full_join = inner_join (no unmatched rows)
        result = full_join(left, right, by=[])
        @test length(result.a) == 4
    end

    @testset "cross join with empty Any[]" begin
        left = (a = [1, 2], x = [10, 20])
        right = (b = ["A", "B"], y = [100, 200])

        # Also works with Any[]
        result = inner_join(left, right, by=Any[])
        @test length(result.a) == 4
    end
end

# ============================================================================
# Join key column should not be suffixed
# ============================================================================
@testset "join key columns not suffixed when right non-key collides" begin
    # Scenario: left.id is the key, right has non-key :id
    # The key column :id should NOT be renamed to :id_x
    left = (id = [1, 2], x = [10, 20])
    right = (key = [1, 2], id = [100, 200], y = [1000, 2000])

    @testset "inner_join preserves key name" begin
        result = inner_join(left, right, by=[:id => :key])
        @test haskey(result, :id)       # key column unchanged
        @test !haskey(result, :id_x)    # key NOT suffixed
        @test haskey(result, :id_y)     # right non-key suffixed
        @test result.id == [1, 2]
    end

    @testset "left_join preserves key name" begin
        result = left_join(left, right, by=[:id => :key])
        @test haskey(result, :id)
        @test !haskey(result, :id_x)
        @test haskey(result, :id_y)
    end

    @testset "right_join preserves key name" begin
        result = right_join(left, right, by=[:id => :key])
        @test haskey(result, :id)
        @test !haskey(result, :id_x)
        @test haskey(result, :id_y)
    end

    @testset "full_join preserves key name" begin
        result = full_join(left, right, by=[:id => :key])
        @test haskey(result, :id)
        @test !haskey(result, :id_x)
        @test haskey(result, :id_y)
    end
end

# ============================================================================
# PanelData select date column handling
# ============================================================================
@testset "PanelData select date column handling" begin
    using Durbyn.ModelSpecs

    @testset "select without date sets panel.date to nothing" begin
        data = (series = ["A", "A", "B", "B"],
                date = [1, 2, 1, 2],
                value = [10, 20, 100, 200])
        panel = PanelData(data; groupby=:series, date=:date)
        @test panel.date == :date

        # Select only value (date not included)
        result = select(panel, :value)
        @test haskey(result.data, :series)  # group always kept
        @test haskey(result.data, :value)
        @test !haskey(result.data, :date)   # date dropped
        @test result.date === nothing       # metadata updated
    end

    @testset "select with date preserves panel.date" begin
        data = (series = ["A", "A", "B", "B"],
                date = [1, 2, 1, 2],
                value = [10, 20, 100, 200])
        panel = PanelData(data; groupby=:series, date=:date)

        result = select(panel, :date, :value)
        @test haskey(result.data, :date)
        @test result.date == :date  # metadata preserved
    end

    @testset "zero-arg select returns grouping columns only" begin
        data = (series = ["A", "A", "B", "B"],
                date = [1, 2, 1, 2],
                value = [10, 20, 100, 200])
        panel = PanelData(data; groupby=:series, date=:date)

        result = select(panel)
        @test haskey(result.data, :series)
        @test !haskey(result.data, :date)
        @test result.date === nothing
    end

    @testset "zero-arg select on ungrouped panel throws error" begin
        data = (date = [1, 2, 3], value = [10, 20, 30])
        panel = PanelData(data; date=:date)  # no groupby
        @test isempty(panel.groups)

        @test_throws ArgumentError select(panel)
    end
end

# ============================================================================
# Join duplicate key validation
# ============================================================================
@testset "join duplicate key validation" begin
    @testset "duplicate keys in by vector throws error" begin
        left = (id = [1, 2], x = [10, 20])
        right = (id = [1, 2], y = [100, 200])

        @test_throws ArgumentError inner_join(left, right, by=[:id, :id])
    end

    @testset "duplicate left keys in by pairs throws error" begin
        left = (a = [1, 2], b = [3, 4], x = [10, 20])
        right = (c = [1, 2], d = [3, 4], y = [100, 200])

        # Same left key used twice
        @test_throws ArgumentError inner_join(left, right, by=[:a => :c, :a => :d])
    end

    @testset "duplicate right keys in by pairs throws error" begin
        left = (a = [1, 2], b = [3, 4], x = [10, 20])
        right = (c = [1, 2], d = [3, 4], y = [100, 200])

        # Same right key used twice
        @test_throws ArgumentError inner_join(left, right, by=[:a => :c, :b => :c])
    end
end

# ============================================================================
# groupby duplicate column validation
# ============================================================================
@testset "groupby duplicate column validation" begin
    @testset "duplicate grouping columns throws error" begin
        tbl = (a = [1, 2, 3], b = [4, 5, 6])
        @test_throws ArgumentError groupby(tbl, :a, :a)
    end

    @testset "duplicate in vector form throws error" begin
        tbl = (a = [1, 2, 3], b = [4, 5, 6])
        @test_throws ArgumentError groupby(tbl, [:a, :b, :a])
    end
end

# ============================================================================
# Empty PanelData operations preserve schema
# ============================================================================
@testset "empty PanelData operations preserve schema" begin
    using Durbyn.ModelSpecs

    @testset "mutate on empty grouped panel adds columns" begin
        # Create empty panel with groups
        data = (series = String[], date = Int[], value = Float64[])
        panel = PanelData(data; groupby=:series, date=:date)
        @test length(panel.data.series) == 0

        # mutate should still add the new column
        result = mutate(panel, doubled = d -> d.value .* 2)
        @test haskey(result.data, :doubled)
        @test length(result.data.series) == 0  # still empty
    end

    @testset "select on empty grouped panel works" begin
        data = (series = String[], date = Int[], value = Float64[], extra = Int[])
        panel = PanelData(data; groupby=:series, date=:date)

        result = select(panel, :value)
        @test haskey(result.data, :series)  # group kept
        @test haskey(result.data, :value)
        @test !haskey(result.data, :extra)  # not selected
    end
end

# ============================================================================
# bind_rows preserves eltypes for empty tables
# ============================================================================
@testset "bind_rows preserves eltypes" begin
    @testset "empty tables preserve element type" begin
        t1 = (a = Int[], b = Float64[])
        t2 = (a = Int[], b = Float64[])
        result = bind_rows(t1, t2)

        @test eltype(result.a) == Int
        @test eltype(result.b) == Float64
        @test length(result.a) == 0
    end

    @testset "mixed empty and non-empty preserves type" begin
        t1 = (a = Int[],)
        t2 = (a = [1, 2, 3],)
        result = bind_rows(t1, t2)

        @test eltype(result.a) == Int
        @test result.a == [1, 2, 3]
    end

    @testset "type promotion across tables" begin
        t1 = (a = Int[1, 2],)
        t2 = (a = Float64[3.0, 4.0],)
        result = bind_rows(t1, t2)

        # Should promote to Float64
        @test eltype(result.a) <: AbstractFloat
    end

    @testset "missing columns get Union{Missing, T} eltype" begin
        # Table 1 has :a (Int, empty), Table 2 has :b (Int, non-empty)
        t1 = (a = Int[],)
        t2 = (b = [1],)
        result = bind_rows(t1, t2)

        # Both columns should be Union{Missing, Int}
        @test eltype(result.a) == Union{Missing, Int}
        @test eltype(result.b) == Union{Missing, Int}
        @test isequal(result.a, [missing])
        @test result.b == [1]
    end
end

# ============================================================================
# summarise on empty grouped data preserves eltypes
# ============================================================================
@testset "summarise empty grouped data" begin
    @testset "key columns preserve eltype" begin
        tbl = (x = Int[], y = Float64[])
        gt = groupby(tbl, :x)
        result = summarise(gt, total = :y => sum)

        @test eltype(result.x) == Int
        @test length(result.x) == 0
    end

    @testset "summary columns infer eltype from function" begin
        tbl = (x = Int[], y = Int[])
        gt = groupby(tbl, :x)
        result = summarise(gt, total = :y => sum)

        # sum on empty Int[] returns Int (0)
        @test eltype(result.total) == Int
    end

    @testset "mean on empty Float64 infers Float64" begin
        using Statistics
        tbl = (x = Int[], y = Float64[])
        gt = groupby(tbl, :x)
        result = summarise(gt, avg = :y => mean)

        @test eltype(result.avg) == Float64
    end

    @testset "invalid column throws error on empty data" begin
        tbl = (g = Int[], x = Float64[])
        gt = groupby(tbl, :g)

        # :y does not exist - should throw KeyError even on empty data
        @test_throws KeyError summarise(gt, bad = :y => sum)
    end
end

# =============================================================================
# Cross join eltype preservation
# =============================================================================
@testset "cross join eltype preservation" begin
    left = (a = [1, 2], x = [10, 20])
    right = (b = [3, 4], y = [30.0, 40.0])

    @testset "left_join cross preserves right column eltype" begin
        result = left_join(left, right, by=Symbol[])
        @test eltype(result.y) == Float64  # Not Union{Missing, Float64}
        @test eltype(result.x) == Int
    end

    @testset "right_join cross preserves left column eltype" begin
        result = right_join(left, right, by=Symbol[])
        @test eltype(result.x) == Int  # Not Union{Missing, Int}
        @test eltype(result.y) == Float64
    end

    @testset "full_join cross preserves both column eltypes" begin
        result = full_join(left, right, by=Symbol[])
        @test eltype(result.x) == Int  # Not Union{Missing, Int}
        @test eltype(result.y) == Float64  # Not Union{Missing, Float64}
    end
end

# =============================================================================
# across with inference-failing functions
# =============================================================================
@testset "across with median preserves eltype" begin
    using Statistics

    tbl = (g = [1, 1, 2, 2], x = [1.0, 2.0, 3.0, 4.0])
    gt = groupby(tbl, :g)

    @testset "median result is Float64 not Any" begin
        result = summarise(gt, across([:x], :med => median))
        @test eltype(result.x_med) == Float64
        @test result.x_med == [1.5, 3.5]
    end

    @testset "mixed inference-failing and passing functions" begin
        result = summarise(gt, across([:x], :med => median, :sum => sum))
        @test eltype(result.x_med) == Float64
        @test eltype(result.x_sum) == Float64
    end
end
