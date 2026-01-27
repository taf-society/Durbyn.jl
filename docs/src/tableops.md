# Table Operations

The `TableOps` module provides a comprehensive set of data manipulation functions for working with Tables.jl-compatible data sources. These functions enable common data wrangling tasks like filtering, grouping, pivoting, and summarizing data.

## Overview

`TableOps` is inspired by popular data manipulation libraries like dplyr and tidyr (R) and pandas (Python), but designed specifically for Julia's Tables.jl ecosystem. All functions work seamlessly with any Tables.jl-compatible data source, including:

- NamedTuples
- DataFrames
- CSV.File objects
- Arrow.Table objects
- And many others

## Time Series and Panel Data

**Durbyn.jl is a forecasting package**, and time series data manipulation is at its core. The `TableOps` module is designed to work seamlessly with `PanelData`, a specialized data structure for handling multiple time series (panel/longitudinal data).

### What is PanelData?

`PanelData` wraps your tabular data with metadata that defines:
- **Grouping columns**: Which columns identify individual time series (e.g., `:series_id`, `:store`, `:product`)
- **Date column**: Which column contains the time index
- **Seasonal period (`m`)**: The number of observations per seasonal cycle (e.g., 12 for monthly data with yearly seasonality)
- **Frequency**: The time frequency (`:daily`, `:weekly`, `:monthly`, `:quarterly`, `:yearly`)
- **Target column**: The variable to forecast

### Why PanelData Matters for Forecasting

When working with forecasting, you typically need to:
1. **Process multiple series independently** - Each series may have different scales, patterns, and missing values
2. **Preserve time ordering** - Operations should respect the temporal structure
3. **Handle missing time points** - Gaps in time series need special treatment
4. **Compute group-relative features** - Features like "deviation from series mean" require within-group calculations

`PanelData` enables all of this by automatically applying operations **within each group** while preserving the panel structure.

### Quick Example

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

# Create panel data with multiple time series
data = (
    series = ["A", "A", "A", "A", "B", "B", "B", "B"],
    date = [1, 2, 3, 4, 1, 2, 3, 4],
    value = [100, 110, 105, 120, 500, 520, 510, 540]
)

# Wrap in PanelData - this defines the panel structure
panel = PanelData(data; groupby=:series, date=:date, m=12)

# Now TableOps functions automatically work within each series
# Compute series-relative features
result = mutate(panel,
    series_mean = d -> fill(mean(d.value), length(d.value)),
    deviation = d -> d.value .- mean(d.value),
    pct_change = d -> [missing; diff(d.value) ./ d.value[1:end-1] .* 100]
)

# Fill missing values within each series (not across series!)
filled = fill_missing(result, :pct_change; direction=:down)
```

### When to Use PanelData vs Regular Tables

| Use Case | Data Type | Why |
|----------|-----------|-----|
| Single time series | Regular table | No grouping needed |
| Multiple independent series | `PanelData` | Operations apply per-series |
| Cross-sectional data | Regular table | No time structure |
| Panel/longitudinal data | `PanelData` | Groups + time structure |
| Forecasting preparation | `PanelData` | Preserves metadata for models |

For detailed PanelData operations, see the [PanelData Operations](#paneldata-operations) section below.

---

## Getting Started

```julia
using CSV
using Downloads
using Tables
using Durbyn
using Durbyn.TableOps
using Durbyn.Grammar
using Durbyn.ModelSpecs

# Download example retail data
local_path = Downloads.download("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/refs/heads/main/Data/retail.csv")
retail = CSV.File(local_path)
tbl = Tables.columntable(retail)

# Preview the data
glimpse(tbl)
```

## Function Reference

### Quick Reference Table

| Category | Function | Description |
|----------|----------|-------------|
| **Preview** | `glimpse` | Quick data preview with types and samples |
| **Select** | `select` | Select and rename columns |
| | `rename` | Rename columns (keep all) |
| | `all_of` | Select columns by name vector |
| | `everything` | Select all columns |
| **Filter** | `query` | Filter rows by predicate |
| | `distinct` | Remove duplicate rows |
| **Sort** | `arrange` | Sort rows by columns |
| **Transform** | `mutate` | Add or modify columns |
| | `across` | Apply functions across multiple columns |
| **Group** | `groupby` | Group data by columns |
| | `ungroup` | Remove grouping |
| | `summarise` | Aggregate grouped data |
| **Reshape** | `pivot_longer` | Wide to long format |
| | `pivot_wider` | Long to wide format |
| **Combine** | `bind_rows` | Stack tables vertically |
| **Join** | `inner_join` | Keep only matching rows |
| | `left_join` | Keep all left rows |
| | `right_join` | Keep all right rows |
| | `full_join` | Keep all rows from both |
| | `semi_join` | Filter left by right keys |
| | `anti_join` | Filter left by missing right keys |
| **String** | `separate` | Split column into multiple |
| | `unite` | Combine columns into one |
| **Missing** | `fill_missing` | Fill missing values |
| | `complete` | Complete missing combinations |

---

## Core Functions

### `glimpse` - Quick Data Preview

Get a compact summary of your data, showing column names, types, and sample values.

```julia
using Durbyn.TableOps

tbl = (date = ["2024-01", "2024-02", "2024-03"],
       A = [100, 110, 120],
       B = [200, 220, 240],
       C = [300, 330, 360])

glimpse(tbl)
# Table glimpse
#   Rows: 3
#   Columns: 4
#   date                :: String  ["2024-01", "2024-02", "2024-03"]
#   A                   :: Int64   [100, 110, 120]
#   B                   :: Int64   [200, 220, 240]
#   C                   :: Int64   [300, 330, 360]
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `maxrows` (keyword, default: 5) - Maximum number of sample values to show
- `io` (keyword, default: stdout) - Output stream

---

### `select` - Choose and Rename Columns

Select specific columns from your data, optionally renaming them.

```julia
using Durbyn.TableOps

tbl = (id = [1, 2, 3],
       name = ["Shler", "Rivka", "Dilan"],
       age = [25, 30, 35],
       salary = [50000, 60000, 70000])

# Select specific columns
select(tbl, :name, :age)
# Output: (name = ["Shler", "Rivka", "Dilan"], age = [25, 30, 35])

# Rename while selecting
select(tbl, :employee => :name, :years => :age)
# Output: (employee = ["Shler", "Rivka", "Dilan"], years = [25, 30, 35])

# Mix selection and renaming
select(tbl, :id, :employee_name => :name)
# Output: (id = [1, 2, 3], employee_name = ["Shler", "Rivka", "Dilan"])
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `specs...` - Column specifications as `Symbol`s or `Pair{Symbol,Symbol}` for renaming

---

### `rename` - Rename Columns

Rename columns while keeping all columns in the table.

```julia
using Durbyn.TableOps

tbl = (a = [1, 2, 3], b = [4, 5, 6], c = [7, 8, 9])

# Rename single column
rename(tbl, :x => :a)
# Output: (x = [1, 2, 3], b = [4, 5, 6], c = [7, 8, 9])

# Rename multiple columns
rename(tbl, :x => :a, :y => :b)
# Output: (x = [1, 2, 3], y = [4, 5, 6], c = [7, 8, 9])
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `specs...` - Rename specifications as `Pair{Symbol,Symbol}`: `:new_name => :old_name`

**Note:** Unlike `select`, `rename` keeps all columns - it only changes names of specified columns.

---

### `query` - Filter Rows

Filter rows based on custom conditions using a predicate function.

```julia
using Durbyn.TableOps

tbl = (product = ["A", "B", "C", "D", "E"],
       price = [10, 25, 15, 30, 20],
       quantity = [100, 50, 75, 25, 60])

# Filter rows where price > 15
query(tbl, row -> row.price > 15)
# Output: (product = ["B", "D", "E"], price = [25, 30, 20], quantity = [50, 25, 60])

# Multiple conditions
query(tbl, row -> row.price > 15 && row.quantity > 30)
# Output: (product = ["B", "E"], price = [25, 20], quantity = [50, 60])

# Using `in` for categorical filtering
query(tbl, row -> row.product in ["A", "C", "E"])
# Output: (product = ["A", "C", "E"], price = [10, 15, 20], quantity = [100, 75, 60])
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `predicate` - A function that takes a row (as NamedTuple) and returns `Bool`

**Handling Missing Values:**
The predicate must return exactly `true` or `false`. If your data contains `missing` values and comparisons might return `missing`, use `coalesce`:

```julia
# This throws an error if price contains missing:
query(tbl, row -> row.price > 15)

# Use coalesce to treat missing as false:
query(tbl, row -> coalesce(row.price > 15, false))
```

---

### `distinct` - Remove Duplicate Rows

Remove duplicate rows based on specified columns.

```julia
using Durbyn.TableOps

tbl = (a = [1, 1, 2, 2, 3],
       b = [1, 1, 2, 3, 3],
       c = [10, 20, 30, 40, 50])

# Distinct by all columns (removes exact duplicate rows)
distinct(tbl)
# Output: (a = [1, 1, 2, 2, 3], b = [1, 1, 2, 3, 3], c = [10, 20, 30, 40, 50])
# (no duplicates in this case)

# Distinct by specific column - keeps only specified columns
distinct(tbl, :a)
# Output: (a = [1, 2, 3],)

# Distinct by specific column but keep all columns
distinct(tbl, :a; keep_all=true)
# Output: (a = [1, 2, 3], b = [1, 2, 3], c = [10, 30, 50])
# (keeps first occurrence of each unique value)

# Distinct by multiple columns
distinct(tbl, :a, :b)
# Output: (a = [1, 2, 2, 3], b = [1, 2, 3, 3])
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `cols...` - Column names to consider for uniqueness (if empty, uses all columns)
- `keep_all` (keyword, default: false) - If true, keep all columns; if false, only keep specified columns

---

### `arrange` - Sort Data

Sort rows by one or more columns in ascending or descending order.

```julia
using Durbyn.TableOps

tbl = (name = ["Shler", "Rivka", "Dilan", "Moshe"],
       department = ["Sales", "IT", "Sales", "IT"],
       salary = [60000, 70000, 55000, 75000])

# Sort by salary (ascending)
arrange(tbl, :salary)
# Output: (name = ["Dilan", "Shler", "Rivka", "Moshe"],
#          department = ["Sales", "Sales", "IT", "IT"],
#          salary = [55000, 60000, 70000, 75000])

# Sort by salary (descending)
arrange(tbl, :salary => :desc)
# Output: (name = ["Moshe", "Rivka", "Shler", "Dilan"],
#          department = ["IT", "IT", "Sales", "Sales"],
#          salary = [75000, 70000, 60000, 55000])

# Multi-column sort: department ascending, then salary descending
arrange(tbl, :department, :salary => :desc)
# Output: (name = ["Moshe", "Rivka", "Shler", "Dilan"],
#          department = ["IT", "IT", "Sales", "Sales"],
#          salary = [75000, 70000, 60000, 55000])
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `cols...` - Column specifications: `Symbol` for ascending, `Pair` (`:col => :desc`) for descending
- `rev` (keyword, default: false) - If true, reverse the entire final sort order

**Descending indicators:** `:desc`, `:descending`, `:reverse`, or `false`

---

### `mutate` - Add or Modify Columns

Create new columns or modify existing ones based on computations.

```julia
using Durbyn.TableOps

tbl = (product = ["A", "B", "C"],
       price = [10.0, 20.0, 15.0],
       quantity = [100, 50, 75])

# Add a new column
mutate(tbl, revenue = data -> data.price .* data.quantity)
# Output: (product = ["A", "B", "C"],
#          price = [10.0, 20.0, 15.0],
#          quantity = [100, 50, 75],
#          revenue = [1000.0, 1000.0, 1125.0])

# Add multiple columns
mutate(tbl,
    revenue = data -> data.price .* data.quantity,
    discounted_price = data -> data.price .* 0.9)

# Modify existing column
mutate(tbl, price = data -> data.price .* 1.1)  # 10% price increase

# Reference previously created columns (sequential evaluation)
mutate(tbl,
    revenue = data -> data.price .* data.quantity,
    revenue_per_unit = data -> data.revenue ./ data.quantity)  # Uses newly created revenue
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `kwargs...` - Named arguments where name is column name and value is either:
  - A function `data -> Vector` that computes the column
  - A vector of values (must match row count)

---

### `groupby` - Group Data

Group rows by unique combinations of values in specified columns.

```julia
using Durbyn.TableOps

tbl = (department = ["Sales", "IT", "Sales", "IT", "Sales"],
       employee = ["Shler", "Rivka", "Dilan", "Moshe", "Jwan"],
       salary = [60000, 70000, 55000, 75000, 65000])

# Group by department
gt = groupby(tbl, :department)
# Output: GroupedTable(2 groups by department)

glimpse(gt)
# GroupedTable glimpse
#   Groups: 2
#   Key columns: department
#   Rows: 5 (avg 2.5, min 2, max 3)
#   Group 1: (department = "IT",) (2 rows)
#     ...
#   Group 2: (department = "Sales",) (3 rows)
#     ...

# Group by multiple columns
sales_data = (region = ["North", "South", "North", "South"],
              product = ["A", "A", "B", "B"],
              revenue = [1000, 1500, 2000, 2500])

gt = groupby(sales_data, :region, :product)
# Output: GroupedTable(4 groups by region, product)
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `cols...` - One or more column names (as `Symbol`s) to group by

**Returns:** A `GroupedTable` object for use with `summarise` or `ungroup`

---

### `ungroup` - Remove Grouping

Remove grouping from a `GroupedTable`, returning the underlying data.

```julia
using Durbyn.TableOps

tbl = (category = ["A", "B", "A", "B"],
       value = [1, 2, 3, 4])

gt = groupby(tbl, :category)
# Output: GroupedTable(2 groups by category)

# Remove grouping
result = ungroup(gt)
# Output: (category = ["A", "B", "A", "B"], value = [1, 2, 3, 4])
```

**Parameters:**
- `gt` - A `GroupedTable` created by `groupby`

**Returns:** The original `NamedTuple` data without grouping

---

### `summarise` / `summarize` - Aggregate Data

Compute summary statistics for each group in a GroupedTable.

```julia
using Durbyn.TableOps
using Statistics

tbl = (department = ["Sales", "IT", "Sales", "IT", "Sales"],
       employee = ["Shler", "Rivka", "Dilan", "Moshe", "Jwan"],
       salary = [60000, 70000, 55000, 75000, 65000])

gt = groupby(tbl, :department)

# Compute mean salary per department
stbl = summarise(gt, avg_salary = :salary => mean)
# Output: (department = ["IT", "Sales"], avg_salary = [72500.0, 60000.0])

# Multiple summary statistics
summarise(gt,
    avg_salary = :salary => mean,
    min_salary = :salary => minimum,
    max_salary = :salary => maximum,
    count = data -> length(data.salary))
# Output: (department = ["IT", "Sales"],
#          avg_salary = [72500.0, 60000.0],
#          min_salary = [70000, 55000],
#          max_salary = [75000, 65000],
#          count = [2, 3])

# Multi-column aggregation
summarise(gt,
    salary_range = (:salary,) => col -> maximum(col) - minimum(col))
```

**Parameters:**
- `gt` - A `GroupedTable` created by `groupby`
- `kwargs...` - Named summary specifications where each value can be:
  - `:column => function` - Apply function to a specific column
  - `(:col1, :col2) => function` - Apply function to multiple columns
  - `data -> scalar` - Function taking the entire group data

**Note:** `summarize` is an alias for `summarise` (American English spelling).

---

## Column Selection Helpers

### `all_of` - Select Columns by Name Vector

Select columns using a vector of column names. Useful when column names are stored in a variable.

```julia
using Durbyn.TableOps

tbl = (a = [1, 2], b = [3, 4], c = [5, 6], d = [7, 8])

# Select columns from a vector
cols_to_select = [:a, :c]
select(tbl, all_of(cols_to_select))
# Output: (a = [1, 2], c = [5, 6])

# Useful for programmatic column selection
numeric_cols = [:a, :b]
select(tbl, all_of(numeric_cols))
# Output: (a = [1, 2], b = [3, 4])
```

**Parameters:**
- `cols` - A vector of column names (as `Symbol`s or `String`s)

---

### `everything` - Select All Columns

Select all columns. Useful for reordering columns or combining with other selections.

```julia
using Durbyn.TableOps

tbl = (a = [1, 2], b = [3, 4], c = [5, 6])

# Select all columns
select(tbl, everything())
# Output: (a = [1, 2], b = [3, 4], c = [5, 6])

# Reorder: put :c first, then all others
select(tbl, :c, everything())
# Output: (c = [5, 6], a = [1, 2], b = [3, 4])

# Reorder: put :b and :c first
select(tbl, :b, :c, everything())
# Output: (b = [3, 4], c = [5, 6], a = [1, 2])
```

**Note:** When combining with other selectors, columns are deduplicated (each column appears only once).

---

### `across` - Apply Functions Across Columns

Apply one or more functions across multiple columns. Used with `mutate` or `summarise`.

```julia
using Durbyn.TableOps
using Statistics

# With summarise
tbl = (group = ["A", "A", "B", "B"],
       x = [1.0, 2.0, 3.0, 4.0],
       y = [10.0, 20.0, 30.0, 40.0])

gt = groupby(tbl, :group)

# Apply mean to multiple columns
summarise(gt, across([:x, :y], :mean => mean))
# Output: (group = ["A", "B"], x_mean = [1.5, 3.5], y_mean = [15.0, 35.0])

# Multiple functions
summarise(gt, across([:x, :y], :mean => mean, :sum => sum))
# Output: (group = ["A", "B"],
#          x_mean = [1.5, 3.5], x_sum = [3.0, 7.0],
#          y_mean = [15.0, 35.0], y_sum = [30.0, 70.0])

# With everything() - applies to all non-grouping columns
summarise(gt, across(everything(), :mean => mean))
# Output: (group = ["A", "B"], x_mean = [1.5, 3.5], y_mean = [15.0, 35.0])

# With mutate
tbl2 = (a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0])
mutate(tbl2, across([:a, :b], :squared => x -> x .^ 2))
# Output: (a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0],
#          a_squared = [1.0, 4.0, 9.0], b_squared = [16.0, 25.0, 36.0])
```

**Parameters:**
- `cols` - Column specification: vector of symbols, `all_of(...)`, or `everything()`
- `fns...` - One or more `Pair{Symbol, Function}`: `:name => function`

**Output column naming:** `{original_column}_{function_name}`

---

## Reshape Functions

### `pivot_longer` - Wide to Long Format

Transform data from wide format to long format by pivoting columns into rows.

```julia
using Durbyn.TableOps

# Wide format data
wide = (date = ["2024-01", "2024-02", "2024-03"],
        A = [100, 110, 120],
        B = [200, 220, 240],
        C = [300, 330, 360])

# Convert to long format
long = pivot_longer(wide, id_cols=:date, names_to=:series, values_to=:value)
# Output: (date = ["2024-01", "2024-01", "2024-01", "2024-02", ...],
#          series = ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
#          value = [100, 200, 300, 110, 220, 330, 120, 240, 360])

glimpse(long)
# Table glimpse
#   Rows: 9
#   Columns: 3
#   date    :: String  ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", ...]
#   series  :: String  ["A", "B", "C", "A", "B", ...]
#   value   :: Int64   [100, 200, 300, 110, 220, ...]

# Specify which columns to pivot
pivot_longer(wide, id_cols=:date, value_cols=[:A, :B], names_to=:series, values_to=:value)
# Only pivots A and B columns, C is excluded
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `id_cols` (keyword) - Column(s) to keep as identifiers (not pivoted)
- `value_cols` (keyword) - Column(s) to pivot (if empty, all non-id columns)
- `names_to` (keyword, default: `:variable`) - Name for the column containing original column names
- `values_to` (keyword, default: `:value`) - Name for the column containing values

**Column Selection Logic:**
- If both `id_cols` and `value_cols` are empty: all columns become value columns
- If only `id_cols` is provided: all other columns become value columns
- If only `value_cols` is provided: all other columns become id columns
- If both are provided: unspecified columns are added to `id_cols` (not dropped)

---

### `pivot_wider` - Long to Wide Format

Transform data from long format to wide format by spreading rows into columns.

```julia
using Durbyn.TableOps

# Long format data
long = (date = ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02"],
        series = ["A", "B", "C", "A", "B", "C"],
        value = [100, 200, 300, 110, 220, 330])

# Convert to wide format
wide = pivot_wider(long, names_from=:series, values_from=:value, id_cols=:date)
# Output: (date = ["2024-01", "2024-02"],
#          A = [100, 110],
#          B = [200, 220],
#          C = [300, 330])

# Sort column names alphabetically
pivot_wider(long, names_from=:series, values_from=:value,
            id_cols=:date, sort_names=true)

# Handle missing combinations with custom fill value
incomplete = (id = [1, 1, 2], category = ["A", "B", "A"], val = [10, 20, 30])
pivot_wider(incomplete, names_from=:category, values_from=:val, fill=0)
# Output: (id = [1, 2], A = [10, 30], B = [20, 0])
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `names_from` - Column containing values to become new column names
- `values_from` - Column containing values to populate new columns
- `id_cols` (keyword) - Column(s) that uniquely identify each row
- `fill` (keyword, default: `missing`) - Value for missing combinations
- `sort_names` (keyword, default: false) - Sort new column names alphabetically

---

## Combine Functions

### `bind_rows` - Stack Tables Vertically

Combine multiple tables by stacking rows. Handles mismatched columns by filling with `missing`.

```julia
using Durbyn.TableOps

# Tables with same columns
tbl1 = (a = [1, 2], b = [3, 4])
tbl2 = (a = [5, 6], b = [7, 8])

bind_rows(tbl1, tbl2)
# Output: (a = [1, 2, 5, 6], b = [3, 4, 7, 8])

# Tables with different columns
tbl3 = (a = [1, 2], b = [3, 4])
tbl4 = (a = [5, 6], c = [7, 8])

bind_rows(tbl3, tbl4)
# Output: (a = [1, 2, 5, 6],
#          b = Union{Missing, Int64}[3, 4, missing, missing],
#          c = Union{Missing, Int64}[missing, missing, 7, 8])

# Multiple tables
tbl5 = (x = [1], y = [2])
tbl6 = (x = [3], y = [4])
tbl7 = (x = [5], y = [6])

bind_rows(tbl5, tbl6, tbl7)
# Output: (x = [1, 3, 5], y = [2, 4, 6])
```

**Parameters:**
- `tables...` - Two or more Tables.jl-compatible data sources

**Note:** Column order is determined by the order columns first appear across all tables.

---

## Join Functions

Join functions combine two tables based on matching key columns. TableOps provides six types of joins to handle different use cases.

### Join Types Overview

| Join Type | Keeps | Use Case |
|-----------|-------|----------|
| `inner_join` | Rows matching in **both** tables | Find common records |
| `left_join` | **All** left rows + matching right | Enrich left data |
| `right_join` | **All** right rows + matching left | Enrich right data |
| `full_join` | **All** rows from both tables | Complete union |
| `semi_join` | Left rows **with** match (no right columns) | Filter by existence |
| `anti_join` | Left rows **without** match | Find missing records |

> **Note:** Joins use Julia's `isequal` semantics for key matching: `missing` matches `missing`
> and `NaN` matches `NaN`. This differs from SQL where `NULL` never equals `NULL`.

### `by` Parameter Specification

All join functions accept a `by` parameter to specify join keys:

```julia
# Auto-detect common columns
inner_join(left, right)

# Single column (same name in both)
inner_join(left, right, by=:id)

# Multiple columns (same names)
inner_join(left, right, by=[:id, :date])

# Different column names
inner_join(left, right, by=:left_id => :right_id)

# Multiple different column names
inner_join(left, right, by=[:id => :key, :date => :timestamp])
```

---

### `inner_join` - Keep Matching Rows

Return only rows where keys exist in **both** tables.

```julia
using Durbyn.TableOps

left = (id = [1, 2, 3], x = [10, 20, 30])
right = (id = [2, 3, 4], y = [200, 300, 400])

inner_join(left, right, by=:id)
# Output: (id = [2, 3], x = [20, 30], y = [200, 300])
# Only ids 2 and 3 are in both tables
```

**Parameters:**
- `left` - Left table
- `right` - Right table
- `by` (keyword) - Join key specification
- `suffix` (keyword, default: `("_x", "_y")`) - Suffixes for duplicate column names

---

### `left_join` - Keep All Left Rows

Return all rows from `left`, with matching data from `right`. Non-matching rows have `missing` for right columns.

```julia
using Durbyn.TableOps

left = (id = [1, 2, 3], x = [10, 20, 30])
right = (id = [2, 3, 4], y = [200, 300, 400])

left_join(left, right, by=:id)
# Output: (id = [1, 2, 3], x = [10, 20, 30], y = [missing, 200, 300])
# All left rows kept; id=1 has no match, so y is missing
```

**Use Case:** Enriching a primary dataset with additional information while preserving all original records.

**Parameters:**
- `left` - Left table (all rows preserved)
- `right` - Right table (only matching rows included)
- `by` (keyword) - Join key specification
- `suffix` (keyword, default: `("_x", "_y")`) - Suffixes for duplicate column names

---

### `right_join` - Keep All Right Rows

Return all rows from `right`, with matching data from `left`. Non-matching rows have `missing` for left columns.

```julia
using Durbyn.TableOps

left = (id = [1, 2, 3], x = [10, 20, 30])
right = (id = [2, 3, 4], y = [200, 300, 400])

right_join(left, right, by=:id)
# Output: (id = [2, 3, 4], x = [20, 30, missing], y = [200, 300, 400])
# All right rows kept; id=4 has no match, so x is missing
```

**Parameters:**
- `left` - Left table (only matching rows included)
- `right` - Right table (all rows preserved)
- `by` (keyword) - Join key specification
- `suffix` (keyword, default: `("_x", "_y")`) - Suffixes for duplicate column names

---

### `full_join` - Keep All Rows

Return all rows from both tables. Non-matching rows have `missing` for columns from the other table.

```julia
using Durbyn.TableOps

left = (id = [1, 2, 3], x = [10, 20, 30])
right = (id = [2, 3, 4], y = [200, 300, 400])

full_join(left, right, by=:id)
# Output: (id = [1, 2, 3, 4],
#          x = [10, 20, 30, missing],
#          y = [missing, 200, 300, 400])
# All ids present; missing values where no match
```

**Use Case:** Creating a complete view of all records from both sources.

**Parameters:**
- `left` - Left table
- `right` - Right table
- `by` (keyword) - Join key specification
- `suffix` (keyword, default: `("_x", "_y")`) - Suffixes for duplicate column names

---

### `semi_join` - Filter by Existence

Return rows from `left` where the key exists in `right`. No columns from `right` are added.

```julia
using Durbyn.TableOps

left = (id = [1, 2, 3, 4], x = [10, 20, 30, 40])
right = (id = [2, 4], y = [200, 400])

semi_join(left, right, by=:id)
# Output: (id = [2, 4], x = [20, 40])
# Only left columns; filtered to ids present in right
```

**Use Case:** Filtering a table to records that exist in another table (e.g., customers who have orders).

**Parameters:**
- `left` - Table to filter
- `right` - Table to check for key existence
- `by` (keyword) - Join key specification

**Note:** Unlike `inner_join`, no columns from `right` are added to the result.

---

### `anti_join` - Filter by Non-Existence

Return rows from `left` where the key does NOT exist in `right`. No columns from `right` are added.

```julia
using Durbyn.TableOps

left = (id = [1, 2, 3, 4], x = [10, 20, 30, 40])
right = (id = [2, 4], y = [200, 400])

anti_join(left, right, by=:id)
# Output: (id = [1, 3], x = [10, 30])
# Only left columns; filtered to ids NOT in right
```

**Use Case:** Finding records that don't have a match (e.g., customers without orders, missing data).

**Parameters:**
- `left` - Table to filter
- `right` - Table to check for key non-existence
- `by` (keyword) - Join key specification

---

### Join Examples

#### Multiple Key Columns

```julia
using Durbyn.TableOps

orders = (customer_id = [1, 1, 2, 2],
          product_id = ["A", "B", "A", "C"],
          quantity = [10, 20, 15, 25])

prices = (customer_id = [1, 2],
          product_id = ["A", "A"],
          price = [100.0, 95.0])

inner_join(orders, prices, by=[:customer_id, :product_id])
# Output: (customer_id = [1, 2], product_id = ["A", "A"],
#          quantity = [10, 15], price = [100.0, 95.0])
```

#### Different Column Names

```julia
using Durbyn.TableOps

employees = (emp_id = [1, 2, 3], name = ["Shler", "Rivka", "Dilan"])
salaries = (employee_key = [1, 2, 4], salary = [50000, 60000, 70000])

left_join(employees, salaries, by=:emp_id => :employee_key)
# Output: (emp_id = [1, 2, 3], name = ["Shler", "Rivka", "Dilan"],
#          salary = [50000, 60000, missing])
```

#### Handling Duplicate Column Names

```julia
using Durbyn.TableOps

df1 = (id = [1, 2], value = [10, 20])
df2 = (id = [1, 2], value = [100, 200])

inner_join(df1, df2, by=:id)
# Output: (id = [1, 2], value_x = [10, 20], value_y = [100, 200])
# Non-key duplicate columns get suffixes

# Custom suffixes
inner_join(df1, df2, by=:id, suffix=("_left", "_right"))
# Output: (id = [1, 2], value_left = [10, 20], value_right = [100, 200])
```

#### One-to-Many Joins

```julia
using Durbyn.TableOps

customers = (id = [1, 2], name = ["Shler", "Rivka"])
orders = (customer_id = [1, 1, 2], order_id = [101, 102, 103], amount = [50, 75, 100])

left_join(customers, orders, by=:id => :customer_id)
# Output: (id = [1, 1, 2], name = ["Shler", "Shler", "Rivka"],
#          order_id = [101, 102, 103], amount = [50, 75, 100])
# Shler appears twice (has 2 orders)
```

---

## String Functions

### `separate` - Split Column into Multiple

Separate a character column into multiple columns by splitting on a delimiter.

```julia
using Durbyn.TableOps

# Basic separation
tbl = (id = [1, 2, 3], name = ["Peshraw-Cohen", "Narin-Levi", "Hawreh-Katz"])

separate(tbl, :name; into=[:first, :last], sep="-")
# Output: (id = [1, 2, 3],
#          first = ["Peshraw", "Narin", "Hawreh"],
#          last = ["Cohen", "Levi", "Katz"])

# Keep original column
separate(tbl, :name; into=[:first, :last], sep="-", remove=false)
# Output: (id = [1, 2, 3],
#          name = ["Peshraw-Cohen", "Narin-Levi", "Hawreh-Katz"],
#          first = ["Peshraw", "Narin", "Hawreh"],
#          last = ["Cohen", "Levi", "Katz"])

# With numeric conversion
tbl2 = (id = [1, 2], coords = ["10,20", "30,40"])
separate(tbl2, :coords; into=[:x, :y], sep=",", convert=true)
# Output: (id = [1, 2], x = [10.0, 30.0], y = [20.0, 40.0])

# Using regex separator
tbl3 = (data = ["a1b", "c2d", "e3f"],)
separate(tbl3, :data; into=[:letter1, :num, :letter2], sep=r"[0-9]")

# Handling uneven splits (extra parts are dropped, missing parts become missing)
tbl4 = (text = ["a-b-c", "x-y"],)
separate(tbl4, :text; into=[:p1, :p2, :p3], sep="-")
# Output: (p1 = ["a", "x"], p2 = ["b", "y"], p3 = ["c", missing])
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `col` - Column name to separate
- `into` (keyword) - Vector of names for the new columns
- `sep` (keyword, default: `" "`) - Separator pattern (`String`, `Char`, or `Regex`)
- `remove` (keyword, default: true) - Remove the input column
- `convert` (keyword, default: false) - Attempt to convert to numeric types. Tries `Int` first, falls back to `Float64` if needed.

---

### `unite` - Combine Columns into One

Combine multiple columns into a single character column.

```julia
using Durbyn.TableOps

tbl = (id = [1, 2, 3],
       year = [2020, 2021, 2022],
       month = [1, 6, 12])

# Basic unite
unite(tbl, :date, :year, :month; sep="-")
# Output: (id = [1, 2, 3], date = ["2020-1", "2021-6", "2022-12"])

# Keep original columns
unite(tbl, :date, :year, :month; sep="-", remove=false)
# Output: (id = [1, 2, 3],
#          year = [2020, 2021, 2022],
#          month = [1, 6, 12],
#          date = ["2020-1", "2021-6", "2022-12"])

# Custom separator
unite(tbl, :period, :year, :month; sep="/")
# Output: (id = [1, 2, 3], period = ["2020/1", "2021/6", "2022/12"])

# Multiple columns
tbl2 = (a = ["x", "y"], b = [1, 2], c = ["!", "?"])
unite(tbl2, :combined, :a, :b, :c; sep="")
# Output: (combined = ["x1!", "y2?"],)
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `new_col` - Name for the new combined column
- `cols...` - Columns to combine (at least one required)
- `sep` (keyword, default: `"_"`) - Separator between values
- `remove` (keyword, default: true) - Remove the input columns

**Note:** If any value is `missing`, the combined result is `missing`.

---

## Missing Value Functions

### `fill_missing` - Fill Missing Values

Fill missing values using the previous or next non-missing value (forward/backward fill).

```julia
using Durbyn.TableOps

tbl = (id = [1, 2, 3, 4, 5],
       value = [10, missing, missing, 40, missing])

# Fill down (forward fill) - default
fill_missing(tbl, :value)
# Output: (id = [1, 2, 3, 4, 5], value = [10, 10, 10, 40, 40])

# Fill up (backward fill)
fill_missing(tbl, :value; direction=:up)
# Output: (id = [1, 2, 3, 4, 5], value = [10, 40, 40, 40, missing])

# Fill both directions (down first, then up)
fill_missing(tbl, :value; direction=:downup)
# Output: (id = [1, 2, 3, 4, 5], value = [10, 10, 10, 40, 40])

# Fill both directions (up first, then down)
fill_missing(tbl, :value; direction=:updown)
# Output: (id = [1, 2, 3, 4, 5], value = [10, 40, 40, 40, 40])

# Fill multiple columns
tbl2 = (a = [1, missing, 3], b = [missing, 2, missing])
fill_missing(tbl2, :a, :b)
# Output: (a = [1, 1, 3], b = [missing, 2, 2])

# Fill all columns (no columns specified)
fill_missing(tbl2)
# Output: (a = [1, 1, 3], b = [missing, 2, 2])
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `cols...` - Columns to fill (if empty, fills all columns)
- `direction` (keyword, default: `:down`) - Fill direction:
  - `:down` - Forward fill (last observation carried forward)
  - `:up` - Backward fill (next observation carried backward)
  - `:downup` - Forward fill, then backward fill
  - `:updown` - Backward fill, then forward fill

---

### `complete` - Complete Missing Combinations

Expand a table to include all combinations of specified columns, filling new rows with a default value.

```julia
using Durbyn.TableOps

tbl = (year = [2020, 2020, 2021],
       quarter = [1, 2, 1],
       value = [100, 200, 150])

# Complete all year-quarter combinations
complete(tbl, :year, :quarter)
# Output: (year = [2020, 2020, 2021, 2021],
#          quarter = [1, 2, 1, 2],
#          value = Union{Missing, Int64}[100, 200, 150, missing])

# With custom fill value
complete(tbl, :year, :quarter; fill_value=0)
# Output: (year = [2020, 2020, 2021, 2021],
#          quarter = [1, 2, 1, 2],
#          value = [100, 200, 150, 0])

# Example with product-region combinations
sales = (product = ["A", "A", "B"],
         region = ["North", "South", "North"],
         sales = [100, 150, 120])
complete(sales, :product, :region; fill_value=0)
# Output: (product = ["A", "A", "B", "B"],
#          region = ["North", "South", "North", "South"],
#          sales = [100, 150, 120, 0])
# Adds missing B-South combination with sales=0
```

**Parameters:**
- `data` - Any Tables.jl-compatible data source
- `cols...` - Columns to expand (creates all unique combinations)
- `fill_value` (keyword, default: `missing`) - Value for new rows

**Note:** Original rows are preserved; only missing combinations are added.

---

## Complete Workflow Examples

### Example 1: Basic Data Analysis Pipeline

```julia
using Durbyn.TableOps
using Statistics

# Sample employee data
employees = (
    department = ["Sales", "IT", "Sales", "IT", "Sales", "HR", "HR"],
    employee = ["Shler", "Rivka", "Dilan", "Moshe", "Jwan", "Avraham", "Miriam"],
    salary = [60000, 70000, 55000, 75000, 65000, 50000, 52000],
    years = [5, 8, 3, 10, 6, 2, 4]
)

# Step 1: Preview data
glimpse(employees)

# Step 2: Filter high earners
filtered = query(employees, row -> row.salary > 55000)

# Step 3: Group by department
grouped = groupby(filtered, :department)

# Step 4: Compute statistics
summary = summarise(grouped,
    avg_salary = :salary => mean,
    avg_years = :years => mean,
    headcount = data -> length(data.salary))

# Step 5: Sort by average salary
result = arrange(summary, :avg_salary => :desc)

glimpse(result)
```

### Example 2: Time Series Panel Data

```julia
using CSV
using Downloads
using Tables
using Durbyn.TableOps
using Statistics

# Download retail data
local_path = Downloads.download("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/refs/heads/main/Data/retail.csv")
retail = CSV.File(local_path)
tbl = Tables.columntable(retail)

# Step 1: Transform from wide to long format
tbl_long = pivot_longer(tbl, id_cols=:date, names_to=:series, values_to=:value)
glimpse(tbl_long)

# Step 2: Filter to specific series
tbl_filtered = query(tbl_long, row -> row.series in ["series_10", "series_20", "series_30"])

# Step 3: Add computed columns
tbl_with_log = mutate(tbl_filtered, log_value = data -> log.(data.value))

# Step 4: Group by series
gt = groupby(tbl_with_log, :series)

# Step 5: Compute summary statistics
summary = summarise(gt,
    mean_value = :value => mean,
    std_value = :value => std,
    min_value = :value => minimum,
    max_value = :value => maximum,
    count = data -> length(data.value))

# Step 6: Sort by mean value
result = arrange(summary, :mean_value => :desc)
glimpse(result)
```

### Example 3: Data Cleaning with Missing Values

```julia
using Durbyn.TableOps

# Messy data with missing values and inconsistent formatting
raw_data = (
    date = ["2024-01", "2024-02", "2024-03", "2024-04"],
    region_product = ["North-A", "North-B", "South-A", "South-B"],
    value = [100, missing, 150, missing]
)

# Step 1: Separate region and product
cleaned = separate(raw_data, :region_product; into=[:region, :product], sep="-")

# Step 2: Fill missing values (forward fill)
filled = fill_missing(cleaned, :value)

# Step 3: Complete all region-product combinations
completed = complete(filled, :region, :product; fill_value=0)

glimpse(completed)
```

### Example 4: Using `across` for Multi-Column Operations

```julia
using Durbyn.TableOps
using Statistics

# Sales data with multiple metrics
sales = (
    region = ["North", "North", "South", "South", "East", "East"],
    product = ["A", "B", "A", "B", "A", "B"],
    revenue = [1000.0, 1500.0, 2000.0, 2500.0, 1800.0, 2200.0],
    units = [100.0, 150.0, 200.0, 250.0, 180.0, 220.0],
    returns = [5.0, 8.0, 10.0, 12.0, 9.0, 11.0]
)

# Compute mean and sum for all numeric columns per region
gt = groupby(sales, :region)

# Apply multiple functions across multiple columns
result = summarise(gt, across([:revenue, :units, :returns], :mean => mean, :total => sum))

glimpse(result)
# Output columns: region, revenue_mean, revenue_total, units_mean, units_total, returns_mean, returns_total
```

---

## Working with GroupedTable

The `GroupedTable` type is a central concept in TableOps, similar to grouped data frames in other languages.

```julia
using Durbyn.TableOps
using Statistics

sales_data = (
    region = ["North", "South", "North", "South", "East", "East", "West"],
    product = ["A", "A", "B", "B", "A", "B", "A"],
    revenue = [1000, 1500, 2000, 2500, 1800, 2200, 1200],
    units = [100, 150, 200, 250, 180, 220, 120]
)

# Group by multiple columns
gt = groupby(sales_data, :region, :product)
glimpse(gt)

# Compute complex summaries
summary = summarise(gt,
    total_revenue = :revenue => sum,
    total_units = :units => sum,
    avg_price = data -> sum(data.revenue) / sum(data.units),
    count = data -> length(data.revenue))

glimpse(summary)

# Ungroup to get back to regular table
ungrouped = ungroup(gt)
```

---

## PanelData Operations

TableOps provides special dispatches for `PanelData` objects that automatically apply operations **within each group**. This is essential for time series forecasting where you need to process multiple series independently while preserving the panel structure.

### How PanelData Operations Work

When you call a TableOps function on a `PanelData` object:
1. The data is automatically grouped by the panel's grouping columns
2. The operation is applied to each group independently
3. Results are combined back into a new `PanelData` with the same metadata

This differs from regular table operations where the function operates on the entire dataset at once.

### Creating PanelData

```julia
using Durbyn.ModelSpecs

# Basic panel data with grouping and date
panel = PanelData(data;
    groupby = :series,           # Column(s) identifying each series
    date = :date,                # Time index column
    m = 12                       # Seasonal period (12 = monthly with yearly cycle)
)

# With frequency (m is inferred automatically)
panel = PanelData(data;
    groupby = [:store, :product],  # Multiple grouping columns
    date = :date,
    frequency = :monthly           # :daily, :weekly, :monthly, :quarterly, :yearly
)

# With target column for forecasting
panel = PanelData(data;
    groupby = :series,
    date = :date,
    frequency = :monthly,
    target = :sales               # The variable to forecast
)

# With preprocessing: fill time gaps and impute missing values
panel = PanelData(data;
    groupby = :series,
    date = :date,
    frequency = :monthly,
    target = :sales,
    fill_time = true,             # Fill missing time points
    target_na = (method = :interpolate,)  # Impute missing target values
)
```

### PanelData Fields

| Field | Type | Description |
|-------|------|-------------|
| `data` | Any | The underlying table data |
| `groups` | `Vector{Symbol}` | Columns that identify each series |
| `date` | `Symbol` or `nothing` | Time index column |
| `m` | `Int`, `Vector{Int}`, or `nothing` | Seasonal period(s) |
| `frequency` | `Symbol` or `nothing` | Time frequency |
| `target` | `Symbol` or `nothing` | Target variable for forecasting |

### Supported Operations Reference

| Function | PanelData Behavior | Returns |
|----------|-------------------|---------|
| `query` | Filter rows within each group | `PanelData` |
| `mutate` | Add/modify columns within each group | `PanelData` |
| `arrange` | Sort rows within each group | `PanelData` |
| `select` | Select columns (grouping/date columns auto-included) | `PanelData` |
| `distinct` | Remove duplicates within each group (grouping columns auto-included) | `PanelData` |
| `fill_missing` | Fill missing values within each group | `PanelData` |
| `rename` | Rename columns (updates group/date metadata) | `PanelData` |
| `pivot_longer` | Pivot to long (grouping/date columns auto-added to id_cols) | `PanelData` |
| `pivot_wider` | Pivot to wide (grouping/date columns auto-added to id_cols) | `PanelData` |
| `summarise` | Summarize each group (collapses time dimension) | `NamedTuple` |

---

### `query` - Filter Rows Per Series

Filter rows independently within each series based on a predicate.

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

data = (series = ["A", "A", "A", "A", "B", "B", "B", "B"],
        date = [1, 2, 3, 4, 1, 2, 3, 4],
        value = [10, 25, 15, 30, 100, 150, 120, 180])

panel = PanelData(data; groupby=:series, date=:date)

# Keep only rows where value > 20 (applied per series)
filtered = query(panel, row -> row.value > 20)
glimpse(filtered)
# Series A: keeps dates 2, 4 (values 25, 30)
# Series B: keeps all dates (all values > 20)
```

**Use Case**: Remove outliers, filter to specific time periods, or apply series-specific conditions.

---

### `mutate` - Group-Relative Feature Engineering

Create new columns with computations that operate within each series. This is essential for creating features like:
- Series-level statistics (mean, std, min, max)
- Deviations from series mean (centering/normalization)
- Lagged values and differences
- Rolling statistics
- Percentage of series total

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs
using Statistics

data = (series = ["A", "A", "A", "A", "B", "B", "B", "B"],
        date = [1, 2, 3, 4, 1, 2, 3, 4],
        value = [100, 110, 90, 120, 500, 520, 480, 540])

panel = PanelData(data; groupby=:series, date=:date, m=12)

# Feature engineering within each series
result = mutate(panel,
    # Series statistics (broadcast to all rows)
    series_mean = d -> fill(mean(d.value), length(d.value)),
    series_std = d -> fill(std(d.value), length(d.value)),

    # Centering and scaling
    centered = d -> d.value .- mean(d.value),
    scaled = d -> (d.value .- mean(d.value)) ./ std(d.value),

    # Percentage of series total
    pct_of_total = d -> d.value ./ sum(d.value) .* 100,

    # Lagged values
    lag1 = d -> [missing; d.value[1:end-1]],

    # Differences
    diff1 = d -> [missing; diff(d.value)],

    # Percent change
    pct_change = d -> [missing; diff(d.value) ./ d.value[1:end-1] .* 100]
)

glimpse(result)
```

**Key Point**: The function receives only the current group's data, so `mean(d.value)` computes the mean for that series only, not the global mean.

---

### `arrange` - Sort Within Each Series

Sort rows by one or more columns within each series independently.

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

# Data with dates out of order
data = (series = ["A", "A", "A", "B", "B", "B"],
        date = [3, 1, 2, 2, 3, 1],
        value = [30, 10, 20, 200, 300, 100])

panel = PanelData(data; groupby=:series)

# Sort by date within each series
sorted = arrange(panel, :date)
# Series A: dates [1, 2, 3], values [10, 20, 30]
# Series B: dates [1, 2, 3], values [100, 200, 300]

# Sort descending
sorted_desc = arrange(panel, :date => :desc)

# Multi-column sort
arrange(panel, :category, :date => :desc)
```

**Use Case**: Ensure time ordering after joins or other operations that may scramble row order.

---

### `select` - Select Columns (Preserving Structure)

Select columns from the panel. **Grouping columns and date column are automatically included** to preserve the panel structure.

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

data = (series = ["A", "A", "B", "B"],
        date = [1, 2, 1, 2],
        value = [100, 110, 200, 210],
        extra = [1, 2, 3, 4],
        temp = [20, 21, 22, 23])

panel = PanelData(data; groupby=:series, date=:date)

# Select value column - series and date are auto-included
result = select(panel, :value)
# Result has columns: series, date, value

# Rename while selecting
result = select(panel, :sales => :value)
# Result has columns: series, date, sales

# Attempting to exclude grouping columns throws an error
# select(panel, :value)  # :series is always included
```

**Structural Protection**: The panel's grouping and date columns cannot be accidentally dropped, ensuring the panel structure remains valid.

---

### `distinct` - Unique Rows Per Series

Remove duplicate rows within each series. **Grouping columns are automatically included** in the uniqueness check.

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

data = (series = ["A", "A", "A", "B", "B", "B"],
        category = ["X", "X", "Y", "X", "X", "Y"],
        value = [100, 100, 200, 300, 300, 400])

panel = PanelData(data; groupby=:series)

# Distinct by category within each series
# (series is auto-included in uniqueness check)
result = distinct(panel, :category)
# Series A: keeps one "X" and one "Y"
# Series B: keeps one "X" and one "Y"

# Keep all columns while deduping
result = distinct(panel, :category; keep_all=true)
```

---

### `fill_missing` - Forward/Backward Fill Per Series

Fill missing values using previous or next values **within each series**. This prevents values from one series "leaking" into another.

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

data = (series = ["A", "A", "A", "A", "B", "B", "B", "B"],
        date = [1, 2, 3, 4, 1, 2, 3, 4],
        value = [10, missing, missing, 40, missing, 200, missing, 400])

panel = PanelData(data; groupby=:series, date=:date)

# Forward fill within each series
filled_down = fill_missing(panel, :value; direction=:down)
# Series A: [10, 10, 10, 40]
# Series B: [missing, 200, 200, 400]  # First value stays missing

# Backward fill
filled_up = fill_missing(panel, :value; direction=:up)
# Series A: [10, 40, 40, 40]
# Series B: [200, 200, 400, 400]

# Forward then backward (fills all if possible)
filled_both = fill_missing(panel, :value; direction=:downup)
# Series A: [10, 10, 10, 40]
# Series B: [200, 200, 200, 400]
```

**Critical for Time Series**: Without PanelData grouping, forward fill would carry values across series boundaries, corrupting your data.

---

### `rename` - Rename Columns (Updating Metadata)

Rename columns in the panel. If you rename a grouping column or date column, the panel metadata is automatically updated.

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

data = (id = ["A", "A", "B", "B"],
        time = [1, 2, 1, 2],
        val = [100, 110, 200, 210])

panel = PanelData(data; groupby=:id, date=:time)

# Rename grouping column
renamed = rename(panel, :series => :id)
# panel.groups is now [:series]

# Rename date column
renamed = rename(panel, :date => :time)
# panel.date is now :date

# Rename regular column
renamed = rename(panel, :value => :val)
```

---

### `pivot_longer` - Wide to Long (Preserving Structure)

Pivot from wide to long format. **Grouping and date columns are automatically added to `id_cols`** to preserve the panel structure.

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

# Wide format: multiple value columns
wide_data = (series = ["A", "A", "B", "B"],
             date = [1, 2, 1, 2],
             sales = [100, 110, 200, 210],
             costs = [80, 85, 150, 160])

panel = PanelData(wide_data; groupby=:series, date=:date)

# Pivot sales and costs to long format
# series and date are auto-included as id_cols
long = pivot_longer(panel,
    value_cols = [:sales, :costs],
    names_to = :metric,
    values_to = :amount)

glimpse(long)
# Columns: series, date, metric, amount
# Each original row becomes 2 rows (one for sales, one for costs)
```

---

### `pivot_wider` - Long to Wide (Preserving Structure)

Pivot from long to wide format. **Grouping and date columns are automatically added to `id_cols`**.

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs

# Long format
long_data = (series = ["A", "A", "A", "A", "B", "B", "B", "B"],
             date = [1, 1, 2, 2, 1, 1, 2, 2],
             metric = ["sales", "costs", "sales", "costs", "sales", "costs", "sales", "costs"],
             amount = [100, 80, 110, 85, 200, 150, 210, 160])

panel = PanelData(long_data; groupby=:series, date=:date)

# Pivot metric values to columns
wide = pivot_wider(panel,
    names_from = :metric,
    values_from = :amount)

glimpse(wide)
# Columns: series, date, sales, costs
```

---

### `summarise` - Aggregate Per Series

Compute summary statistics for each series. **Returns a `NamedTuple`** (not PanelData) since the time dimension is collapsed.

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs
using Statistics

data = (series = ["A", "A", "A", "A", "B", "B", "B", "B"],
        date = [1, 2, 3, 4, 1, 2, 3, 4],
        value = [100, 110, 90, 120, 500, 520, 480, 540])

panel = PanelData(data; groupby=:series, date=:date)

# Compute statistics per series
stats = summarise(panel,
    n = d -> length(d.value),
    mean_val = :value => mean,
    std_val = :value => std,
    min_val = :value => minimum,
    max_val = :value => maximum,
    range_val = d -> maximum(d.value) - minimum(d.value),
    cv = d -> std(d.value) / mean(d.value)  # Coefficient of variation
)

glimpse(stats)
# (series = ["A", "B"], n = [4, 4], mean_val = [105.0, 510.0], ...)
```

**Note**: `summarise` returns a `NamedTuple`, not `PanelData`, because the result has one row per series (time dimension is gone).

---

### Using `across` with PanelData

Apply the same function(s) across multiple columns within each group.

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs
using Statistics

data = (series = ["A", "A", "B", "B"],
        date = [1, 2, 1, 2],
        sales = [100.0, 110.0, 200.0, 210.0],
        costs = [80.0, 85.0, 150.0, 160.0],
        units = [10.0, 11.0, 20.0, 21.0])

panel = PanelData(data; groupby=:series, date=:date)

# Summarize multiple columns
stats = summarise(panel, across([:sales, :costs, :units], :mean => mean, :sum => sum))
# Result has: series, sales_mean, sales_sum, costs_mean, costs_sum, units_mean, units_sum
```

---

### Complete Forecasting Workflow Example

```julia
using Durbyn.TableOps
using Durbyn.ModelSpecs
using Statistics

# Raw retail data with multiple stores
raw_data = (
    store = ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2"],
    date = [1, 2, 3, 4, 1, 2, 3, 4],
    sales = [100.0, missing, 120.0, 130.0, 500.0, 520.0, missing, 560.0],
    promo = [0, 1, 0, 1, 1, 0, 1, 0]
)

# Step 1: Create PanelData
panel = PanelData(raw_data; groupby=:store, date=:date, m=12, target=:sales)

# Step 2: Handle missing values per series
cleaned = fill_missing(panel, :sales; direction=:downup)

# Step 3: Feature engineering within each series
features = mutate(cleaned,
    # Lag features
    sales_lag1 = d -> [missing; d.sales[1:end-1]],
    sales_lag2 = d -> [missing; missing; d.sales[1:end-2]],

    # Rolling mean (simple 2-period)
    sales_ma2 = d -> [missing; (d.sales[1:end-1] .+ d.sales[2:end]) ./ 2],

    # Series-level features
    series_mean = d -> fill(mean(d.sales), length(d.sales)),
    deviation = d -> d.sales .- mean(d.sales),

    # Trend proxy
    time_idx = d -> collect(1:length(d.sales))
)

# Step 4: Filter to rows with complete features
model_data = query(features, r -> !ismissing(r.sales_lag2))

# Step 5: Summarize for exploration
summary = summarise(panel,
    n = d -> length(d.sales),
    mean_sales = :sales => x -> mean(skipmissing(x)),
    total_sales = :sales => x -> sum(skipmissing(x)))

glimpse(summary)
```

---

### Key Benefits of PanelData Operations

1. **Automatic Grouping**: No need to manually `groupby` - the panel's group structure is used automatically

2. **Preserved Metadata**: Operations return a new `PanelData` with the same grouping columns, date column, seasonal period, and other metadata

3. **Structural Protection**: Grouping and date columns cannot be accidentally dropped by `select` or `distinct`

4. **Group-Relative Computations**: In `mutate`, functions receive only the current group's data, enabling proper within-series feature engineering

5. **Independent Processing**: Each group is processed independently, preventing data leakage between series (critical for `fill_missing`)

6. **Seamless Forecasting Integration**: The preserved metadata flows directly into Durbyn's forecasting models

---

## Tips and Best Practices

1. **Use `glimpse` frequently**: It's a quick way to understand your data's structure and verify transformations.

2. **Predicate functions in `query`**: Keep them simple and readable. For complex filters, break them into logical parts or define named functions.

3. **Type stability in `mutate`**: Ensure your computed columns have consistent types across all rows.

4. **Group before summarize**: Always create a `GroupedTable` with `groupby` before using `summarise`.

5. **Column naming**: Use descriptive names in `mutate` and `summarise` to make your data self-documenting.

6. **Pivot operations**:
   - Use `pivot_longer` when you need to reshape data for modeling or plotting
   - Use `pivot_wider` when you need to create summary tables or compare values across categories

7. **Memory efficiency**: TableOps functions return new `NamedTuple`s, so be mindful of memory when working with very large datasets.

8. **Chaining operations**: Use intermediate variables for readability:
   ```julia
   # Recommended: Clear and debuggable
   filtered = query(data, row -> row.x > 0)
   grouped = groupby(filtered, :category)
   result = summarise(grouped, mean_x = :x => mean)
   ```

9. **`fill_missing` direction**: Use `:downup` or `:updown` to ensure all missing values are filled when you have missing values at both ends.

10. **`complete` for time series**: Use with `fill_missing` to handle gaps in time series data:
    ```julia
    data |> x -> complete(x, :date) |> x -> fill_missing(x, :value)
    ```
