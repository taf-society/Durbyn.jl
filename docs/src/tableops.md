# Table Operations

The `TableOps` module provides a comprehensive set of data manipulation functions for working with Tables.jl-compatible data sources. These functions enable common data wrangling tasks like filtering, grouping, pivoting, and summarizing data.

## Overview

`TableOps` is inspired by popular data manipulation libraries like dplyr (R) and pandas (Python), but designed specifically for Julia's Tables.jl ecosystem. All functions work seamlessly with any Tables.jl-compatible data source, including:

- NamedTuples
- DataFrames
- CSV.File objects
- And many others

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
#   date                :: String  [2024-01, 2024-02, 2024-03]
#   A                   :: Int64   [100, 110, 120]
#   B                   :: Int64   [200, 220, 240]
#   C                   :: Int64   [300, 330, 360]
```

### `select` - Choose and Rename Columns

Select specific columns from your data, optionally renaming them.

```julia
using Durbyn.TableOps

tbl = (id = [1, 2, 3],
       name = ["Alice", "Bob", "Charlie"],
       age = [25, 30, 35],
       salary = [50000, 60000, 70000])

# Select specific columns
select(tbl, :name, :age)
# Output: (name = ["Alice", "Bob", "Charlie"], age = [25, 30, 35])

# Rename while selecting
select(tbl, :employee => :name, :years => :age)
# Output: (employee = ["Alice", "Bob", "Charlie"], years = [25, 30, 35])
```

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
```

### `arrange` - Sort Data

Sort rows by one or more columns in ascending or descending order.

```julia
using Durbyn.TableOps

tbl = (name = ["Alice", "Bob", "Charlie", "David"],
       department = ["Sales", "IT", "Sales", "IT"],
       salary = [60000, 70000, 55000, 75000])

# Sort by salary (ascending)
arrange(tbl, :salary)
# Output: (name = ["Charlie", "Alice", "Bob", "David"],
#          department = ["Sales", "Sales", "IT", "IT"],
#          salary = [55000, 60000, 70000, 75000])

# Sort by salary (descending)
arrange(tbl, :salary => :desc)
# Output: (name = ["David", "Bob", "Alice", "Charlie"],
#          department = ["IT", "IT", "Sales", "Sales"],
#          salary = [75000, 70000, 60000, 55000])

# Multi-column sort: department ascending, then salary descending
arrange(tbl, :department, :salary => :desc)
# Output: (name = ["David", "Bob", "Alice", "Charlie"],
#          department = ["IT", "IT", "Sales", "Sales"],
#          salary = [75000, 70000, 60000, 55000])
```

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
```

### `groupby` - Group Data

Group rows by unique combinations of values in specified columns.

```julia
using Durbyn.TableOps

tbl = (department = ["Sales", "IT", "Sales", "IT", "Sales"],
       employee = ["Alice", "Bob", "Charlie", "David", "Eve"],
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
```

### `summarise` / `summarize` - Aggregate Data

Compute summary statistics for each group in a GroupedTable.

```julia
using Durbyn.TableOps
using Statistics

tbl = (department = ["Sales", "IT", "Sales", "IT", "Sales"],
       employee = ["Alice", "Bob", "Charlie", "David", "Eve"],
       salary = [60000, 70000, 55000, 75000, 65000])

gt = groupby(tbl, :department)

# Compute mean salary per department
stbl = summarise(gt, avg_salary = :salary => mean)
# Output: (department = ["IT", "Sales"], avg_salary = [72500.0, 60000.0])
glimpse(stbl)

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
```

### `pivot_longer` - Wide to Long Format

Transform data from wide format to long format by pivoting columns into rows.

```julia
using CSV
using Downloads
using Tables
using Durbyn
using Durbyn.TableOps

# Download and load retail data
local_path = Downloads.download("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/refs/heads/main/Data/retail.csv")
retail = CSV.File(local_path)
tbl = Tables.columntable(retail)

# Preview wide format
glimpse(tbl)

# Convert from wide to long format
tbl_long = pivot_longer(tbl, id_cols=:date, names_to=:series, values_to=:value)
glimpse(tbl_long)

# Example with simpler data
wide = (date = ["2024-01", "2024-02", "2024-03"],
        A = [100, 110, 120],
        B = [200, 220, 240],
        C = [300, 330, 360])

long = pivot_longer(wide, id_cols=:date, names_to=:series, values_to=:value)
# Output: (date = ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02", "2024-03", "2024-03", "2024-03"],
#          series = ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
#          value = [100, 200, 300, 110, 220, 330, 120, 240, 360])
glimpse(long)
glimpse(wide)
```

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

glimpse(long)
glimpse(wide)

# Sort column names alphabetically
pivot_wider(long, names_from=:series, values_from=:value,
            id_cols=:date, sort_names=true)

# Handle missing values with custom fill
incomplete = (id = [1, 1, 2], category = ["A", "B", "A"], val = [10, 20, 30])
pivot_wider(incomplete, names_from=:category, values_from=:val, fill=0)
# Output: (id = [1, 2], A = [10, 30], B = [20, 0])
```

## Complete Workflow Example

Here's a complete example demonstrating how to chain multiple operations together for a typical data analysis workflow:

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

# Step 1: Preview the data
glimpse(tbl)

# Step 2: Transform from wide to long format
tbl_long = pivot_longer(tbl, id_cols=:date, names_to=:series, values_to=:value)
glimpse(tbl_long)

# Step 3: Filter to specific series
tbl_filtered = query(tbl_long, row -> row.series in ["series_10", "series_20", "series_30"])

# Step 4: Add computed columns
tbl_with_log = mutate(tbl_filtered, log_value = data -> log.(data.value))

# Step 5: Group by series
gt = groupby(tbl_with_log, :series)

# Step 6: Compute summary statistics
summary = summarise(gt,
    mean_value = :value => mean,
    std_value = :value => std,
    min_value = :value => minimum,
    max_value = :value => maximum,
    count = data -> length(data.value))

glimpse(summary)

# Step 7: Sort by mean value
result = arrange(summary, :mean_value => :desc)
glimpse(result)
```

## Chaining Operations

While Julia doesn't have a built-in pipe operator for data manipulation (like R's `%>%` or `|>`), you can chain operations by nesting function calls or using intermediate variables:

```julia
using Durbyn.TableOps
using Statistics

tbl = (department = ["Sales", "IT", "Sales", "IT", "Sales", "HR", "HR"],
       employee = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"],
       salary = [60000, 70000, 55000, 75000, 65000, 50000, 52000],
       years = [5, 8, 3, 10, 6, 2, 4])

# Method 1: Nested functions
result = arrange(
    summarise(
        groupby(
            query(tbl, row -> row.salary > 52000),
            :department),
        avg_salary = :salary => mean,
        avg_years = :years => mean,
        count = data -> length(data.salary)),
    :avg_salary => :desc)

glimpse(result)

# Method 2: Step by step with intermediate variables (recommended for readability)
filtered = query(tbl, row -> row.salary > 52000)
grouped = groupby(filtered, :department)
summarized = summarise(grouped,
    avg_salary = :salary => mean,
    avg_years = :years => mean,
    count = data -> length(data.salary))
result = arrange(summarized, :avg_salary => :desc)

glimpse(result)
```

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
```

## Tips and Best Practices

1. **Use `glimpse` frequently**: It's a quick way to understand your data's structure and verify transformations.

2. **Predicate functions in `query`**: Keep them simple and readable. For complex filters, break them into logical parts.

3. **Type stability in `mutate`**: Ensure your computed columns have consistent types across all rows.

4. **Group before summarize**: Always create a `GroupedTable` with `groupby` before using `summarise`.

5. **Column naming**: Use descriptive names in `mutate` and `summarise` to make your data self-documenting.

6. **Pivot operations**:
   - Use `pivot_longer` when you need to reshape data for modeling or plotting
   - Use `pivot_wider` when you need to create summary tables or compare values across categories

7. **Memory efficiency**: TableOps functions return new `NamedTuple`s, so be mindful of memory when working with very large datasets.

## API Reference

For detailed API documentation of each function, see:

- [`select`](@ref)
- [`query`](@ref)
- [`arrange`](@ref)
- [`groupby`](@ref)
- [`mutate`](@ref)
- [`summarise`](@ref) / [`summarize`](@ref)
- [`pivot_longer`](@ref)
- [`pivot_wider`](@ref)
- [`glimpse`](@ref)
