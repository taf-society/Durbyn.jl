using Dates

# =============================================================================
# Frequency handling
# =============================================================================

"""
Valid frequency symbols for time series data.
"""
const VALID_FREQUENCIES = Set([
    :second, :minute, :halfhour, :hour, :hourly,  # Sub-daily
    :daily, :day, :businessday, :weekly, :week,   # Daily+
    :biweekly, :monthly, :month, :quarterly,      # Longer periods
    :quarter, :yearly, :year
])

"""
    frequency_to_m(frequency::Symbol) -> Int

Convert a named frequency to the default seasonal period (m).
Returns the most common seasonality for each frequency.

# Examples
- `:daily` → 7 (weekly seasonality)
- `:hourly` → 24 (daily seasonality)
- `:monthly` → 12 (yearly seasonality)
"""
function frequency_to_m(frequency::Symbol)
    freq_map = Dict(
        # Sub-daily
        :second => 60,      # minute seasonality
        :minute => 60,      # hour seasonality
        :halfhour => 48,    # day seasonality (48 half-hours per day)
        :hour => 24,        # day seasonality
        :hourly => 24,      # alias
        # Daily
        :day => 7,          # week seasonality
        :daily => 7,        # alias
        :businessday => 5,  # week seasonality (5 business days)
        # Weekly+
        :week => 52,        # year seasonality
        :weekly => 52,      # alias
        :biweekly => 26,    # year seasonality
        # Monthly+
        :month => 12,       # year seasonality
        :monthly => 12,     # alias
        :quarter => 4,      # year seasonality
        :quarterly => 4,    # alias
        :year => 1,         # no sub-yearly seasonality
        :yearly => 1,       # alias
    )

    haskey(freq_map, frequency) || throw(ArgumentError(
        "Unknown frequency :$frequency. Valid frequencies: $(join(sort(collect(VALID_FREQUENCIES)), ", "))"))

    return freq_map[frequency]
end

"""
    frequency_to_period(frequency::Symbol) -> Period

Convert a named frequency to a Julia Dates Period for time grid completion.
"""
function frequency_to_period(frequency::Symbol)
    period_map = Dict(
        :second => Second(1),
        :minute => Minute(1),
        :halfhour => Minute(30),
        :hour => Hour(1),
        :hourly => Hour(1),
        :day => Day(1),
        :daily => Day(1),
        :businessday => Day(1),  # Handled specially in fill_time
        :week => Week(1),
        :weekly => Week(1),
        :biweekly => Week(2),
        :month => Month(1),
        :monthly => Month(1),
        :quarter => Month(3),
        :quarterly => Month(3),
        :year => Year(1),
        :yearly => Year(1),
    )

    haskey(period_map, frequency) || throw(ArgumentError(
        "Unknown frequency :$frequency"))

    return period_map[frequency]
end

"""
    format_frequency(frequency::Symbol) -> String

Return a human-readable description of a frequency.
"""
function format_frequency(frequency::Symbol)
    desc_map = Dict(
        :second => "second",
        :minute => "minute",
        :halfhour => "half-hour (30 min)",
        :hour => "hour",
        :hourly => "hour",
        :day => "day",
        :daily => "day",
        :businessday => "business day (weekdays)",
        :week => "week",
        :weekly => "week",
        :biweekly => "bi-weekly (14 days)",
        :month => "month",
        :monthly => "month",
        :quarter => "quarter",
        :quarterly => "quarter",
        :year => "year",
        :yearly => "year",
    )
    return get(desc_map, frequency, string(frequency))
end

"""
    m_description(m::Int, frequency::Symbol) -> String

Return a description of what the seasonal period means.
"""
function m_description(m::Int, frequency::Union{Symbol, Nothing})
    if frequency === nothing
        return ""
    end

    desc_map = Dict(
        (:daily, 7) => "weekly",
        (:day, 7) => "weekly",
        (:daily, 365) => "yearly",
        (:day, 365) => "yearly",
        (:hourly, 24) => "daily",
        (:hour, 24) => "daily",
        (:hourly, 168) => "weekly",
        (:hour, 168) => "weekly",
        (:monthly, 12) => "yearly",
        (:month, 12) => "yearly",
        (:weekly, 52) => "yearly",
        (:week, 52) => "yearly",
        (:quarterly, 4) => "yearly",
        (:quarter, 4) => "yearly",
    )

    return get(desc_map, (frequency, m), "")
end

# =============================================================================
# Multi-seasonality support
# =============================================================================

"""
    supports_multi_seasonality(spec) -> Bool

Trait function indicating whether a model specification supports multiple seasonal periods.
Override this for model types that support multi-seasonality (e.g., TBATS, MSTL).
"""
supports_multi_seasonality(::Any) = false

"""
    resolve_m(m, spec) -> Int or Vector{Int} or Nothing

Resolve the seasonal period(s) based on model capabilities.
For models that don't support multi-seasonality, extracts the first element
from a vector and logs an informative message.

# Arguments
- `m`: Seasonal period - can be `nothing`, `Int`, or `Vector{Int}`
- `spec`: Model specification (used to check multi-seasonality support)

# Returns
- `nothing` if m is nothing
- `Int` for single seasonality or models that don't support multi-seasonality
- `Vector{Int}` for models that support multi-seasonality
"""
resolve_m(::Nothing, _) = nothing

resolve_m(m::Int, _) = m

function resolve_m(m::Vector{Int}, spec)
    if supports_multi_seasonality(spec)
        return m
    else
        model_name = nameof(typeof(spec))
        @info "$model_name supports single seasonality. Using m=$(m[1]). For multi-seasonality, use TBATS or MSTL."
        return m[1]
    end
end

# =============================================================================
# Imputation metadata
# =============================================================================

"""
    ImputationMeta

Metadata about gap-filling/imputation performed on a column.
"""
struct ImputationMeta
    strategy::Symbol
    n_imputed::Int
    n_total::Int
    params::NamedTuple
end

function Base.show(io::IO, meta::ImputationMeta)
    pct = round(100 * meta.n_imputed / meta.n_total, digits=1)
    print(io, "ImputationMeta(:$(meta.strategy), $(meta.n_imputed)/$(meta.n_total) = $(pct)% imputed)")
end

"""
    TimeFillMeta

Metadata about time grid completion.
"""
struct TimeFillMeta
    n_added::Int
    n_total::Int
    step::Period
end

function Base.show(io::IO, meta::TimeFillMeta)
    print(io, "TimeFillMeta($(meta.n_added) rows added, step=$(meta.step))")
end

# =============================================================================
# PanelData struct
# =============================================================================

"""
    PanelData(data; groupby=nothing, date=nothing, m=nothing, frequency=nothing,
              target=nothing, fill_time=false, target_na=nothing, xreg_na=nothing)

Container for panel/time series datasets with optional preprocessing pipeline.

The `PanelData` interface follows the **tidy forecasting workflow** from
Hyndman & Athanasopoulos (2021), providing a structured approach to time series
forecasting: data preparation → visualization → model specification → training →
evaluation → forecasting. See: https://otexts.com/fpp3/

# Core Arguments
- `data`: The underlying tabular data (any Tables.jl-compatible format)
- `groupby`: Column(s) for grouping (Symbol or Vector{Symbol})
- `date`: Date/time column name (Symbol)
- `m`: Seasonal period(s) - Int or Vector{Int} for multi-seasonality

# New Arguments (opt-in preprocessing)
- `frequency`: Named frequency (:daily, :hourly, :monthly, etc.)
  - If provided and `m` is nothing, derives `m` automatically
- `target`: Target column name for forecasting (Symbol)
- `fill_time`: If true, complete missing dates in time grid (default: false)
- `balanced`: If true with `fill_time`, all groups use the same global time span (balanced panel)
- `target_na`: Gap-filling config for target column
  - `(strategy=:locf,)` - Last observation carried forward
  - `(strategy=:nocb,)` - Next observation carried backward
  - `(strategy=:linear,)` - Linear interpolation
  - `(strategy=:zero,)` - Fill with zeros
- `xreg_na`: Dict of gap-filling configs for exogenous columns
  - `Dict(:price => (strategy=:locf,), :promo => (strategy=:zero,))`

# Examples

```julia
# Basic usage (backward compatible)
panel = PanelData(data; groupby=:store, date=:date, m=12)

# With frequency (auto-derives m)
panel = PanelData(data; groupby=:store, date=:date, frequency=:daily)
# Info: Seasonal period m=7 (weekly) inferred from frequency=:daily

# Full preprocessing pipeline
panel = PanelData(data;
    groupby = :store,
    date = :date,
    frequency = :daily,
    target = :sales,
    fill_time = true,
    target_na = (strategy=:locf, max_gap=7),
    xreg_na = Dict(:price => (strategy=:locf,), :promo => (strategy=:zero,))
)

# Balanced panel (all groups padded to same global time span)
panel = PanelData(data;
    groupby = :store,
    date = :date,
    frequency = :daily,
    fill_time = true,
    balanced = true
)
```
"""
struct PanelData
    data
    groups::Vector{Symbol}
    date::Union{Symbol, Nothing}
    m::Union{Int, Vector{Int}, Nothing}

    # New fields
    frequency::Union{Symbol, Nothing}
    target::Union{Symbol, Nothing}

    # Preprocessing metadata
    time_fill_meta::Union{TimeFillMeta, Nothing}
    target_meta::Union{ImputationMeta, Nothing}
    xreg_meta::Dict{Symbol, ImputationMeta}
end

# Helper to normalize groupby argument
_normalize_groups(::Nothing) = Symbol[]
_normalize_groups(col::Symbol) = Symbol[col]
function _normalize_groups(cols::AbstractVector)
    return Symbol[Symbol(c) for c in cols]
end

# Helper to normalize m argument
_normalize_m(::Nothing) = nothing
_normalize_m(m::Int) = m
_normalize_m(m::Vector{Int}) = m
_normalize_m(m::AbstractVector) = Int[Int(x) for x in m]

"""
    PanelData(data; kwargs...)

Construct a PanelData with optional preprocessing.
"""
function PanelData(data;
                   groupby::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing,
                   date::Union{Symbol, Nothing} = nothing,
                   m::Union{Nothing, Int, AbstractVector{<:Integer}} = nothing,
                   frequency::Union{Symbol, Nothing} = nothing,
                   target::Union{Symbol, Nothing} = nothing,
                   fill_time::Bool = false,
                   balanced::Bool = false,
                   target_na::Union{Nothing, NamedTuple} = nothing,
                   xreg_na::Union{Nothing, Dict} = nothing)

    groups = _normalize_groups(groupby)
    m_val = _normalize_m(m)

    # Validate frequency if provided
    if frequency !== nothing && !(frequency in VALID_FREQUENCIES)
        throw(ArgumentError(
            "Unknown frequency :$frequency. Valid frequencies: $(join(sort(collect(VALID_FREQUENCIES)), ", "))"))
    end

    # Derive m from frequency if not provided
    if m_val === nothing && frequency !== nothing
        m_val = frequency_to_m(frequency)
        desc = m_description(m_val, frequency)
        desc_str = isempty(desc) ? "" : " ($desc)"
        @info "Seasonal period m=$m_val$desc_str inferred from frequency=:$frequency"
    end

    # Initialize metadata
    time_fill_meta = nothing
    target_meta = nothing
    xreg_meta = Dict{Symbol, ImputationMeta}()

    # Get column table for preprocessing
    ct = Tables.columntable(data)
    processed_data = ct

    # Validate date column exists if specified
    if date !== nothing
        col_names = propertynames(ct)
        date in col_names || throw(ArgumentError("Date column :$date not found in data"))
    end

    # Validate target column exists if specified
    if target !== nothing
        col_names = propertynames(ct)
        target in col_names || throw(ArgumentError("Target column :$target not found in data"))
    end

    # === Preprocessing Pipeline ===

    # 1. Sort by groups and date
    if date !== nothing
        processed_data = _sort_panel_data(processed_data, groups, date)
    end

    # 2. Fill time gaps if requested
    if fill_time
        if date === nothing
            throw(ArgumentError("Cannot fill_time without a date column"))
        end
        if frequency === nothing
            throw(ArgumentError("Cannot fill_time without a frequency"))
        end

        processed_data, time_fill_meta = _fill_time_gaps(
            processed_data, groups, date, frequency, balanced)
    elseif balanced
        throw(ArgumentError("balanced=true requires fill_time=true"))
    end

    # 3. Fill target gaps if requested
    if target_na !== nothing
        if target === nothing
            throw(ArgumentError("Cannot use target_na without specifying target"))
        end

        processed_data, target_meta = _fill_column_gaps(
            processed_data, groups, target, target_na)
    end

    # 4. Fill exogenous column gaps if requested
    if xreg_na !== nothing
        col_names = propertynames(processed_data)
        for (col, config) in xreg_na
            col in col_names || throw(ArgumentError("Exogenous column :$col not found in data"))
            processed_data, col_meta = _fill_column_gaps(
                processed_data, groups, col, config)
            xreg_meta[col] = col_meta
        end
    end

    return PanelData(processed_data, groups, date, m_val, frequency, target,
                     time_fill_meta, target_meta, xreg_meta)
end

# =============================================================================
# Preprocessing helper functions
# =============================================================================

"""
Sort data by groups and date.
"""
function _sort_panel_data(ct::NamedTuple, groups::Vector{Symbol}, date::Symbol)
    n = length(ct[first(propertynames(ct))])

    # Build sort keys
    sort_cols = vcat(groups, [date])

    # Create sort permutation
    perm = sortperm(1:n, by = i -> Tuple(ct[col][i] for col in sort_cols))

    # Apply permutation to all columns
    col_names = propertynames(ct)
    new_cols = [ct[col][perm] for col in col_names]

    return NamedTuple{col_names}(Tuple(new_cols))
end

"""
Fill time gaps by completing the time grid per group.
If `balanced=true`, all groups are padded to the same global time span.
"""
function _fill_time_gaps(ct::NamedTuple, groups::Vector{Symbol}, date::Symbol,
                         frequency::Symbol, balanced::Bool=false)
    n_original = length(ct[date])
    period = frequency_to_period(frequency)
    col_names = propertynames(ct)

    if isempty(groups)
        # Single series - complete globally
        if balanced
            @warn "balanced=true has no effect without groupby columns"
        end
        dates = ct[date]
        min_date, max_date = extrema(dates)
        complete_dates = collect(min_date:period:max_date)

        # Build lookup for existing dates
        date_to_idx = Dict(d => i for (i, d) in enumerate(dates))

        # Create new data with complete dates
        n_complete = length(complete_dates)
        new_cols = Dict{Symbol, Vector}()

        new_cols[date] = complete_dates

        for col in col_names
            if col == date
                continue
            end

            old_vec = ct[col]
            T = Union{Missing, eltype(old_vec)}
            new_vec = Vector{T}(missing, n_complete)

            for (i, d) in enumerate(complete_dates)
                if haskey(date_to_idx, d)
                    new_vec[i] = old_vec[date_to_idx[d]]
                end
            end

            new_cols[col] = new_vec
        end

        # Reconstruct in original column order
        result_cols = [new_cols[col] for col in col_names]
        result = NamedTuple{col_names}(Tuple(result_cols))

        n_added = n_complete - n_original
        meta = TimeFillMeta(n_added, n_complete, period)

        if n_added > 0
            @info "Time grid completed: $n_added rows added (step: $period)"
        end

        return result, meta
    else
        # Grouped data - complete per group
        # Group the data
        group_indices = _group_indices(ct, groups)

        # Compute global bounds if balanced panel requested
        if balanced
            global_min, global_max = extrema(ct[date])
        end

        results = NamedTuple[]
        total_added = 0

        for idxs in group_indices
            # Extract subgroup
            subgroup = NamedTuple{col_names}(Tuple(ct[col][idxs] for col in col_names))

            dates = subgroup[date]
            if balanced
                min_date, max_date = global_min, global_max
            else
                min_date, max_date = extrema(dates)
            end
            complete_dates = collect(min_date:period:max_date)

            # Build lookup
            date_to_idx = Dict(d => i for (i, d) in enumerate(dates))

            n_complete = length(complete_dates)
            new_cols = Dict{Symbol, Vector}()

            new_cols[date] = complete_dates

            # Group columns - fill with constant value
            for g in groups
                new_cols[g] = fill(subgroup[g][1], n_complete)
            end

            # Other columns - missing for new rows
            for col in col_names
                if col == date || col in groups
                    continue
                end

                old_vec = subgroup[col]
                T = Union{Missing, eltype(old_vec)}
                new_vec = Vector{T}(missing, n_complete)

                for (i, d) in enumerate(complete_dates)
                    if haskey(date_to_idx, d)
                        new_vec[i] = old_vec[date_to_idx[d]]
                    end
                end

                new_cols[col] = new_vec
            end

            result_cols = [new_cols[col] for col in col_names]
            push!(results, NamedTuple{col_names}(Tuple(result_cols)))

            total_added += n_complete - length(idxs)
        end

        # Combine all groups
        combined = _vcat_namedtuples(results)
        n_total = length(combined[date])
        meta = TimeFillMeta(total_added, n_total, period)

        if total_added > 0
            balanced_str = balanced ? " (balanced panel)" : ""
            @info "Time grid completed: $total_added rows added (step: $period)$balanced_str"
        end

        return combined, meta
    end
end

"""
Fill missing values in a column using the specified strategy.
"""
function _fill_column_gaps(ct::NamedTuple, groups::Vector{Symbol}, col::Symbol,
                           config::NamedTuple)
    strategy = get(config, :strategy, :locf)
    max_gap = get(config, :max_gap, typemax(Int))

    col_names = propertynames(ct)
    old_vec = ct[col]
    n_total = length(old_vec)

    # Create imputed flag column
    imputed_col = Symbol(string(col) * "_is_imputed")

    if isempty(groups)
        # Single series
        new_vec, is_imputed = _apply_fill_strategy(old_vec, strategy, max_gap)

        # Build new data with imputed column
        new_cols = Vector{Any}(undef, length(col_names) + 1)
        new_col_names = (col_names..., imputed_col)

        for (i, c) in enumerate(col_names)
            if c == col
                new_cols[i] = new_vec
            else
                new_cols[i] = ct[c]
            end
        end
        new_cols[end] = is_imputed

        result = NamedTuple{new_col_names}(Tuple(new_cols))
    else
        # Grouped data
        group_indices = _group_indices(ct, groups)

        # Initialize result vectors
        result_vec = copy(Vector{Union{Missing, eltype(old_vec)}}(old_vec))
        is_imputed_vec = fill(false, n_total)

        for idxs in group_indices
            subvec = old_vec[idxs]
            filled, is_imp = _apply_fill_strategy(subvec, strategy, max_gap)
            result_vec[idxs] = filled
            is_imputed_vec[idxs] = is_imp
        end

        # Build new data with imputed column
        new_cols = Vector{Any}(undef, length(col_names) + 1)
        new_col_names = (col_names..., imputed_col)

        for (i, c) in enumerate(col_names)
            if c == col
                new_cols[i] = result_vec
            else
                new_cols[i] = ct[c]
            end
        end
        new_cols[end] = is_imputed_vec

        result = NamedTuple{new_col_names}(Tuple(new_cols))
    end

    n_imputed = count(result[imputed_col])
    meta = ImputationMeta(strategy, n_imputed, n_total,
                          (max_gap=max_gap,))

    if n_imputed > 0
        pct = round(100 * n_imputed / n_total, digits=1)
        @info "Column :$col gap-filled: $n_imputed values imputed ($pct%) using :$strategy"
    end

    return result, meta
end

"""
Apply a fill strategy to a vector.
Returns (filled_vector, is_imputed_vector).
"""
function _apply_fill_strategy(vec::AbstractVector, strategy::Symbol, max_gap::Int)
    n = length(vec)
    result = Vector{Union{Missing, nonmissingtype(eltype(vec))}}(vec)
    is_imputed = fill(false, n)

    if strategy == :locf || strategy == :down
        # Last observation carried forward
        last_valid = nothing
        gap_count = 0

        for i in 1:n
            if !ismissing(result[i])
                last_valid = result[i]
                gap_count = 0
            elseif last_valid !== nothing
                gap_count += 1
                if gap_count <= max_gap
                    result[i] = last_valid
                    is_imputed[i] = true
                end
            end
        end

    elseif strategy == :nocb || strategy == :up
        # Next observation carried backward
        next_valid = nothing
        gap_count = 0

        for i in n:-1:1
            if !ismissing(result[i])
                next_valid = result[i]
                gap_count = 0
            elseif next_valid !== nothing
                gap_count += 1
                if gap_count <= max_gap
                    result[i] = next_valid
                    is_imputed[i] = true
                end
            end
        end

    elseif strategy == :linear
        # Linear interpolation
        i = 1
        while i <= n
            if ismissing(result[i])
                # Find start of gap
                gap_start = i

                # Find end of gap
                gap_end = i
                while gap_end <= n && ismissing(result[gap_end])
                    gap_end += 1
                end
                gap_end -= 1

                gap_length = gap_end - gap_start + 1

                # Check bounds and max_gap
                if gap_start > 1 && gap_end < n && gap_length <= max_gap
                    v1 = result[gap_start - 1]
                    v2 = result[gap_end + 1]

                    if !ismissing(v1) && !ismissing(v2)
                        for j in gap_start:gap_end
                            t = (j - gap_start + 1) / (gap_length + 1)
                            result[j] = v1 + t * (v2 - v1)
                            is_imputed[j] = true
                        end
                    end
                end

                i = gap_end + 1
            else
                i += 1
            end
        end

    elseif strategy == :zero
        # Fill with zeros
        for i in 1:n
            if ismissing(result[i])
                result[i] = zero(nonmissingtype(eltype(vec)))
                is_imputed[i] = true
            end
        end

    else
        throw(ArgumentError("Unknown fill strategy :$strategy. " *
            "Valid strategies: :locf, :nocb, :linear, :zero"))
    end

    return result, is_imputed
end

"""
Get indices for each group.
"""
function _group_indices(ct::NamedTuple, groups::Vector{Symbol})
    n = length(ct[first(propertynames(ct))])

    # Build group keys
    group_keys = Dict{Any, Vector{Int}}()

    for i in 1:n
        key = Tuple(ct[g][i] for g in groups)
        if !haskey(group_keys, key)
            group_keys[key] = Int[]
        end
        push!(group_keys[key], i)
    end

    return collect(values(group_keys))
end

"""
Vertically concatenate NamedTuples.
"""
function _vcat_namedtuples(nts::Vector{<:NamedTuple})
    isempty(nts) && return NamedTuple()

    col_names = propertynames(nts[1])

    result_cols = Vector{Vector}(undef, length(col_names))
    for (i, col) in enumerate(col_names)
        result_cols[i] = vcat([nt[col] for nt in nts]...)
    end

    return NamedTuple{col_names}(Tuple(result_cols))
end

# =============================================================================
# Accessor functions
# =============================================================================

"""Return the grouping columns."""
groups(panel::PanelData) = panel.groups

"""Return the date column name."""
datecol(panel::PanelData) = panel.date

"""Return the seasonal period(s)."""
seasonal_period(panel::PanelData) = panel.m

"""Return the frequency."""
frequency(panel::PanelData) = panel.frequency

"""Return the target column name."""
target(panel::PanelData) = panel.target

# =============================================================================
# Display methods
# =============================================================================

function Base.show(io::IO, panel::PanelData)
    group_str = isempty(panel.groups) ? "(none)" : join(string.(panel.groups), ", ")
    date_str = isnothing(panel.date) ? "nothing" : string(panel.date)
    m_str = isnothing(panel.m) ? "nothing" :
            panel.m isa Vector ? "[$(join(panel.m, ", "))]" : string(panel.m)
    freq_str = isnothing(panel.frequency) ? "" : ", frequency=:$(panel.frequency)"
    target_str = isnothing(panel.target) ? "" : ", target=:$(panel.target)"

    print(io, "PanelData(groups=$group_str, date=$date_str, m=$m_str$freq_str$target_str)")
end

function Base.show(io::IO, ::MIME"text/plain", panel::PanelData)
    println(io, "PanelData")
    println(io, "─────────")

    # Core info
    group_str = isempty(panel.groups) ? "(none)" : join(string.(panel.groups), ", ")
    println(io, "  Groups: ", group_str)
    println(io, "  Date column: ", isnothing(panel.date) ? "(none)" : string(panel.date))

    # Frequency and m
    if panel.frequency !== nothing
        println(io, "  Frequency: ", format_frequency(panel.frequency))
    end
    if panel.m !== nothing
        if panel.m isa Vector
            println(io, "  Seasonal period(s): ", "[$(join(panel.m, ", "))]")
        else
            desc = m_description(panel.m, panel.frequency)
            desc_str = isempty(desc) ? "" : " ($desc)"
            println(io, "  Seasonal period: ", panel.m, desc_str)
        end
    end

    # Target
    if panel.target !== nothing
        println(io, "  Target: ", panel.target)
    end

    # Data info
    ct = Tables.columntable(panel.data)
    n_rows = length(ct[first(propertynames(ct))])
    n_cols = length(propertynames(ct))
    println(io, "  Data: $n_rows rows × $n_cols columns")

    # Preprocessing metadata
    has_meta = panel.time_fill_meta !== nothing ||
               panel.target_meta !== nothing ||
               !isempty(panel.xreg_meta)

    if has_meta
        println(io, "  Preprocessing:")

        if panel.time_fill_meta !== nothing
            meta = panel.time_fill_meta
            println(io, "    Time grid: $(meta.n_added) rows added (step: $(meta.step))")
        end

        if panel.target_meta !== nothing
            meta = panel.target_meta
            pct = round(100 * meta.n_imputed / meta.n_total, digits=1)
            println(io, "    Target (:$(panel.target)): $(meta.n_imputed) imputed ($pct%) via :$(meta.strategy)")
        end

        for (col, meta) in panel.xreg_meta
            pct = round(100 * meta.n_imputed / meta.n_total, digits=1)
            println(io, "    Exog (:$col): $(meta.n_imputed) imputed ($pct%) via :$(meta.strategy)")
        end
    end
end

# =============================================================================
# Multi-seasonality trait overrides
# Note: These are defined here because panel_data.jl is included after the
# spec files, so the types are available.
# =============================================================================

# TBATS supports multiple seasonal periods via Fourier representation
supports_multi_seasonality(::TbatsSpec) = true
