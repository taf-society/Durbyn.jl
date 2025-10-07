"""
# Example Time Series Datasets

This module provides classic time series datasets commonly used in forecasting
literature and examples. All datasets are returned as `Vector{Float64}`.

## Available Datasets

- [`air_passengers`](@ref): Monthly airline passengers (1949-1960)
- [`ausbeer`](@ref): Quarterly Australian beer production (1956-2010)
- [`lynx`](@ref): Annual Canadian lynx trappings (1821-1934)
- [`sunspots`](@ref): Monthly sunspot numbers (1749-1983)
- [`pedestrian_counts`](@ref): Daily pedestrian counts with weekly and annual seasonality (2009-2016)

## References

- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
"""

"""
    air_passengers() -> Vector{Float64}

Monthly airline passenger numbers (in thousands) from January 1949 to December 1960.

This is the classic Box & Jenkins airline passengers dataset, which exhibits both
trend and multiplicative seasonal patterns. The dataset contains 144 observations.

# Source
Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*.

# Properties
- **Frequency**: Monthly (12 observations per year)
- **Period**: 1949-01 to 1960-12
- **Length**: 144 observations
- **Characteristics**: Strong trend, multiplicative seasonality

# Example
```julia
using Durbyn.Utils

# Load the dataset
passengers = air_passengers()

# Basic statistics
println("Length: ", length(passengers))
println("Min: ", minimum(passengers))
println("Max: ", maximum(passengers))

# Plot or analyze the series
# using Plots
# plot(passengers, title="Air Passengers", ylabel="Thousands", xlabel="Month")
```
"""
function air_passengers()
    return [
        112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0,
        148.0, 136.0, 119.0, 104.0, 118.0, 115.0, 126.0,
        141.0, 135.0, 125.0, 149.0, 170.0, 170.0, 158.0,
        133.0, 114.0, 140.0, 145.0, 150.0, 178.0, 163.0,
        172.0, 178.0, 199.0, 199.0, 184.0, 162.0, 146.0,
        166.0, 171.0, 180.0, 193.0, 181.0, 183.0, 218.0,
        230.0, 242.0, 209.0, 191.0, 172.0, 194.0, 196.0,
        196.0, 236.0, 235.0, 229.0, 243.0, 264.0, 272.0,
        237.0, 211.0, 180.0, 201.0, 204.0, 188.0, 235.0,
        227.0, 234.0, 264.0, 302.0, 293.0, 259.0, 229.0,
        203.0, 229.0, 242.0, 233.0, 267.0, 269.0, 270.0,
        315.0, 364.0, 347.0, 312.0, 274.0, 237.0, 278.0,
        284.0, 277.0, 317.0, 313.0, 318.0, 374.0, 413.0,
        405.0, 355.0, 306.0, 271.0, 306.0, 315.0, 301.0,
        356.0, 348.0, 355.0, 422.0, 465.0, 467.0, 404.0,
        347.0, 305.0, 336.0, 340.0, 318.0, 362.0, 348.0,
        363.0, 435.0, 491.0, 505.0, 404.0, 359.0, 310.0,
        337.0, 360.0, 342.0, 406.0, 396.0, 420.0, 472.0,
        548.0, 559.0, 463.0, 407.0, 362.0, 405.0, 417.0,
        391.0, 419.0, 461.0, 472.0, 535.0, 622.0, 606.0,
        508.0, 461.0, 390.0, 432.0
    ]
end

"""
    ausbeer() -> Vector{Float64}

Quarterly Australian beer production in megalitres from Q1 1956 to Q2 2010.

This dataset shows Australian quarterly beer production with strong seasonal patterns
and a long-term trend. The data is from the Australian Bureau of Statistics.

# Source
Australian Bureau of Statistics. Cat. 8301.0.55.001.
R package `fpp2` (Hyndman, R.J., & Athanasopoulos, G.)

# Properties
- **Frequency**: Quarterly (4 observations per year)
- **Period**: 1956-Q1 to 2010-Q2
- **Length**: 218 observations
- **Characteristics**: Seasonal pattern, varying trend

# Example
```julia
using Durbyn.Utils

# Load the dataset
beer = ausbeer()

# Basic statistics
println("Length: ", length(beer))
println("Mean production: ", round(mean(beer), digits=2), " megalitres")

# Seasonal analysis (group by quarters)
# using Statistics
# q1 = beer[1:4:end]  # Q1 observations
# q4 = beer[4:4:end]  # Q4 observations
```
"""
function ausbeer()
    return [
        284.0, 213.0, 227.0, 308.0, 262.0, 228.0, 236.0, 320.0, 272.0, 233.0,
        237.0, 313.0, 261.0, 227.0, 250.0, 314.0, 286.0, 227.0, 260.0, 311.0,
        295.0, 233.0, 257.0, 339.0, 279.0, 250.0, 270.0, 346.0, 294.0, 255.0,
        278.0, 363.0, 313.0, 273.0, 300.0, 370.0, 331.0, 288.0, 306.0, 386.0,
        335.0, 288.0, 308.0, 402.0, 353.0, 316.0, 325.0, 405.0, 393.0, 319.0,
        327.0, 442.0, 383.0, 332.0, 361.0, 446.0, 387.0, 357.0, 374.0, 466.0,
        410.0, 370.0, 379.0, 487.0, 419.0, 378.0, 393.0, 506.0, 458.0, 387.0,
        427.0, 565.0, 465.0, 445.0, 450.0, 556.0, 500.0, 452.0, 435.0, 554.0,
        510.0, 433.0, 453.0, 548.0, 486.0, 453.0, 457.0, 566.0, 515.0, 464.0,
        431.0, 588.0, 503.0, 443.0, 448.0, 555.0, 513.0, 427.0, 473.0, 526.0,
        548.0, 440.0, 469.0, 575.0, 493.0, 433.0, 480.0, 576.0, 475.0, 405.0,
        435.0, 535.0, 453.0, 430.0, 417.0, 552.0, 464.0, 417.0, 423.0, 554.0,
        459.0, 428.0, 429.0, 534.0, 481.0, 416.0, 440.0, 538.0, 474.0, 440.0,
        447.0, 598.0, 467.0, 439.0, 446.0, 567.0, 485.0, 441.0, 429.0, 599.0,
        464.0, 424.0, 436.0, 574.0, 443.0, 410.0, 420.0, 532.0, 433.0, 421.0,
        410.0, 512.0, 449.0, 381.0, 423.0, 531.0, 426.0, 408.0, 416.0, 520.0,
        409.0, 398.0, 398.0, 507.0, 432.0, 398.0, 406.0, 526.0, 428.0, 397.0,
        403.0, 517.0, 435.0, 383.0, 424.0, 521.0, 421.0, 402.0, 414.0, 500.0,
        451.0, 380.0, 416.0, 492.0, 428.0, 408.0, 406.0, 506.0, 435.0, 380.0,
        421.0, 490.0, 435.0, 390.0, 412.0, 454.0, 416.0, 403.0, 408.0, 482.0,
        438.0, 386.0, 405.0, 491.0, 427.0, 383.0, 394.0, 473.0, 420.0, 390.0,
        410.0, 488.0, 415.0, 398.0, 419.0, 488.0, 414.0, 374.0
    ]
end

"""
    lynx() -> Vector{Float64}

Annual Canadian lynx trappings from 1821 to 1934.

This classic dataset shows the number of lynx trapped in the Mackenzie River district
of Canada. It exhibits a well-known cyclic pattern with a period of approximately 10 years,
often used to demonstrate cyclical behavior in time series.

# Source
Campbell, M.J. & Walker, A.M. (1977). "A Survey of statistical work on the Mackenzie River series of annual Canadian lynx trappings for the years 1821-1934 and a new analysis." *Journal of the Royal Statistical Society series A*, 140, 411-431.

# Properties
- **Frequency**: Annual
- **Period**: 1821 to 1934
- **Length**: 114 observations
- **Characteristics**: Strong cyclic pattern (~10 year cycle)

# Example
```julia
using Durbyn.Utils

# Load the dataset
lynx_data = lynx()

# Basic statistics
println("Length: ", length(lynx_data))
println("Mean trappings: ", round(mean(lynx_data), digits=2))
```
"""
function lynx()
    return [
        269.0, 321.0, 585.0, 871.0, 1475.0, 2821.0, 3928.0, 5943.0, 4950.0, 2577.0,
        523.0, 98.0, 184.0, 279.0, 409.0, 2285.0, 2685.0, 3409.0, 1824.0, 409.0,
        151.0, 45.0, 68.0, 213.0, 546.0, 1033.0, 2129.0, 2536.0, 957.0, 361.0,
        377.0, 225.0, 360.0, 731.0, 1638.0, 2725.0, 2871.0, 2119.0, 684.0, 299.0,
        236.0, 245.0, 552.0, 1623.0, 3311.0, 6721.0, 4254.0, 687.0, 255.0, 473.0,
        358.0, 784.0, 1594.0, 1676.0, 2251.0, 1426.0, 756.0, 299.0, 201.0, 229.0,
        469.0, 736.0, 2042.0, 2811.0, 4431.0, 2511.0, 389.0, 73.0, 39.0, 49.0,
        59.0, 188.0, 377.0, 1292.0, 4031.0, 3495.0, 587.0, 105.0, 153.0, 387.0,
        758.0, 1307.0, 3465.0, 6991.0, 6313.0, 3794.0, 1836.0, 345.0, 382.0, 808.0,
        1388.0, 2713.0, 3800.0, 3091.0, 2985.0, 3790.0, 674.0, 81.0, 80.0, 108.0,
        229.0, 399.0, 1132.0, 2432.0, 3574.0, 2935.0, 1537.0, 529.0, 485.0, 662.0,
        1000.0, 1590.0, 2657.0, 3396.0
    ]
end

"""
    sunspots() -> Vector{Float64}

Monthly mean relative sunspot numbers from January 1749 to December 1983.

This dataset contains monthly averages of daily sunspot numbers, showing the well-known
approximately 11-year solar cycle. Sunspot numbers are a measure of solar activity.

# Source
World Data Center-SILSO, Royal Observatory of Belgium.
Andrews, D.F. & Herzberg, A.M. (1985). *Data: A Collection of Problems from Many Fields for the Student and Research Worker*. Springer.

# Properties
- **Frequency**: Monthly
- **Period**: 1749-01 to 1983-12
- **Length**: 2820 observations
- **Characteristics**: Cyclic pattern (~11 years), non-stationary

# Example
```julia
using Durbyn.Utils

# Load the dataset
spots = sunspots()

# Basic statistics
println("Length: ", length(spots))
println("Max sunspot number: ", maximum(spots))

# Note: This is a large dataset (2820 observations)
# Useful for demonstrating long-term forecasting and cycle detection
```

# Note
Due to its length (2820 observations), this function returns a truncated version
(first 235 observations, approximately 1749-1768). For the full dataset, consider
downloading from official sources.
"""
function sunspots()
    # First ~235 observations (1749-1768) for demonstration purposes
    # Full dataset available from SILSO: http://www.sidc.be/silso/datafiles
    return [58.0, 62.6, 70.0, 55.7, 85.0, 83.5, 94.8, 66.3, 75.9, 75.5, 158.6, 85.2, 
    73.3, 75.9, 89.2, 88.3, 90.0, 100.0, 85.4, 103.0, 91.2, 65.7, 63.3, 75.4, 70.0, 
    43.5, 45.3, 56.4, 60.7, 50.7, 66.3, 59.8, 23.5, 23.2, 28.5, 44.0, 35.0, 50.0, 71.0,
    59.3, 59.7, 39.6, 78.4, 29.3, 27.1, 46.6, 37.6, 40.0, 44.0, 32.0, 45.7, 38.0, 36.0,
    31.7, 22.2, 39.0, 28.0, 25.0, 20.0, 6.7, 0.0, 3.0, 1.7, 13.7, 20.7, 26.7, 18.8, 12.3, 
    8.2, 24.1, 13.2, 4.2, 10.2, 11.2, 6.8, 6.5, 0.0, 0.0, 8.6, 3.2, 17.8, 23.7, 6.8, 20.0, 
    12.5, 7.1, 5.4, 9.4, 12.5, 12.9, 3.6, 6.4, 11.8, 14.3, 17.0, 9.4, 14.1, 21.2, 26.2, 
    30.0, 38.1, 12.8, 25.0, 51.3, 39.7, 32.5, 64.7, 33.5, 37.6, 52.0, 49.0, 72.3, 46.4, 45.0,
    44.0, 38.7, 62.5, 37.7, 43.0, 43.0, 48.3, 44.0, 46.8, 47.0, 49.0, 50.0, 51.0, 71.3, 77.2,
    59.7, 46.3, 57.0, 67.3, 59.5, 74.7, 58.3, 72.0, 48.3, 66.0, 75.6, 61.3, 50.6, 59.7, 61.0,
    70.0, 91.0, 80.7, 71.7, 107.2, 99.3, 94.1, 91.1, 100.7, 88.7, 89.7, 46.0, 43.8, 72.8, 45.7, 
    60.2, 39.9, 77.1, 33.8, 67.7, 68.5, 69.3, 77.8, 77.2, 56.5, 31.9, 34.2, 32.9, 32.7, 35.8, 54.2,
    26.5, 68.1, 46.3, 60.9, 61.4, 59.7, 59.7, 40.2, 34.4, 44.3, 30.0, 30.0, 30.0, 28.2, 28.0,
    26.0, 25.7, 24.0, 26.0, 25.0, 22.0, 20.2, 20.0, 27.0, 29.7, 16.0, 14.0, 14.0, 13.0, 12.0,
    11.0, 36.6, 6.0, 26.8, 3.0, 3.3, 4.0, 4.3, 5.0, 5.7, 19.2, 27.4, 30.0, 43.0, 32.9, 29.8,
    33.3, 21.9, 40.8, 42.7, 44.1, 54.7, 53.3, 53.5, 66.1]

end

"""
    pedestrian_counts() -> Vector{Float64}

Daily pedestrian counts from a city sensor location (2009-2016).

This dataset contains daily pedestrian traffic counts exhibiting both weekly seasonality
(period=7, lower counts on weekends) and annual seasonality (period=365.25, seasonal
patterns across the year). Ideal for demonstrating multiple seasonal patterns and
complex seasonality in forecasting models.

# Source
Simulated based on patterns from Melbourne Pedestrian Counting System.
Similar datasets available from: City of Melbourne Open Data Portal.

# Properties
- **Frequency**: Daily
- **Period**: 2009-01-01 to 2016-12-31 (2922 days, ~8 years)
- **Length**: 2922 observations
- **Characteristics**:
  - Weekly seasonality (period=7): Lower weekend traffic
  - Annual seasonality (period=365.25): Seasonal variations
  - Trend component
  - Multiple seasonal patterns

# Example
```julia
using Durbyn.Utils

# Load the dataset
pedestrians = pedestrian_counts()

# Basic statistics
println("Length: ", length(pedestrians))
println("Mean daily count: ", round(mean(pedestrians), digits=2))

# Weekly pattern analysis
# weekly_avg = [mean(pedestrians[i:7:end]) for i in 1:7]
# println("Mon-Sun averages: ", weekly_avg)

# This dataset is useful for testing models with multiple seasonal periods
# like TBATS, complex exponential smoothing, or Prophet-style models
```

# Note
This is a subset covering 8 years (2922 days) to demonstrate daily data with
dual seasonality patterns common in business/retail contexts.
"""
function pedestrian_counts()
    # Generate realistic daily pedestrian data with weekly (7) and annual (365) seasonality
    # Covering ~8 years (2922 days from 2009-2016)
    n = 2922
    t = 1:n

    # Base level with upward trend
    base = 5000.0 .+ 0.5 .* t

    # Weekly seasonality (period=7) - lower on weekends
    weekly = [i % 7 in [6, 0] ? -800.0 : 400.0 * sin(2π * i / 7) for i in t]

    # Annual seasonality (period=365.25) - summer peak, winter low
    annual = 1500.0 .* sin.(2π .* t ./ 365.25 .+ π/2)

    # Add some realistic noise
    noise = randn(n) .* 200.0

    # Combine components
    data = base .+ weekly .+ annual .+ noise

    # Ensure no negative values
    data = max.(data, 100.0)

    return Vector{Float64}(data)
end

"""
    simulate_seasonal_data(n::Int=365; m::Int=12, trend::Bool=true,
                          seasonal_strength::Float64=1.0, noise_level::Float64=0.1,
                          base_level::Float64=100.0, trend_coef::Float64=0.1) -> Vector{Float64}

Simulate time series data with configurable seasonality patterns.

This function generates synthetic time series data with trend, seasonal, and noise components.
It's useful for testing forecasting models, creating examples, and understanding different
seasonal patterns.

# Arguments
- `n::Int=365`: Number of observations to generate
- `m::Int=12`: Seasonal period (e.g., 12 for monthly, 4 for quarterly, 7 for daily/weekly, 24 for hourly)
- `trend::Bool=true`: Include linear trend component
- `seasonal_strength::Float64=1.0`: Multiplier for seasonal amplitude (0.0 = no seasonality)
- `noise_level::Float64=0.1`: Standard deviation of random noise as fraction of base_level
- `base_level::Float64=100.0`: Base level of the series
- `trend_coef::Float64=0.1`: Trend coefficient per observation

# Returns
- `Vector{Float64}`: Simulated time series data

# Common Frequency Values
- `m=1`: No seasonality (random walk with trend)
- `m=4`: Quarterly data
- `m=7`: Weekly pattern (for daily data)
- `m=12`: Monthly data
- `m=24`: Hourly pattern (for sub-daily data)
- `m=52`: Weekly data (for annual patterns)
- `m=365`: Daily data (for multi-year patterns)

# Examples

```julia
using Durbyn.Utils

# Monthly data with seasonality (like air passengers)
monthly_data = simulate_seasonal_data(144, m=12, base_level=100.0, trend_coef=0.5)

# Quarterly data (like ausbeer)
quarterly_data = simulate_seasonal_data(100, m=4, seasonal_strength=1.5)

# Daily data with weekly pattern
daily_data = simulate_seasonal_data(365, m=7, base_level=1000.0)

# Hourly data with daily pattern
hourly_data = simulate_seasonal_data(24*30, m=24, base_level=50.0, noise_level=0.2)

# No seasonality (random walk)
no_seasonal = simulate_seasonal_data(100, m=1, seasonal_strength=0.0)

# Strong seasonality, no trend
strong_seasonal = simulate_seasonal_data(200, m=12, trend=false, seasonal_strength=2.0)

# Multiple seasonal periods (simulate daily data with both weekly and annual patterns)
# For complex seasonality, call the function and add components manually:
n = 365 * 2
weekly = simulate_seasonal_data(n, m=7, trend=false, base_level=0.0, seasonal_strength=0.5)
annual = simulate_seasonal_data(n, m=365, trend=false, base_level=0.0, seasonal_strength=1.0)
trend_only = simulate_seasonal_data(n, m=1, seasonal_strength=0.0, base_level=100.0)
complex_data = trend_only .+ weekly .+ annual
```

# Details

The generated series follows the decomposition:
```
Y(t) = Base + Trend(t) + Seasonal(t) + Noise(t)
```

where:
- `Base` = `base_level`
- `Trend(t)` = `trend_coef * t` (if trend=true)
- `Seasonal(t)` = `seasonal_strength * base_level * sin(2π * t / m)`
- `Noise(t)` ~ Normal(0, `noise_level * base_level`)
"""
function simulate_seasonal_data(n::Int=365;
                               m::Int=12,
                               trend::Bool=true,
                               seasonal_strength::Float64=1.0,
                               noise_level::Float64=0.1,
                               base_level::Float64=100.0,
                               trend_coef::Float64=0.1)

    # Validate inputs
    n > 0 || throw(ArgumentError("n must be positive"))
    m > 0 || throw(ArgumentError("m must be positive"))
    seasonal_strength >= 0 || throw(ArgumentError("seasonal_strength must be non-negative"))
    noise_level >= 0 || throw(ArgumentError("noise_level must be non-negative"))

    t = 1:n

    # Base level
    data = fill(base_level, n)

    # Add trend component
    if trend
        data .+= trend_coef .* t
    end

    # Add seasonal component
    if m > 1 && seasonal_strength > 0
        seasonal = seasonal_strength .* base_level .* sin.(2π .* t ./ m)
        data .+= seasonal
    end

    # Add noise component
    if noise_level > 0
        noise = randn(n) .* (noise_level * base_level)
        data .+= noise
    end

    return Vector{Float64}(data)
end
