# Utilities Module

The Utils module provides helper functions and utilities used throughout Durbyn.jl. This includes example datasets for testing and learning, data manipulation functions, and other supporting tools.

---

## Example Datasets

Durbyn.jl provides several classic time series datasets commonly used in forecasting literature and examples. All datasets are returned as `Vector{Float64}`.

### Available Datasets

| Dataset | Frequency | Length | Period | Characteristics |
|---------|-----------|--------|--------|-----------------|
| `air_passengers` | Monthly | 144 | 1949-1960 | Trend, multiplicative seasonality |
| `ausbeer` | Quarterly | 218 | 1956-2010 | Seasonal pattern, varying trend |
| `lynx` | Annual | 114 | 1821-1934 | Cyclic (~10 year cycle) |
| `sunspots` | Monthly | 235* | 1749-1768 | Cyclic (~11 year solar cycle) |
| `pedestrian_counts` | Daily | 2922 | 2009-2016 | Weekly + annual seasonality |
| `simulate_seasonal_data` | Configurable | User-defined | - | Synthetic data generator |

*Truncated for demonstration; full dataset has 2820 observations.

---

### Real-World Datasets

#### `air_passengers`

Monthly airline passenger numbers (in thousands) from January 1949 to December 1960. This is the classic Box & Jenkins dataset exhibiting both trend and multiplicative seasonal patterns.

```julia
using Durbyn

ap = air_passengers()
println("Length: ", length(ap))      # 144
println("Range: ", extrema(ap))      # (104.0, 622.0)

# Fit a model
fit = ets(ap, 12, "ZZZ")
fc = forecast(fit, h=12)
plot(fc)
```

**Properties:**
- **Frequency**: 12 (monthly)
- **Characteristics**: Strong upward trend, multiplicative seasonality with increasing amplitude

**Source:** Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*.

---

#### `ausbeer`

Quarterly Australian beer production in megalitres from Q1 1956 to Q2 2010.

```julia
using Durbyn

beer = ausbeer()
println("Length: ", length(beer))    # 218
println("Mean: ", round(mean(beer), digits=1))  # ~430 megalitres

# Fit Holt-Winters
fit = holt_winters(beer, 4)
fc = forecast(fit, h=8)
plot(fc)
```

**Properties:**
- **Frequency**: 4 (quarterly)
- **Characteristics**: Strong Q4 peak (summer in Australia), varying long-term trend

**Source:** Australian Bureau of Statistics, Cat. 8301.0.55.001.

---

#### `lynx`

Annual Canadian lynx trappings from 1821 to 1934 in the Mackenzie River district.

```julia
using Durbyn

lynx_data = lynx()
println("Length: ", length(lynx_data))  # 114
println("Range: ", extrema(lynx_data))  # (39.0, 6991.0)

# Good for demonstrating cyclic patterns
# Note: This is annual data, so m=1 (no within-year seasonality)
```

**Properties:**
- **Frequency**: 1 (annual)
- **Characteristics**: Famous ~10-year population cycle, predator-prey dynamics

**Source:** Campbell, M.J. & Walker, A.M. (1977). *Journal of the Royal Statistical Society Series A*, 140, 411-431.

---

#### `sunspots`

Monthly mean relative sunspot numbers showing the ~11-year solar cycle.

```julia
using Durbyn

spots = sunspots()
println("Length: ", length(spots))   # 235 (truncated)
println("Max: ", maximum(spots))     # ~158

# Demonstrates long cycles in time series
```

**Properties:**
- **Frequency**: 12 (monthly)
- **Characteristics**: ~11-year solar cycle, non-stationary

**Note:** This is a truncated version (1749-1768). Full dataset available from [SILSO](http://www.sidc.be/silso/datafiles).

**Source:** World Data Center-SILSO, Royal Observatory of Belgium.

---

#### `pedestrian_counts`

Daily pedestrian counts from a city sensor location (2009-2016), exhibiting multiple seasonal patterns.

```julia
using Durbyn

pedestrians = pedestrian_counts()
println("Length: ", length(pedestrians))  # 2922 (~8 years)

# Ideal for testing multiple seasonality models
# Weekly pattern (period=7) + Annual pattern (period=365)
using Durbyn.Bats
fit = tbats(pedestrians, [7, 365.25])
fc = forecast(fit, h=30)
```

**Properties:**
- **Frequency**: Daily (multiple seasonal periods)
- **Characteristics**:
  - Weekly seasonality (period=7): Lower weekend traffic
  - Annual seasonality (period=365.25): Seasonal variations
  - Upward trend

**Source:** Simulated based on Melbourne Pedestrian Counting System patterns.

---

### Synthetic Data Generator

#### `simulate_seasonal_data`

Generate synthetic time series with configurable components for testing and experimentation.

```julia
simulate_seasonal_data(n=365; m=12, trend=true, seasonal_strength=1.0,
                       noise_level=0.1, base_level=100.0, trend_coef=0.1)
```

**Arguments:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | 365 | Number of observations |
| `m` | 12 | Seasonal period |
| `trend` | true | Include linear trend |
| `seasonal_strength` | 1.0 | Seasonal amplitude multiplier |
| `noise_level` | 0.1 | Noise as fraction of base level |
| `base_level` | 100.0 | Base level of the series |
| `trend_coef` | 0.1 | Trend coefficient per observation |

**Common Frequency Values:**

| `m` | Data Type |
|-----|-----------|
| 4 | Quarterly |
| 7 | Daily with weekly pattern |
| 12 | Monthly |
| 24 | Hourly with daily pattern |
| 52 | Weekly with annual pattern |
| 365 | Daily with annual pattern |

**Examples:**

```julia
using Durbyn

# Monthly data similar to air_passengers
monthly = simulate_seasonal_data(144; m=12, base_level=100.0, trend_coef=0.5)

# Quarterly data similar to ausbeer
quarterly = simulate_seasonal_data(100; m=4, seasonal_strength=1.5)

# Daily data with weekly pattern
daily = simulate_seasonal_data(365; m=7, base_level=1000.0)

# No seasonality (for testing trend-only models)
trend_only = simulate_seasonal_data(100; m=1, seasonal_strength=0.0)

# Strong seasonality, no trend
seasonal_only = simulate_seasonal_data(200; m=12, trend=false, seasonal_strength=2.0)
```

**Complex Seasonality (Multiple Periods):**

```julia
# Daily data with both weekly and annual patterns
n = 365 * 2
weekly = simulate_seasonal_data(n; m=7, trend=false, base_level=0.0, seasonal_strength=0.5)
annual = simulate_seasonal_data(n; m=365, trend=false, base_level=0.0, seasonal_strength=1.0)
trend_only = simulate_seasonal_data(n; m=1, seasonal_strength=0.0, base_level=100.0)
complex_data = trend_only .+ weekly .+ annual
```

**Generated Series Structure:**

```math
Y(t) = \text{Base} + \text{Trend}(t) + \text{Seasonal}(t) + \text{Noise}(t)
```

where:
- Base = `base_level`
- Trend(t) = `trend_coef * t` (if `trend=true`)
- Seasonal(t) = `seasonal_strength * base_level * sin(2π * t / m)`
- Noise(t) ~ Normal(0, `noise_level * base_level`)

---

### Quick Reference

```julia
using Durbyn

# Load all datasets
ap = air_passengers()      # Monthly, m=12, classic Box-Jenkins
beer = ausbeer()           # Quarterly, m=4, Australian beer
lynx_data = lynx()         # Annual, m=1, cyclic pattern
spots = sunspots()         # Monthly, m=12, solar cycle
peds = pedestrian_counts() # Daily, m=[7, 365], multi-seasonal

# Generate custom data
custom = simulate_seasonal_data(100; m=12, seasonal_strength=1.5)
```

---

## References

- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Campbell, M.J. & Walker, A.M. (1977). *A Survey of statistical work on the Mackenzie River series*. Journal of the Royal Statistical Society Series A, 140, 411-431.
