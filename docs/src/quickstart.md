# Quick Start

## Installation

Install the development version:

```julia
using Pkg
Pkg.add(url="https://github.com/taf-society/Durbyn.jl")
```

## Formula Interface (Recommended)

Durbyn provides a modern, declarative interface for model specification with full support for tables, regressors (features in ML terminology), model comparison, and panel data.

### Example 1: Single Time Series

```julia
using Durbyn

# Load data
data = (sales = [120, 135, 148, 152, 141, 158, 170, 165, 180, 195],)

# ARIMA with automatic order selection
spec = ArimaSpec(@formula(sales = p() + q() + P() + Q()))
fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 12)
plot(fc)


# Load data another data
data = (passengers = air_passengers(),)

# ARIMA with automatic order selection
spec = ArimaSpec(@formula(passengers = p() + q() + P() + Q()))
fitted = fit(spec, data, m = 12)
fc = forecast(fitted, h = 12)
plot(fc)

```

### Example 2: With Regressors (Features)

```julia
using Durbyn
using Random
Random.seed!(123)

# Simulate data 
n = 120
idx = 1:n
temperature = -(15 .+ 8 .* sin.(2π .* idx ./ 12) .+ 0.5 .* randn(n))
marketing = -(0.3 .+ 0.15 .* sin.(2π .* idx ./ 6) .+ 0.05 .* randn(n))
sales = 120 .+ 1.5 .* temperature .+ 30 .* marketing .+ randn(n)
data = (sales = sales, temperature = temperature, marketing = marketing)

spec = ArimaSpec(@formula(sales = temperature + marketing + p() + d() + q() + P() + D() + Q()))
# spec = ArimaSpec(@formula(sales = auto()))
fitted = fit(spec, data, m = 12)

# Simulate future data
n_ahead = 24
future_idx = (n + 1):(n + n_ahead)
future_temp = -(15 .+ 8 .* sin.(2π .* future_idx ./ 12))
future_marketing = -(0.3 .+ 0.15 .* sin.(2π .* future_idx ./ 6))
newdata = (temperature = future_temp, marketing = future_marketing)


fc = forecast(fitted, h = n_ahead, newdata = newdata)

plot(fc)

```

### Example 3: Fitting Multiple Models Together

```julia
# Fit multiple model specifications at once
models = model(
    ArimaSpec(@formula(sales = p() + q())),
    EtsSpec(@formula(sales = e("A") + t("A") + s("A"))),
    ArimaSpec(@formula(sales = p(2) + d(1) + q(2))),
    names = ["auto_arima", "ets_aaa", "arima_212"]
)

# Fit all models
fitted = fit(models, data, m = 12)

# Forecast with all models
fc = forecast(fitted, h = 12)

# Compare results
for (name, model_result) in zip(models.names, fitted.models)
    println("$name: AIC = $(round(model_result.fit.aic, digits=2)), BIC = $(round(model_result.fit.bic, digits=2))")
end
```

### Example 4: Panel Data (Multiple Series)

!!! note "Optional Dependencies"
    This example requires `CSV` and `Downloads`:
    ```julia
    using Pkg
    Pkg.add(["CSV", "Downloads"])
    ```

```julia
using Durbyn, Durbyn.TableOps
using CSV, Downloads, Tables

# Load and reshape data
path = Downloads.download("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/refs/heads/main/Data/retail.csv")
wide = Tables.columntable(CSV.File(path))

long = pivot_longer(wide; id_cols = :date, names_to = :series, values_to = :value)
panel = PanelData(long; groupby = :series, date = :date, m = 12)
glimpse(panel)
# Fit model to all series at once
spec = ArimaSpec(@formula(value = p() + q()))
fitted = fit(spec, panel)

# Forecast all series
fc = forecast(fitted, h = 12)

# Get tidy forecast table
tbl = forecast_table(fc)
glimpse(tbl)
```

### Example 5: ETS Models with Formula

```julia

using Durbyn.Grammar

# Automatic ETS model selection
spec_ets = EtsSpec(@formula(sales = e("Z") + t("Z") + s("Z")))
fitted = fit(spec_ets, data, m = 12)
fc = forecast(fitted, h = 12)
plot(fc)
# Specialized ETS specifications
spec_ses = SesSpec(@formula(sales = ses()))
spec_holt = HoltSpec(@formula(sales = holt(damped=true)))
spec_hw = HoltWintersSpec(@formula(sales = hw(seasonal=:multiplicative)))

# Fit and forecast
fitted_ses = fit(spec_ses, data)
fc_ses = forecast(fitted_ses, h = 12)

plot(fc_ses)

```

---

## Base Models (Array Interface)

The array interface provides direct access to forecasting engines for working with numeric vectors.

### Exponential Smoothing (ETS)

```julia
using Durbyn
using Durbyn.ExponentialSmoothing

ap = air_passengers()

# Automatic ETS model selection
fit_ets = ets(ap, 12, "ZZZ")
fc_ets  = forecast(fit_ets, h = 12)
plot(fc_ets)

# Simple exponential smoothing
ses_fit = ses(ap)
ses_fc  = forecast(ses_fit, h = 12)
plot(ses_fc)

# Holt's linear trend method
holt_fit = holt(ap)
holt_fc  = forecast(holt_fit, h = 12)
plot(holt_fc)

# Holt-Winters seasonal method
hw_fit = holt_winters(ap, 12)
hw_fc  = forecast(hw_fit, h = 12)
plot(hw_fc)
```

### ARIMA

```julia
using Durbyn.Arima

ap = air_passengers()

# Manual ARIMA specification
arima_model = arima(ap, 12, order = PDQ(2,1,1), seasonal = PDQ(0,1,0))
fc  = forecast(arima_model, h = 12)
plot(fc)

# Automatic ARIMA selection
auto_arima_model = auto_arima(ap, 12)
fc_auto  = forecast(auto_arima_model, h = 12)
plot(fc_auto)
```

---

## Performance: Multi-Threading for Parallel Computing

Durbyn's `fit` function automatically leverages Julia's multi-threading for **massive parallel computing** when fitting models to panel data (multiple time series) or comparing multiple model specifications. Performance scales nearly linearly with CPU cores—from laptops to large cloud instances with 96+ cores.

### When Does Multi-Threading Help?

Multi-threading provides dramatic performance improvements when:
- **Fitting models to panel data** with many series (e.g., 40+ series)
- **Comparing multiple models** across series simultaneously
- **Running ensemble methods** or model selection procedures

**Example: Without multi-threading (1 thread)**
- Fitting 6 models to 42 series = 252 individual fits
- Time: ~5-10 minutes (sequential processing)

**With multi-threading**
- Same 252 fits processed in parallel
- 8 cores (laptop): ~60-90 seconds (5-8x faster)
- 16 cores (workstation): ~30-45 seconds (10-15x faster)
- 32+ cores (cloud): ~15-30 seconds (15-20x faster)

### How to Enable Multi-Threading

Julia must be started with multiple threads to enable parallel processing. Here are all the methods:

#### Method 1: VS Code Julia Extension Settings (Recommended for VS Code Users)

Add to your VS Code `settings.json` (File → Preferences → Settings → Open Settings JSON):

```json
{
    "julia.additionalArgs": [
        "-t",
        "auto"
    ]
}
```

**Options:**
- `"auto"` — Use all available CPU cores (recommended)
- Specific number (e.g., `"8"`, `"12"`) — Limit threads if you want to reserve cores for other tasks

**To apply:**
1. Save settings.json
2. Restart Julia REPL in VS Code (click trash icon in Julia REPL, then restart)
3. Verify with `Threads.nthreads()`

#### Method 2: Command Line

```bash
# Use all available cores (recommended)
julia -t auto

# Use specific number of threads (e.g., 8 threads)
julia -t 8

# Alternative syntax
julia --threads=auto
```

#### Method 3: Environment Variable (Persistent)

Set once and applies to all Julia sessions.

**Linux/macOS** — Add to `~/.bashrc`, `~/.zshrc`, or `~/.profile`:
```bash
export JULIA_NUM_THREADS=auto
```

**Windows (PowerShell)** — Add to PowerShell profile:
```powershell
$env:JULIA_NUM_THREADS = "auto"
```

**Windows (System Environment Variables):**
1. Search "Environment Variables" in Windows
2. Add new system variable: `JULIA_NUM_THREADS` = `auto`

After setting, restart terminal/IDE for changes to take effect.

#### Method 4: Julia Startup File

Create/edit `~/.julia/config/startup.jl` (Linux/macOS) or `%USERPROFILE%\.julia\config\startup.jl` (Windows):

```julia
# Set before Julia starts - less reliable, use environment variable instead
ENV["JULIA_NUM_THREADS"] = "auto"
```

**Note:** This method is less reliable. Prefer environment variable or command-line methods.

### Verifying Multi-Threading Is Active

```julia
julia> Threads.nthreads()
8  # Number of threads available (depends on your CPU and settings)

julia> Threads.threadpoolsize()
8  # Confirms thread pool size
```

### Real-World Example: Panel Data Model Comparison

Here's how multi-threading accelerates fitting multiple models to panel data:

```julia
using Durbyn, Durbyn.TableOps, Durbyn.Grammar
using CSV, Downloads, Tables

# Load panel data (42 retail series)
path = Downloads.download("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/refs/heads/main/Data/retail.csv")
wide = Tables.columntable(CSV.File(path))
long = pivot_longer(wide; id_cols=:date, names_to=:series, values_to=:value)
panel = PanelData(long; groupby=:series, date=:date, m=12)

# Define multiple models for comparison
models = model(
    ArarSpec(@formula(value = arar())),                                # ARAR
    ArimaSpec(@formula(value = p() + q())),                              # Auto ARIMA
    EtsSpec(@formula(value = e("Z") + t("Z") + s("Z") + drift(:auto))),  # Auto ETS with drift
    SesSpec(@formula(value = ses())),                                    # Simple exponential smoothing
    HoltSpec(@formula(value = holt(damped=true))),                       # Damped Holt
    HoltWintersSpec(@formula(value = hw(seasonal=:multiplicative))),     # Holt-Winters multiplicative
    CrostonSpec(@formula(value = croston())),                            # Croston's method
    names=["arar", "arima", "ets_auto", "ses", "holt_damped", "hw_mul", "croston"]
)

# Fit all models to all series IN PARALLEL
# Automatically uses available threads - no code changes needed!
fitted = fit(models, panel)
# Performance scales with cores:
#   1 thread:    ~400-500 seconds (baseline)
#   8 threads:   ~60-90 seconds (laptop/desktop)
#   16 threads:  ~30-45 seconds (workstation)
#   32+ threads: ~15-30 seconds (cloud/HPC)

# Generate forecasts (also parallelized)
fc = forecast(fitted, h=12)

# Convert to tidy table format
fc_tbl = forecast_table(fc)
glimpse(fc_tbl)
```

**What's happening under the hood:**
- 42 series × 7 models = **294 model fits**
- With multiple threads: Fits are distributed across available cores
- Each thread handles a series/model combination independently
- No code changes needed — parallelization is automatic!

### Troubleshooting

**Problem:** `Threads.nthreads()` returns 1
- **Solution:** Julia was started without `-t` flag. Restart Julia with multi-threading enabled.

**Problem:** VS Code settings not working
- **Solution:** Fully restart VS Code (not just Julia REPL). Settings only apply on fresh start.

**Problem:** Performance not improving
- **Solution:** Check you have enough series/models. Small datasets (< 10 series) may not show speedup due to threading overhead.

### Recommended Settings

- **Development/Interactive:** `julia -t auto` or VS Code settings with `"auto"`
- **Production/Scripts:** `export JULIA_NUM_THREADS=auto` in environment
- **Cloud/HPC Systems:** `julia -t auto` to leverage all available cores (e.g., 32, 64, 128+ threads)
- **Shared Systems:** Use specific number (e.g., `-t 8`) to avoid consuming all resources and leave cores for other users

---

## Next Steps

!!! tip "Complete End-to-End Example"
    Want to see a comprehensive workflow with train/test split, model comparison, accuracy evaluation, and visualization?
    Check out the **[Complete End-to-End Example in Grammar Guide](grammar.md#complete-end-to-end-example)** — demonstrates fitting 6 models to panel data with full evaluation pipeline.

**Documentation:**
- **[Grammar Guide](grammar.md)** — Complete formula interface documentation for ARIMA and ETS
- **[Table Operations](tableops.md)** — Data wrangling for time series and panel data
- **[ARIMA](arima.md)** — Formula interface and base models (ARIMA, SARIMA, Auto ARIMA)
- **[Exponential Smoothing](expsmoothing.md)** — Formula interface and base models (SES, Holt, Holt-Winters, ETS)
- **[Intermittent Demand](intermittent.md)** — Croston methods for sparse/intermittent data
- **[ARAR/ARARMA](ararma.md)** — Memory-shortening algorithms
