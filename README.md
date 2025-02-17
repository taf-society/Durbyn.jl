# Durbyn

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://akai01.github.io/Durbyn.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://akai01.github.io/Durbyn.jl/dev/)
[![Build Status](https://github.com/akai01/Durbyn.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/akai01/Durbyn.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/akai01/Durbyn.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/akai01/Durbyn.jl)


## About

**Durbyn** is a Julia package that implements the functionality of the R **forecast** package, providing tools for time series forecasting. The name "Durbyn" comes from the Kurdish word for binoculars—where *Dur* means "far" and *Byn* means "to see."

This package is currently under development and will be part of the **TAFS Forecasting Ecosystem**, an open-source initiative.

## About TAFS

**TAFS (Time Series Analysis and Forecasting Society)** is a non-profit association registered as a **"Verein"** in Vienna, Austria. The organization connects a global audience of academics, experts, practitioners, and students to engage, share, learn, and innovate in the fields of data science and artificial intelligence, with a particular focus on time-series analysis, forecasting, and decision science. [TAFS](https://taf-society.org/)


TAFS's mission includes:

- **Connecting**: Hosting events and discussion groups to establish connections and build a community of like-minded individuals.
- **Learning**: Providing a platform to learn about the latest research, real-world problems, and applications.
- **Sharing**: Inviting experts, academics, practitioners, and others to present and discuss problems, research, and solutions.
- **Innovating**: Supporting the transfer of research into solutions and helping to drive innovations.

As a registered non-profit association under Austrian law, TAFS ensures that all contributions remain fully open source and cannot be privatized or commercialized. [TAFS](https://taf-society.org/)

## License

The Durbyn package is licensed under the **MIT License**, allowing for open-source distribution and collaboration.

## Installation

Durbyn is still in development. Once it is officially released, you will be able to install it using Julia’s package manager:

```julia
using Pkg
Pkg.add("Durbyn")
```

For the latest development version, you can install directly from GitHub:

```julia
Pkg.add(url="https://github.com/akai01/Durbyn.jl")
```

## Usage

Durbyn provides ETS (Error, Trend, Seasonal) model functionality for time series forecasting. Here’s a basic example:

```julia
using Durbyn

# Load example dataset
ap = air_passengers()

# Fit an ETS model
fit = ets(ap, 12, "MAM", damped = false)

# Generate a forecast
fc = forecast_ets_base(fit, h = 12)

# Plot the forecast
plot(fc)
```

## Future Plans

Durbyn will introduce a **DataFrame-based interface** (tidy forecasting) similar to the **R fable** package, allowing for a more intuitive workflow when working with time series data.
