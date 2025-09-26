
# Forecasting Using ARIMA, SARIMA, ARIMAX, SARIMAX, and Auto ARIMA

## 1. ARIMA (AutoRegressive Integrated Moving Average)

### Definition
An ARIMA model is denoted as **ARIMA(p, d, q)**, where:
- **p**: order of the autoregressive (AR) part
- **d**: degree of differencing needed to achieve stationarity
- **q**: order of the moving average (MA) part

Formally, the model is written as:

```math
\Phi(B) \Delta^d X_t = \Theta(B) \varepsilon_t,
```

where:

- $B$ is the backshift operator ($BX_t = X_{t-1}$),
- $\Phi(B) = 1 - \phi_1B - \cdots - \phi_pB^p$,
- $\Theta(B) = 1 + \theta_1B + \cdots + \theta_qB^q$,
- $\Delta^d = (1 - B)^d$ is the differencing operator,
- $\varepsilon_t$ is white noise.

If $d = 0$, the model reduces to ARMA(p, q).

### Key Features
- Handles **non-stationary time series** via differencing.
- Shocks (innovations) have **permanent effects** for $d > 0$.
- Commonly used for macroeconomic and financial data.

---

## 2. SARIMA (Seasonal ARIMA)

### Definition
Seasonal ARIMA extends ARIMA to account for **seasonality**. It is denoted as:

```math
ARIMA(p, d, q)(P, D, Q)_m,
```

where:
- $P, D, Q$ are the seasonal AR, differencing, and MA orders,
- $m$ is the seasonal period (e.g., 12 for monthly data with yearly seasonality).

### Model Form
```math
\Phi(B)\Phi_s(B^m) \Delta^d \Delta_m^D X_t = \Theta(B)\Theta_s(B^m)\varepsilon_t,
```

where:
- $\Phi_s(B^m)$ and $\Theta_s(B^m)$ capture seasonal AR and MA terms,
- $\Delta_m^D = (1 - B^m)^D$ applies seasonal differencing.

### Key Features
- Captures both **short-term dynamics** (p, d, q) and **seasonal effects** (P, D, Q).
- Widely applied to monthly or quarterly economic indicators, sales, or climate data.

---

## 3. ARIMAX (ARIMA with Exogenous Variables)

### Definition
An ARIMAX model incorporates external regressors (covariates) into the ARIMA framework:

```math
\Phi(B) \Delta^d X_t = \beta Z_t + \Theta(B) \varepsilon_t,
```

where:

- $Z_t$ is a vector of exogenous predictors,
- $\beta$ are their coefficients.

### Key Features
- Useful when external factors (e.g., interest rates, marketing spend, policy variables) explain additional variance beyond past values of the series.
- Requires careful checking of exogeneity assumptions.

---

## 4. SARIMAX (Seasonal ARIMAX)

### Definition
SARIMAX generalizes SARIMA by including **exogenous regressors**:

```math
\Phi(B)\Phi_s(B^m) \Delta^d \Delta_m^D X_t = \beta Z_t + \Theta(B)\Theta_s(B^m)\varepsilon_t.
```

### Key Features
- Combines **seasonality** and **exogenous influences**.
- Powerful for real-world applications such as:
  - Forecasting retail sales with promotions (exogenous variable) and seasonal cycles.
  - Modeling energy demand with weather as an exogenous driver.

---

## 5. Auto ARIMA

### Definition
**Auto ARIMA** automates the process of identifying the best ARIMA/SARIMA model by searching across possible values of (p, d, q) and seasonal (P, D, Q), selecting the model that minimizes an information criterion such as AIC, AICc, or BIC.

### Algorithm (Hyndman & Khandakar, 2008)
1. **Unit root tests** (ADF, KPSS, or combinations) to determine differencing orders \( d \) and \( D \).
2. **Initial model selection** based on heuristics.  
3. **Stepwise search** over (p, q, P, Q) with bounds (e.g., up to 5 for non-seasonal and 2 for seasonal).  
4. Evaluate models by likelihood and information criteria.  
5. Refit the best model with full maximum likelihood.  

### Advantages
- Removes the manual effort of model identification.  
- Scales well to large numbers of series.  
- Ensures differencing is tested systematically (avoids over-differencing).

### Limitations
- Stepwise search may not find the global optimum.  
- Computationally expensive for very large seasonal periods.  
- Still requires diagnostic checking of residuals.  

---

## 6. Model Selection & Diagnostics

### Identification
- Use **ACF/PACF plots** and **unit root tests** (ADF, PP, KPSS) to choose orders manually (or confirm Auto ARIMA results).
- Differencing ensures stationarity ($d, D$).

### Estimation
- Maximum Likelihood Estimation (MLE) or Conditional Sum of Squares.

### Diagnostics
- Residual analysis: check for white noise.
- Information criteria: AIC, BIC, AICc.  
- Out-of-sample forecast validation.

---
# Forecasing in Julia Using Seasonal Arima Model

```julia
using Durbyn
using Durbyn.Arima

ap  = air_passengers()
fit = arima(ap, 12, order = PDQ(2,1,1), seasonal = PDQ(0,1,0))
fc  = forecast(fit, h = 12)
plot(fc)

```

# Forecasing in Julia Using Auto-Arima Model
```julia
fit2 = auto_arima(ap, 12)
fc2  = forecast(fit2, h = 12)
plot(fc)
```


## References
- Kunst, R. (2011). *Applied Time Series Analysis — Part II*. University of Vienna.  
- Hyndman, R.J., & Khandakar, Y. (2008). *Automatic Time Series Forecasting: The forecast Package for R*. Journal of Statistical Software, 27(3).  
- Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (1994). *Time Series Analysis, Forecasting and Control*.  
- Hamilton, J.D. (1994). *Time Series Analysis*.  
