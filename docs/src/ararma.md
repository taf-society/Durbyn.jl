# ARAR and ARARMA Models

## ARAR Model
The ARAR model applies a memory-shortening transformation if the underlying process of a given time series  
$Y_{t},\; t = 1,2,\dots,n$ is "long-memory." After transformation, it fits an autoregressive model.

---

### Memory Shortening

The model follows five steps to classify $Y_t$ and take one of the following three actions:

- **L**: declare $Y_t$ as long memory and form  
  $$
  \tilde{Y}_t = Y_t - \hat{\phi} Y_{t-\hat{\tau}}
  $$

- **M**: declare $Y_t$ as moderately long memory and form  
  $$
  \tilde{Y}_t = Y_t - \hat{\phi}_{1} Y_{t-1} - \hat{\phi}_{2} Y_{t-2}
  $$

- **S**: declare $Y_t$ as short memory.

If $Y_t$ is declared L or M, then the series is transformed again until classified as short memory.  
The transformation process continues until classification as short memory, but at most three times (rarely more than two).

---

### Steps

1. For each $\tau = 1,2,\dots,15$, find $\hat{\phi}(\tau)$ that minimizes

   $$
   ERR(\phi,\tau) \;=\;
   \frac{\sum_{t=\tau+1}^{n} \big(Y_t - \phi Y_{t-\tau}\big)^2}
        {\sum_{t=\tau+1}^{n} Y_t^{2}}
   $$

   Define $Err(\tau) = ERR(\hat{\phi}(\tau), \tau)$ and choose  
   $\hat{\tau}$ as the value of $\tau$ that minimizes $Err(\tau)$.

2. If $Err(\hat{\tau}) \leq \tfrac{8}{n}$, declare $Y_t$ long-memory.  
3. If $\hat{\phi}(\hat{\tau}) \geq 0.93$ and $\hat{\tau}>2$, declare long-memory.  
4. If $\hat{\phi}(\hat{\tau}) \geq 0.93$ and $\hat{\tau}=1$ or $2$, declare long-memory.  
5. If $\hat{\phi}(\hat{\tau}) < 0.93$, declare short-memory.

---

### Subset Autoregressive Model

After shortening, define the mean-corrected series

$$
X_t = S_t - \bar{S}, \quad t = k+1,\dots,n
$$

where $S_t$ is the memory-shortened series and $\bar{S}$ is the sample mean of $S_{k+1},\dots,S_n$.

The fitted model is

$$
X_t = \phi_1 X_{t-1} + \phi_{l_1} X_{t-l_1} + \phi_{l_2} X_{t-l_2} + \phi_{l_3} X_{t-l_3} + Z_t,
\qquad Z_t \sim WN(0,\sigma^2).
$$

### Yule–Walker Equations

The coefficients $\phi_j$ and white-noise variance $\sigma^2$ come from:

$$
\begin{bmatrix}
1 & \hat{\rho}(l_1-1) & \hat{\rho}(l_2-1) & \hat{\rho}(l_3-1) \\
\hat{\rho}(l_1-1) & 1 & \hat{\rho}(l_2-l_1) & \hat{\rho}(l_3-l_1) \\
\hat{\rho}(l_2-1) & \hat{\rho}(l_2-l_1) & 1 & \hat{\rho}(l_3-l_2) \\
\hat{\rho}(l_3-1) & \hat{\rho}(l_3-l_1) & \hat{\rho}(l_3-l_2) & 1
\end{bmatrix}
\begin{bmatrix}
\phi_1 \\ \phi_{l_1} \\ \phi_{l_2} \\ \phi_{l_3}
\end{bmatrix}
=
\begin{bmatrix}
\hat{\rho}(1) \\ \hat{\rho}(l_1) \\ \hat{\rho}(l_2) \\ \hat{\rho}(l_3)
\end{bmatrix}
$$

and

$$
\sigma^2 = \hat{\gamma}(0) \Big(
  1 - \phi_1 \hat{\rho}(1)
    - \phi_{l_1} \hat{\rho}(l_1)
    - \phi_{l_2} \hat{\rho}(l_2)
    - \phi_{l_3} \hat{\rho}(l_3)
\Big),
$$

where $\hat{\gamma}(j)$ and $\hat{\rho}(j)$ are the sample autocovariances and autocorrelations.

The algorithm computes $\phi(j)$ for all sets of lags with $1<l_1<l_2<l_3\leq m$, $m=13$ or $26$.  
It selects the model with minimal $\sigma^2$.

---

### Forecasting

If a short-memory filter is found in the first step, with coefficients  
$\Psi_0,\Psi_1,\dots,\Psi_k$ ($\Psi_0=1$), then

$$
S_t = \Psi(B) Y_t = Y_t + \Psi_1 Y_{t-1} + \dots + \Psi_k Y_{t-k},
$$

where

$$
\Psi(B) = 1 + \Psi_1 B + \dots + \Psi_k B^k
$$

is a polynomial in the backshift operator.

If subset AR coefficients are found, the model is

$$
\phi(B) X_t = Z_t,
$$

with

$$
\phi(B) = 1 - \phi_1 B - \phi_{l_1} B^{l_1} - \phi_{l_2} B^{l_2} - \phi_{l_3} B^{l_3},
\quad X_t = S_t - \bar{S}.
$$

From this:

$$
\xi(B) Y_t = \phi(1)\bar{S} + Z_t,
\qquad \xi(B) = \Psi(B)\phi(B).
$$

---

### Predictors

Assuming the fitted model is valid and $Z_t$ is uncorrelated with past $Y_j$:

$$
P_n Y_{n+h} = -\sum_{j=1}^{k+l_3} \xi_j \, P_n Y_{n+h-j} + \phi(1)\bar{S},
\qquad h \geq 1,
$$

with initial conditions

$$
P_n Y_{n+h} = Y_{n+h}, \quad h \leq 0.
$$

---

## Reference
- Brockwell, Peter J, and Richard A. Davis. *Introduction to Time Series and Forecasting*. [Springer, 2016](https://link.springer.com/book/10.1007/978-3-319-29854-2)


# Forecasing in Julia using Arar Model

```julia
using Durbyn
using Durbyn.Ararma

ap = air_passengers()

fit = arar(ap, max_ar_depth = 13)
fc  = forecast(fit, h = 12)
plot(fc)

```
# Forecasing in Julia using Ararma Model
docs #TODO

```julia
fit2 = ararma(ap, p = 0, q = 1)
fc2  = forecast(fit2, h = 12)
plot(fc2)

fit3 = auto_ararma(ap)
fc3  = forecast(fit3, h = 12)
plot(fc3)
```
