# ARAR and ARARMA Models

## ARAR Model
The ARAR model applies a memory-shortening transformation; if the underlying process of a time series ``\{Y_t,\ t=1,2,\ldots,n\}`` is “long-memory”, it then fits an autoregressive model.

### Memory Shortening

The model follows five steps to classify ``Y_t`` and take one of three actions:

- **L:** declare ``Y_t`` as long memory and form ``\tilde Y_t = Y_t - \hat\phi\, Y_{t-\hat\tau}``
- **M:** declare ``Y_t`` as moderately long memory and form ``\tilde Y_t = Y_t - \hat\phi_1 Y_{t-1} - \hat\phi_2 Y_{t-2}``
- **S:** declare ``Y_t`` as short memory.

If ``Y_t`` is declared **L** or **M**, the series is transformed again until the transformed series is classified as short memory. (At most three transformations are applied; in practice, more than two is rare.)

### Steps

1. For each ``\tau=1,2,\ldots,15``, find ``\hat\phi(\tau)`` that minimizes  
   ```math
   \mathrm{ERR}(\phi,\tau) \;=\; 
   \frac{\displaystyle\sum_{t=\tau+1}^{n}\!\big(Y_t-\phi\,Y_{t-\tau}\big)^2}
        {\displaystyle\sum_{t=\tau+1}^{n}\!Y_t^{\,2}},
   ```
   then set ``\mathrm{Err}(\tau)=\mathrm{ERR}\big(\hat\phi(\tau),\tau\big)`` and choose  
   ``\hat\tau=\arg\min_{\tau}\mathrm{Err}(\tau)``.
2. If ``\mathrm{Err}(\hat\tau)\le 8/n``, then ``Y_t`` is a long-memory series.
3. If ``\hat\phi(\hat\tau)\ge 0.93`` and ``\hat\tau>2``, then ``Y_t`` is a long-memory series.
4. If ``\hat\phi(\hat\tau)\ge 0.93`` and ``\hat\tau\in\{1,2\}``, then ``Y_t`` is a long-memory series.
5. If ``\hat\phi(\hat\tau)<0.93``, then ``Y_t`` is a short-memory series.

### Subset Autoregressive Model

We now describe how ARAR fits an autoregression to the mean-corrected series  
``X_t=S_t-\bar S``, ``t=k+1,\ldots,n``, where ``\{S_t\}`` is the memory-shortened version of ``\{Y_t\}`` obtained above and ``\bar S`` is the sample mean of ``S_{k+1},\ldots,S_n``.

The fitted model has the form
```math
X_t \;=\; \phi_1 X_{t-1} \;+\; \phi_{l_1} X_{t-l_1} \;+\; \phi_{l_2} X_{t-l_2} \;+\; \phi_{l_3} X_{t-l_3} \;+\; Z_t,
\qquad Z_t \sim \mathrm{WN}(0,\sigma^2).
```

### Yule–Walker Equations

The coefficients ``\phi_j`` and the noise variance ``\sigma^2`` follow from the Yule–Walker equations for given lags ``l_1,l_2,l_3``:

```math
\begin{bmatrix}
1 & \hat\rho(l_1-1) & \hat\rho(l_2-1) & \hat\rho(l_3-1)\\
\hat\rho(l_1-1) & 1 & \hat\rho(l_2-l_1) & \hat\rho(l_3-l_1)\\
\hat\rho(l_2-1) & \hat\rho(l_2-l_1) & 1 & \hat\rho(l_3-l_2)\\
\hat\rho(l_3-1) & \hat\rho(l_3-l_1) & \hat\rho(l_3-l_2) & 1
\end{bmatrix}
\begin{bmatrix}
\phi_1\\[2pt]
\phi_{l_1}\\[2pt]
\phi_{l_2}\\[2pt]
\phi_{l_3}
\end{bmatrix}
=
\begin{bmatrix}
\hat\rho(1)\\[2pt]
\hat\rho(l_1)\\[2pt]
\hat\rho(l_2)\\[2pt]
\hat\rho(l_3)
\end{bmatrix}.
```

```math
\sigma^2 \;=\; \hat\gamma(0)\,\Big( 1 - \phi_1\hat\rho(1) - \phi_{l_1}\hat\rho(l_1) - \phi_{l_2}\hat\rho(l_2) - \phi_{l_3}\hat\rho(l_3) \Big),
```
where ``\hat\gamma(j)`` and ``\hat\rho(j)``, ``j=0,1,2,\ldots``, are the sample autocovariances and autocorrelations of ``X_t``.  
The algorithm computes ``\phi(\cdot)`` for each set of lags with ``1<l_1<l_2<l_3\le m`` (``m`` typically 13 or 26) and selects the model with minimal Yule–Walker estimate of ``\sigma^2``.

### Forecasting

If the short-memory filter found in the first step has coefficients ``\Psi_0,\Psi_1,\ldots,\Psi_k`` (``k\ge0``, ``\Psi_0=1``), then
```math
S_t \;=\; \Psi(B)Y_t \;=\; Y_t + \Psi_1 Y_{t-1} + \cdots + \Psi_k Y_{t-k},
\qquad
\Psi(B) \;=\; 1 + \Psi_1 B + \cdots + \Psi_k B^k .
```

If the subset AR coefficients are ``\phi_1,\phi_{l_1},\phi_{l_2},\phi_{l_3}`` then, for ``X_t=S_t-\bar S``, 
```math
\phi(B)X_t \;=\; Z_t, \qquad 
\phi(B) \;=\; 1 - \phi_1 B - \phi_{l_1} B^{l_1} - \phi_{l_2} B^{l_2} - \phi_{l_3} B^{l_3}.
```

From the two displays above,
```math
\xi(B)Y_t \;=\; \phi(1)\,\bar S \;+\; Z_t, 
\qquad \xi(B) \;=\; \Psi(B)\,\phi(B).
```

Assuming this model is appropriate and ``Z_t`` is uncorrelated with ``Y_j`` for ``j<t``, the minimum-MSE linear predictors ``P_n Y_{n+h}`` of ``Y_{n+h}`` (for ``n>k+l_3``) satisfy the recursion
```math
P_n Y_{n+h} \;=\; - \sum_{j=1}^{k+l_3} \xi_j \, P_n Y_{n+h-j} \;+\; \phi(1)\,\bar S, \qquad h\ge 1,
```
with initial conditions ``P_n Y_{n+h}=Y_{n+h}`` for ``h\le 0``.

### Reference
- Brockwell, Peter J., and Richard A. Davis. *Introduction to Time Series and Forecasting*. [Springer](https://link.springer.com/book/10.1007/978-3-319-29854-2) (2016)


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
