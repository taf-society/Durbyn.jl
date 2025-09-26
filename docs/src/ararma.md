# ARAR / ARARMA

## Forecasting Using Arar

### How Arar Model Works
The ARAR model applies a memory-shortening transformation if the underlying process of a given time series ${Y_{t}, t = 1, 2, ..., n}$ is "long-memory" then it fits an autoregressive model.

### Memory Shortening

The model follows five steps to classify ${Y_{t}}$ and take one of the following three actions:


* L: declare ${Y_{t}}$ as long memory and form  ${Y_{t}}$ by ${\tilde{Y}_{t} = Y_{t} - \hat{\phi}Y_{t - \hat{\tau}}}$
* M: declare ${Y_{t}}$ as moderately long memory and form  ${Y_{t}}$ by ${\tilde{Y}_{t} = Y_{t} - \hat{\phi}_{1}Y_{t -1} - \hat{\phi}_{2}Y_{t -2}}$
* S: declare ${Y_{t}}$ as short memory.

If ${Y_{t}}$ declared to be $L$ or $M$ then the series ${Y_{t}}$ is transformed again until. The transformation process continuous until the transformed series is classified as short memory. However, the maximum number of transformation process is three, it is very rare a time series require more than 2.

* 1. For each $\tau = 1, 2, ..., 15$, we find the value $\hat{\phi(\tau)} $ of $\hat{\phi}$ that minimizes $ERR(\phi, \tau) = \frac{\sum_{t=\tau +1 }^{n} [Y_{t} - \phi Y_{t-\tau}]^2 }{\sum_{t=\tau +1 }^{n} Y_{t}^{2}}$ then define $Err(\tau) = ERR(\hat{\phi(\tau), \tau})$ and choose the lag $\hat{\tau}$ to be the value of $\tau$ that minimizes  $Err(\tau)$.
* 2. If $Err(\hat{\tau}) \leq  8/n$,  ${Y_{t}}$ is a long-memory series.
* 3. If $\hat{\phi}( \hat{\tau} ) \geq 0.93$ and $\hat{\tau} > 2$,  ${Y_{t}}$ is a long-memory series.
* 4. If $\hat{\phi}( \hat{\tau} ) \geq 0.93$ and $\hat{\tau} = 1$ or $2$, ${Y_{t}}$ is a long-memory series.
* 5. If $\hat{\phi}( \hat{\tau} ) < 0.93$, ${Y_{t}}$ is a short-memory series.


### Subset Autoregressive Model:

In the following we will describe how ARAR algorithm fits an autoregressive process to the mean-corrected series $X_{t} = S_{t}- {\bar{S}}$, $t = k+1, ..., n$ where ${S_{t}, t = k + 1, ..., n}$ is the memory-shortened version of  ${Y_{t}}$ which derived from the five steps we described above and $\bar{S}$ is the sample mean of $S_{k+1}, ..., S_{n}$.

The fitted model has the following form:

$X_{t} = \phi_{1}X{t-1} + \phi_{1}X_{t-l_{1}} + \phi_{1}X_{t- l_{1}} + \phi_{1}X_{t-l_{1}} + Z$

where $Z \sim WN(0, \sigma^{2})$. The coefficients $\phi_{j}$ and white noise variance $\sigma^2$ can be derived from the Yule-Walker equations for given lags $l_1, l_2,$ and $l_3$ :

\begin{equation}
\begin{bmatrix}
1 & \hat{\rho}(l_1 - 1) & \hat{\rho}(l_2 - 1) & \hat{\rho}(l_3 - 1)\\
\hat{\rho}(l_1 - 1) &1 & \hat{\rho}(l_2 - l_1) & \hat{\rho}(l_3 - l_1)\\
\hat{\rho}(l_2 - 1) & \hat{\rho}(l_2 - l_1) & 1 & \hat{\rho}(l_2 - l_2)\\
\hat{\rho}(l_3 - 1) & \hat{\rho}(l_3 - l_1) & \hat{\rho}(l_3 - l_1) & 1
\end{bmatrix}*\begin{bmatrix}
\phi_{1} \\
\phi_{l_1} \\
\phi_{l_2}\\
\phi_{l_3}
\end{bmatrix} = \begin{bmatrix} \hat{\rho}(1) \\ \hat{\rho}(l_1) \\ \hat{\rho}(l_2)\\ \hat{\rho}(l_3) \end{bmatrix}
\end{equation}

and $\sigma^2 = \hat{\gamma}(0) [1-\phi_1\hat{\rho}(1)] - \phi_{l_1}\hat{\rho}(l_1)] - \phi_{l_2}\hat{\rho}(l_2)] - \phi_{l_3}\hat{\rho}(l_3)]$, where $\hat{\gamma}(j)$ and $\hat{\rho}(j), j = 0, 1, 2, ...,$ are the sample autocovariances and autocorelations of the series $X_{t}$.

The algorithm computes the coefficients of $\phi(j)$ for each set of lags where $1<l_1<l_2<l_3 \leq m$ where m chosen to be 13 or 26. The algorithm selects the model that the Yule-Walker estimate of $\sigma^2$ is minimal.

### Forecasting

If short-memory filter found in first step it has coefficients $\Psi_0, \Psi_1, ..., \Psi_k (k \geq0)$ where $\Psi_0 = 1$. In this case the transforemed series can be expressed as
\begin{equation}
    S_t = \Psi(B)Y_t = Y_t + \Psi_1 Y_{t-1} + ...+ \Psi_k Y_{t-k},
\end{equation}
where $\Psi(B) = 1 + \Psi_1B + ...+ \Psi_k B^k$ is polynomial in the back-shift operator.

If the coefficients of the subset autoregression found in the second step it has coefficients $\phi_1, \phi_{l_1},  \phi_{l_2}$ and $\phi_{l_3}$ then the subset AR model for $X_t = S_t - \bar{S}$ is

\begin{equation}
    \phi(B)X_t = Z_t,
\end{equation}

where $Z_t$ is a white-noise series with zero mean and constant variance and $\phi(B) = 1 - \phi_1B - \phi_{l_1}B^{l_1} - \phi_{l_2}B^{l_2} - \phi_{l_3}B^{l_3}$. From equation (1) and (2) one can obtain

\begin{equation}
    \xi(B)Y_t = \phi(1)\bar{S} + Z_t,
\end{equation}
where $\xi (B) = \Psi(B)\phi(B)$.

Assuming the fitted model in equation (3) is an appropriate model, and $Z_t$ is uncorrelated with $Y_j, j <t$ $\forall t \in T$, one can determine minimum mean squared error linear predictors $P_n Y_{n + h}$ of $Y_{n+h}$ in terms of ${1, Y_1, ..., Y_n}$ for $n > k + l_3$, from recursions

\begin{equation}
    P_n Y_{n+h} = - \sum_{j = 1}^{k + l_3} \xi P_nY_{n+h-j} + \phi(1)\bar{S},  h\geq 1,
\end{equation}
with the initial conditions $P_n Y_{n+h} = Y_{n + h}$, for $h\leq0$.

Ref: Brockwell, Peter J, and Richard A. Davis. Introduction to Time Series and Forecasting. [Springer](https://link.springer.com/book/10.1007/978-3-319-29854-2) (2016)

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
