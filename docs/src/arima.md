# ARIMA

Fit ARIMA models or let `auto_arima` choose orders.

```julia
using Durbyn
using Durbyn.Arima

ap  = air_passengers()
fit = arima(ap, 12, order = PDQ(2,1,1), seasonal = PDQ(0,1,0))
fc  = forecast(fit, h = 12)

fit2 = auto_arima(ap, 12, d = 1, D = 1)
fc2  = forecast(fit2, h = 12)

plot(fc)
```
