# ARAR / ARARMA

```julia
using Durbyn
using Durbyn.Ararma

ap = air_passengers()

fit = arar(ap, max_ar_depth = 13)
fc  = forecast(fit, h = 12)
plot(fc)

fit2 = ararma(ap, p = 0, q = 1)
fc2  = forecast(fit2, h = 12)
plot(fc2)

fit3 = auto_ararma(ap)
fc3  = forecast(fit3, h = 12)
plot(fc3)
```
