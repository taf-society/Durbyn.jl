# Intermittent Demand

Tools for sparse series with many zeros, including Croston variants.

```julia
using Durbyn
using Durbyn.IntermittentDemand

data = [6, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 
0, 0, 0, 0, 0];

# Classical Croston method
crst = croston_classic(data)
fc   = forecast(crst, h = 12)
plot(fc, show_fitted = true)

# SBA bias-corrected
crst2 = croston_sba(data)
fc2   = forecast(crst2, h = 12)

# SBJ correction
crst3 = croston_sbj(data)
fc3   = forecast(crst3, h = 12)
```

**Inspection**
```julia
residuals(crst); residuals(fc);
fitted(crst);    fitted(fc);
```
