function show(io::IO, s::ArimaStateSpace)
    println(io, "ArimaStateSpace:")
    println(io, "  phi   (AR coefficients):         ", s.phi)
    println(io, "  theta (MA coefficients):         ", s.theta)
    println(io, "  Delta (Differencing coeffs):     ", s.Delta)
    println(io, "  Z     (Observation coeffs):      ", s.Z)
    println(io, "  a     (Current state estimate):  ", s.a)
    println(io, "  P     (Current state covariance):")
    show(io, "text/plain", s.P)
    println(io, "\n  T     (Transition matrix):")
    show(io, "text/plain", s.T)
    println(io, "\n  V     (Innovations or 'RQR'):    ", s.V)
    println(io, "  h     (Observation variance):    ", s.h)
    println(io, "  Pn    (Prior state covariance):")
    show(io, "text/plain", s.Pn)
end

function show(io::IO, fit::ArimaFit)
    println(io, "ARIMA Fit Summary")
    println(io, "-----------------")
    println(io, "Coefficients:")
    show(io, fit.coef)
    println(io, "\nSigma²: ", fit.sigma2)
    println(io, "Log-likelihood: ", fit.loglik)
    aic_val = fit.aic
    if !isnothing(aic_val) && !isnan(aic_val)
        println(io, "AIC: ", aic_val)
    end
end
