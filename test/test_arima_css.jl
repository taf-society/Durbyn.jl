y = collect(1.0:10.0) 
arma = [0,0,0,0,0,0,0]
phi = [ 0.5 ,  1.  , -0.25,  0.25, -0.25, -0.25]
theta = [0.5 , 1.  , 0.25, 0.75, 0.25, 0.25]
ncond = 3


resid_var, resid = arima_css(y, arma, phi, theta, ncond)
@test isapprox(resid_var, 0.18831307547433035)
@test isapprox(resid, [0., 0., 0. , 0.75, -0.125, -0.6875, 0.28125, 0.015625, -0.2109375 ,0.37890625])

y = collect(1.0:10.0) 
arma = [0,0,0,0,0, 0,0]
phi   = Float64[]
theta = Float64[]
ncond = 3

resid_var, resid = arima_css(y, arma, phi, theta, ncond)

@test isapprox(resid_var, 53.0; atol=1e-12)
@test resid[1:3] == [0,0,0]
@test resid[4:10] == [4,5,6,7,8,9,10]




y = collect(1.0:5.0)
arma = [0,0,0,0,0, 1,0]
phi   = Float64[]
theta = Float64[]
ncond = 2

resid_var, resid = arima_css(y, arma, phi, theta, ncond)

@test isapprox(resid_var, 1.0; atol=1e-12)
@test resid == [0, 0, 1, 1, 1]


y = collect(1.0:12.0)
arma = [0, 0, 0, 0, 4, 0, 1]
phi = Float64[]
theta = Float64[]
ncond = 2
resid_var, resid = arima_css(y, arma, phi, theta, ncond)
@test resid[1:2] == [0.0, 0.0]
@test !any(isnan.(resid))
@test resid_var > 0.0


y = [1.0, 2.0, NaN, 4.0, 5.0, 6.0]
arma = [0, 0, 0, 0, 0, 0, 0]
phi = [1.0]
theta = Float64[]
ncond = 1

resid_var, resid = arima_css(y, arma, phi, theta, ncond)
@test resid[1] == 0.0
@test isnan(resid[3])
@test resid_var > 0.0
