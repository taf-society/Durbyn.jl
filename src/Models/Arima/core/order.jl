arma_vector(order::SARIMAOrder) = [order.p, order.q, order.P, order.Q, order.s, order.d, order.D]

# --- Parameter-layout helpers (index into the flat ARMA+xreg coefficient vector) ---

"""Number of ARMA parameters: p + q + P + Q."""
n_arma_params(o::SARIMAOrder) = o.p + o.q + o.P + o.Q

"""Indices of non-seasonal AR parameters in the coefficient vector."""
ar_indices(o::SARIMAOrder) = 1:o.p

"""Indices of non-seasonal MA parameters in the coefficient vector."""
ma_indices(o::SARIMAOrder) = (o.p + 1):(o.p + o.q)

"""Indices of seasonal AR parameters in the coefficient vector."""
sar_indices(o::SARIMAOrder) = (o.p + o.q + 1):(o.p + o.q + o.P)

"""Indices of seasonal MA parameters in the coefficient vector."""
sma_indices(o::SARIMAOrder) = (o.p + o.q + o.P + 1):(o.p + o.q + o.P + o.Q)

"""Indices of exogenous regressor coefficients in the coefficient vector."""
xreg_indices(o::SARIMAOrder, n_xreg::Int) = (n_arma_params(o) + 1):(n_arma_params(o) + n_xreg)

"""Number of conditioning observations for CSS estimation."""
css_conditioning(o::SARIMAOrder) = o.d + o.D * o.s + o.p + o.P * o.s
