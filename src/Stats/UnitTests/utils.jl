# Filter both missing values and NaN values from a vector
function _skipmissing_to_vec(x)
    floats = float.(x)
    # Filter out both missing and NaN values
    return collect(v for v in skipmissing(floats) if !isnan(v))
end
_isconstant(v::AbstractVector) = is_constant(v)


function _ols_fallback(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real})
    yv = Vector{Float64}(y)
    Xv = Matrix{Float64}(X)
    β = Xv \ yv
    fitted = Xv * β
    residuals = yv .- fitted
    n, p = size(Xv)
    df_residual = n - p
    σ2 = sum(residuals .^ 2) / df_residual
    XtX = Xv' * Xv
    cov_β = σ2 * inv(XtX)
    se = sqrt.(diag(cov_β))
    return (β=β, se=se, residuals=residuals, σ2=σ2, df_residual=df_residual)
end

function _ols(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real})
    if isdefined(@__MODULE__, :ols)
        fit = ols(y, X)
        return (β = fit.coef,
                se = fit.se,
                residuals = fit.residuals,
                σ2 = fit.sigma2,
                df_residual = fit.df_residual)
    else
        return _ols_fallback(y, X)
    end
end

# handling heteroskedasticity and autocorrelation in data
function _bartlett_LRV(res::Vector{Float64}, n::Int, lmax::Int)
    if lmax == 0
        return nothing
    end
    idx = 1:lmax
    # autocovariance sums γ̂_l = ∑_{t=l+1}^{n} res_t * res_{t-l}
    xcov = [dot(@view(res[(l+1):end]), @view(res[1:(end-l)])) for l in idx]
    bartlett = 1 .- (idx ./ (lmax + 1))
    # returns the 2/n * sum(w_l * γ̂_l)
    return (2 / n) * dot(bartlett, xcov)
end

function _pvalue_from_cvals(teststat::Float64, cvals::Vector{Float64}, probs::Vector{Float64})
    @assert length(cvals) == length(probs) && length(cvals) >= 2
    
    xs = collect(cvals)
    ys = collect(probs)
    
    if teststat <= minimum(xs)
        return ys[argmin(xs)]
    elseif teststat >= maximum(xs)
        return ys[argmax(xs)]
    end
    
    for i in 1:length(xs)-1
        x1, x2 = xs[i], xs[i+1]
        y1, y2 = ys[i], ys[i+1]
        if (x1 <= teststat <= x2) || (x2 <= teststat <= x1)
            t = (teststat - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
        end
    end
    return ys[end]  # fallback (shouldn’t hit)
end


_ic(RSS::Real, n::Int, p::Int, kpen::Real) = n * log(RSS / n) + kpen * p


function _ftest_R_vs_F(RSSr::Real, dfr::Int, RSSf::Real, dff::Int)::Float64
    q = dfr - dff
    num = (RSSr - RSSf) / q
    den = RSSf / dff
    return num / den
end