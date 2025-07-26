"""
    STLResult

Container for the results of an STL decomposition.  The time series
components are stored in the `time_series` field as a named tuple
with keys `:seasonal`, `:trend` and `:remainder`.  Additional
metadata such as the robustness weights, smoothing windows, local
polynomial degrees, jump parameters and iteration counts are also
stored.  This struct loosely follows the structure of the list
returned by the R `stl` function.
"""
struct STLResult{T<:Real}
    time_series::NamedTuple{(:seasonal,:trend,:remainder),Tuple{Vector{T},Vector{T},Vector{T}}}
    weights::Vector{T}
    windows::NamedTuple{(:s,:t,:l),Tuple{Int,Int,Int}}
    degrees::NamedTuple{(:s,:t,:l),Tuple{Int,Int,Int}}
    jumps::NamedTuple{(:s,:t,:l),Tuple{Int,Int,Int}}
    inner::Int
    outer::Int
end

function stlest!(y::AbstractVector{Float64}, n::Int, len::Int, ideg::Int,
                 xs::Float64, nleft::Int, nright::Int,
                 w::AbstractVector{Float64}, userw::Bool, rw::AbstractVector{Float64})
                 
    range = float(n) - 1.0
    h = max(xs - float(nleft), float(nright) - xs)
    if len > n
        h += float(len - n) / 2.0
    end
    h9 = 0.999 * h
    h1 = 0.001 * h

    asum = 0.0
    for j in nleft:nright
        r = abs(float(j) - xs)
        if r <= h9
            if r <= h1 || h == 0.0
                w[j] = 1.0
            else
                rr = r / h
                w[j] = (1.0 - rr^3)^3
            end
            if userw
                w[j] *= rw[j]
            end
            asum += w[j]
        else
            w[j] = 0.0
        end
    end

    if asum <= 0.0
        return 0.0, false
    end
    
    inva = 1.0 / asum
    for j in nleft:nright
        w[j] *= inva
    end

    if h > 0.0 && ideg > 0
        
        a_mean = 0.0
        for j in nleft:nright
            a_mean += w[j] * float(j)
        end
        b = xs - a_mean
        c = 0.0
        for j in nleft:nright
            d = float(j) - a_mean
            c += w[j] * d^2
        end
        
        if sqrt(c) > 0.001 * range
            b /= c
            for j in nleft:nright
                w[j] = w[j] * (b * (float(j) - a_mean) + 1.0)
            end
        end
    end
    
    ys = 0.0
    for j in nleft:nright
        ys += w[j] * y[j]
    end
    return ys, true
end

function stless!(y::AbstractVector{Float64}, n::Int, len::Int, ideg::Int, njump::Int,
                 userw::Bool, rw::AbstractVector{Float64},
                 ys::AbstractVector{Float64}, res::AbstractVector{Float64})
                 
    if n < 2
        
        ys[firstindex(ys)] = y[1]
        return
    end
    
    newnj = min(njump, n - 1)
    
    nleft = 1
    nright = min(len, n)

    if len >= n
        nleft = 1
        nright = n
        i = 1
        while i <= n
            xs = float(i)
            ysi, ok = stlest!(y, n, len, ideg, xs, nleft, nright, res, userw, rw)
            if ok
                ys[firstindex(ys) - 1 + i] = ysi
            else
                ys[firstindex(ys) - 1 + i] = y[i]
            end
            i += newnj
        end
    else
        if newnj == 1
            nsh = (len + 1) ÷ 2
            nleft = 1
            nright = len
            for i in 1:n
                if (i > nsh) && (nright != n)
                    nleft += 1
                    nright += 1
                end
                xs = float(i)
                ysi, ok = stlest!(y, n, len, ideg, xs, nleft, nright, res, userw, rw)
                if ok
                    ys[firstindex(ys) - 1 + i] = ysi
                else
                    ys[firstindex(ys) - 1 + i] = y[i]
                end
            end
        else
            nsh = (len + 1) ÷ 2
            i = 1
            while i <= n
                if i < nsh
                    nleft = 1
                    nright = len
                elseif i >= n - nsh + 1
                    nleft = n - len + 1
                    nright = n
                else
                    nleft = i - nsh + 1
                    nright = len + i - nsh
                end
                xs = float(i)
                ysi, ok = stlest!(y, n, len, ideg, xs, nleft, nright, res, userw, rw)
                if ok
                    ys[firstindex(ys) - 1 + i] = ysi
                else
                    ys[firstindex(ys) - 1 + i] = y[i]
                end
                i += newnj
            end
        end
    end

    if newnj != 1
        i = 1
        while i <= n - newnj
            ysi = ys[firstindex(ys) - 1 + i]
            ysj = ys[firstindex(ys) - 1 + i + newnj]
            delta = (ysj - ysi) / float(newnj)
            for j in (i + 1):(i + newnj - 1)
                ys[firstindex(ys) - 1 + j] = ysi + delta * float(j - i)
            end
            i += newnj
        end
        
        k = ((n - 1) ÷ newnj) * newnj + 1
        if k != n
            
            xs = float(n)
            ysn, ok = stlest!(y, n, len, ideg, xs, nleft, nright, res, userw, rw)
            if ok
                ys[firstindex(ys) - 1 + n] = ysn
            else
                ys[firstindex(ys) - 1 + n] = y[n]
            end
            if k != n - 1
                
                valk = ys[firstindex(ys) - 1 + k]
                valn = ys[firstindex(ys) - 1 + n]
                delta2 = (valn - valk) / float(n - k)
                for j in (k + 1):(n - 1)
                    ys[firstindex(ys) - 1 + j] = valk + delta2 * float(j - k)
                end
            end
        end
    end
    return
end

function stlma!(x::AbstractVector{Float64}, n::Int, len::Int, ave::AbstractVector{Float64})
    
    if len <= 0 || n < len
        return
    end
    newn = n - len + 1
    
    v = 0.0
    for i in 1:len
        v += x[i]
    end
    flen = float(len)
    ave[1] = v / flen
    if newn > 1
        k = len
        m = 0
        for j in 2:newn
            k += 1
            m += 1
            v = v - x[m] + x[k]
            ave[j] = v / flen
        end
    end
    return
end

function stlfts!(x::AbstractVector{Float64}, n::Int, np::Int,
                 trend::AbstractVector{Float64}, work::AbstractVector{Float64})
                 
    stlma!(x, n, np, trend)
    
    stlma!(trend, n - np + 1, np, work)
    
    stlma!(work, n - 2 * np + 2, 3, trend)
    return
end

function stlss!(y::AbstractVector{Float64}, n::Int, np::Int, ns::Int, isdeg::Int,
                nsjump::Int, userw::Bool, rw::AbstractVector{Float64},
                season_ext::AbstractVector{Float64},
                work1::AbstractVector{Float64}, work2::AbstractVector{Float64},
                work3::AbstractVector{Float64}, work4::AbstractVector{Float64})
                
    if np < 1
        return
    end
    
    for j in 1:np
        
        k = ((n - j) ÷ np) + 1
        
        for i in 1:k
            idx = (i - 1) * np + j
            work1[i] = y[idx]
        end
        
        if userw
            for i in 1:k
                idx = (i - 1) * np + j
                work3[i] = rw[idx]
            end
        end
        
        stless!(work1, k, ns, isdeg, nsjump, userw, work3, view(work2, 2:(k + 1)), work4)
        
        xs = 0.0
        nright = min(ns, k)
        yfit, ok = stlest!(work1, k, ns, isdeg, xs, 1, nright, work4, userw, work3)
        if !ok
            yfit = work2[2]
        end
        work2[1] = yfit
        
        xs = float(k + 1)
        nleft = max(1, k - ns + 1)
        yfit, ok = stlest!(work1, k, ns, isdeg, xs, nleft, k, work4, userw, work3)
        if !ok
            yfit = work2[k + 1]
        end
        work2[k + 2] = yfit
        
        for m in 1:(k + 2)
            idx = (m - 1) * np + j
            season_ext[idx] = work2[m]
        end
    end
    return
end


function stlrwt!(y::AbstractVector{Float64}, fit::AbstractVector{Float64},
                 rw::AbstractVector{Float64})
    n = min(length(y), length(fit), length(rw))
    
    tmp = Vector{Float64}(undef, n)
    for i in 1:n
        tmp[i] = abs(y[i] - fit[i])
    end
    
    mad = median(tmp)
    cmad = 6.0 * mad
    
    c9 = 0.999 * cmad
    c1 = 0.001 * cmad
    for i in 1:n
        r = abs(y[i] - fit[i])
        if r <= c1
            rw[i] = 1.0
        elseif r <= c9 && cmad > 0.0
            x = r / cmad
            rw[i] = (1.0 - x^2)^2
        else
            rw[i] = 0.0
        end
    end
    return
end

function stlstp!(y::AbstractVector{Float64}, n::Int, np::Int, ns::Int, nt::Int, nl::Int,
                 isdeg::Int, itdeg::Int, ildeg::Int,
                 nsjump::Int, ntjump::Int, nljump::Int,
                 ni::Int, userw::Bool, rw::AbstractVector{Float64},
                 season::AbstractVector{Float64}, trend::AbstractVector{Float64})
                 
    n2 = n + 2 * np
    col1 = zeros(Float64, n2)
    col2 = zeros(Float64, n2)
    col3 = zeros(Float64, n2)
    col4 = zeros(Float64, n2)
    col5 = zeros(Float64, n2)
    for _iter in 1:ni
        
        for i in 1:n
            col1[i] = y[i] - trend[i]
        end
        
        stlss!(col1, n, np, ns, isdeg, nsjump, userw, rw, col2, col3, col4, col5, season)
        stlfts!(col2, n2, np, col3, col1)
        stless!(col3, n, nl, ildeg, nljump, false, col4, col1, col5)
        for i in 1:n
            season[i] = col2[np + i] - col1[i]
        end
        for i in 1:n
            col1[i] = y[i] - season[i]
        end
        stless!(col1, n, nt, itdeg, ntjump, userw, rw, trend, col3)
    end
    return
end

function stl_base(
    y::AbstractVector{Float64},
    np::Int,
    ns::Int,
    nt::Int,
    nl::Int,
    isdeg::Int,
    itdeg::Int,
    ildeg::Int,
    nsjump::Int,
    ntjump::Int,
    nljump::Int,
    ni::Int,
    no::Int,
    rw::AbstractVector{Float64},
    season::AbstractVector{Float64},
    trend::AbstractVector{Float64},
)
    n = length(y)
    fill!(trend, 0.0)
    newns = max(3, ns)
    if newns % 2 == 0
        newns += 1
    end
    newnt = max(3, nt)
    if newnt % 2 == 0
        newnt += 1
    end
    newnl = max(3, nl)
    if newnl % 2 == 0
        newnl += 1
    end
    newnp = max(2, np)
    userw = false
    k = 0
    while true
        stlstp!(
            y,
            n,
            newnp,
            newns,
            newnt,
            newnl,
            isdeg,
            itdeg,
            ildeg,
            nsjump,
            ntjump,
            nljump,
            ni,
            userw,
            rw,
            season,
            trend,
        )
        k += 1
        if k > no
            break
        end
        fit = Vector{Float64}(undef, n)
        for i = 1:n
            fit[i] = trend[i] + season[i]
        end
        stlrwt!(y, fit, rw)
        userw = true
    end
    if no <= 0
        for i = 1:n
            rw[i] = 1.0
        end
    end
    return
end

function stl_core(
    y::AbstractVector{Float64},
    np::Int,
    ns::Int,
    nt::Int,
    nl::Int,
    isdeg::Int,
    itdeg::Int,
    ildeg::Int,
    nsjump::Int,
    ntjump::Int,
    nljump::Int,
    ni::Int,
    no::Int,
)
    n = length(y)
    season = zeros(Float64, n)
    trend = zeros(Float64, n)
    rw = zeros(Float64, n)
    stl_base(
        y,
        np,
        ns,
        nt,
        nl,
        isdeg,
        itdeg,
        ildeg,
        nsjump,
        ntjump,
        nljump,
        ni,
        no,
        rw,
        season,
        trend,
    )
    return season, trend, rw
end

function nextodd(x::Real)::Int
    cx = Int(round(x))
    return isodd(cx) ? cx : cx + 1
end

function check_degree(deg, name::AbstractString)
    d = Int(deg)
    if d < 0 || d > 1
        error("$name must be 0 or 1")
    end
    return d
end



"""
    stl(x, m; kwargs...)

High level interface for performing a seasonal-trend decomposition
based on Loess (STL) on the one-dimensional array `x`.  The
argument `m` specifies the frequency of the series (the number of
observations per cycle) and must be at least two.  The function
closely mirrors the R `stl` API and offers a range of keyword
arguments controlling the smoothing spans, polynomial degrees,
subsampling steps and robustness iterations.

Mandatory arguments:

* `x`: A numeric vector containing the time series to be decomposed.
* `m` :An integer specifying the frequency (periodicity) of the series.

Keyword arguments (defaults follow the R implementation):

* `s_window` : Span of the seasonal smoothing window.  May be an integer
  (interpreted as a span and rounded to the next odd value) or the
  string `"periodic"` to request a periodic seasonal component.
* `s_degree` : Degree of the local polynomial used for seasonal
  smoothing (0 or 1).  Defaults to 0.
* `t_window` : Span of the trend smoothing window.  If omitted, a
  default based on `m` and `s_window` is computed.  Must be odd.
* `t_degree` : Degree of the local polynomial used for trend
  smoothing (0 or 1).  Defaults to 1.
* `l_window` : Span of the low-pass filter.  Defaults to the next
  odd integer greater than or equal to `m`.
* `l_degree` : Degree of the local polynomial used for the low-pass
  filter.  Defaults to the value of `t_degree`.
* `s_jump`, `t_jump`, `l_jump` : Subsampling step sizes used when
  evaluating the loess smoother.  Defaults are one tenth of the
  corresponding window lengths (rounded up).
* `robust` : Logical flag indicating whether to compute robustness
  weights.  When true up to 15 outer iterations are performed; when
  false no robustness iterations are used.
* `inner` : Number of inner loop iterations.  Defaults to 1 when
  `robust` is true and 2 otherwise.
* `outer` : Number of outer robustness iterations.  Defaults to 15
  when `robust` is true and 0 otherwise.

The function returns an `STLResult` containing the seasonal,
trend and remainder components along with ancillary information.
"""
function stl(
    x::AbstractVector{T},
    m::Integer;
    s_window,
    s_degree::Integer = 0,
    t_window::Union{Nothing,Integer} = nothing,
    t_degree::Integer = 1,
    l_window::Union{Nothing,Integer} = nothing,
    l_degree::Integer = t_degree,
    s_jump::Union{Nothing,Integer} = nothing,
    t_jump::Union{Nothing,Integer} = nothing,
    l_jump::Union{Nothing,Integer} = nothing,
    robust::Bool = false,
    inner::Union{Nothing,Integer} = nothing,
    outer::Union{Nothing,Integer} = nothing,
) where {T<:Real}

    n = length(x)
    if m < 2 || n <= 2 * m
        error("series is not periodic or has less than two periods")
    end

    if any(ismissing, x)
        error(
            "Input data contains missing values; consider imputing or removing them before calling stl",
        )
    end

    periodic = false
    if isa(s_window, AbstractString) || isa(s_window, Symbol)

        sval = String(s_window)
        if startswith(lowercase(sval), "periodic")
            periodic = true
            s_window_val = 10 * n + 1
            s_degree = 0
        else
            error("unknown string value for s_window: $s_window")
        end
    elseif isa(s_window, Integer)
        s_window_val = nextodd(s_window)
    else
        s_window_val = nextodd(round(Int, s_window))
    end

    s_degree = check_degree(s_degree, "s_degree")
    t_degree = check_degree(t_degree, "t_degree")
    l_degree = check_degree(l_degree, "l_degree")

    if t_window === nothing
        t_window_val = nextodd(ceil(Int, 1.5 * m / (1.0 - 1.5 / s_window_val)))
    else
        t_window_val = nextodd(t_window)
    end

    if l_window === nothing
        l_window_val = nextodd(m)
    else
        l_window_val = nextodd(l_window)
    end

    if s_jump === nothing
        s_jump_val = max(1, Int(ceil(s_window_val / 10)))
    else
        s_jump_val = s_jump
    end
    if t_jump === nothing
        t_jump_val = max(1, Int(ceil(t_window_val / 10)))
    else
        t_jump_val = t_jump
    end
    if l_jump === nothing
        l_jump_val = max(1, Int(ceil(l_window_val / 10)))
    else
        l_jump_val = l_jump
    end

    if inner === nothing
        inner_val = robust ? 1 : 2
    else
        inner_val = inner
    end
    if outer === nothing
        outer_val = robust ? 15 : 0
    else
        outer_val = outer
    end

    xvec = collect(float.(x))

    season, trend, weights = stl_core(
        xvec,
        m,
        s_window_val,
        t_window_val,
        l_window_val,
        s_degree,
        t_degree,
        l_degree,
        s_jump_val,
        t_jump_val,
        l_jump_val,
        inner_val,
        outer_val,
    )
    remainder = xvec .- season .- trend

    if periodic
        cycle = [(i - 1) % m + 1 for i = 1:n]
        mean_by_cycle = zeros(Float64, m)
        counts = zeros(Int, m)
        for i = 1:n
            idx = cycle[i]
            mean_by_cycle[idx] += season[i]
            counts[idx] += 1
        end
        for j = 1:m
            if counts[j] > 0
                mean_by_cycle[j] /= counts[j]
            end
        end
        for i = 1:n
            season[i] = mean_by_cycle[cycle[i]]
        end
        remainder = xvec .- season .- trend
    end
    return STLResult{Float64}(
        (seasonal = season, trend = trend, remainder = remainder),
        weights,
        (s = s_window_val, t = t_window_val, l = l_window_val),
        (s = s_degree, t = t_degree, l = l_degree),
        (s = s_jump_val, t = t_jump_val, l = l_jump_val),
        inner_val,
        outer_val,
    )
end


"""
    Base.show(io::IO, result::STLResult)

Pretty print an `STLResult`.  The time series components are
displayed along with basic metadata.  This mimics the behaviour of
the R `print.stl` method.  The `show` function is invoked
automatically when a result is printed at the REPL.
"""
function Base.show(io::IO, result::STLResult)
    ts = result.time_series
    println(io, "STL decomposition")
    println(io, "Seasonal component (first 10 values): ", ts.seasonal[1:min(end,10)])
    println(io, "Trend component    (first 10 values): ", ts.trend[1:min(end,10)])
    println(io, "Remainder          (first 10 values): ", ts.remainder[1:min(end,10)])
    println(io, "Windows: ", result.windows)
    println(io, "Degrees: ", result.degrees)
    println(io, "Jumps: ", result.jumps)
    println(io, "Inner iterations: ", result.inner, ", Outer iterations: ", result.outer)
    return
end

"""
    summary(result::STLResult; digits=4)

Display a statistical summary of an `STLResult`.  The summary includes
simple descriptive statistics of the time series components (mean,
standard deviation, minimum, maximum and interquartile range), the
IQR expressed as a percentage of the reconstructed data, and a
summary of the robustness weights.  
"""
function summary(result::STLResult; digits::Integer=4)
    ts = result.time_series
    n = length(ts.seasonal)
    data = ts.seasonal .+ ts.trend .+ ts.remainder
    comps = Dict(
        :seasonal => ts.seasonal,
        :trend    => ts.trend,
        :remainder=> ts.remainder,
        :data     => data,
    )
    
    function iqr(v::AbstractVector)
        q25, q75 = quantile(v, [0.25, 0.75])
        return q75 - q25
    end
    println("STL decomposition summary")
    println("Time series components:")
    for (name, vec) in comps
        println("  ", name)
        mv = mean(vec)
        sv = std(vec)
        mn = minimum(vec)
        mx = maximum(vec)
        iqr_v = iqr(vec)
        
        component_fmt = string("% .", digits, "f")
        full_fmt = "    mean=" * component_fmt * "  sd=" * component_fmt *
                   "  min=" * component_fmt * "  max=" * component_fmt *
                   "  IQR=" * component_fmt
        f = Printf.Format(full_fmt)
        println(Printf.format(f, mv, sv, mn, mx, iqr_v))
    end
    println("IQR as percentage of total:")
    iqr_vals = Dict(name => iqr(vec) for (name, vec) in comps)
    total_iqr = iqr_vals[:data]
    for (name, v) in iqr_vals
        pct = total_iqr == 0 ? NaN : 100.0 * v / total_iqr
        pct_str = isnan(pct) ? "NaN" : string(round(pct; digits=1))
        println("  ", Symbol(name), ": ", pct_str, "%")
    end
    
    if all(w -> w == 1.0, result.weights)
        println("Weights: all equal to 1")
    else
        w = result.weights
        mv = mean(w)
        sv = std(w)
        mn = minimum(w)
        mx = maximum(w)
        iqr_w = iqr(w)
        println("Weights summary:")
        s_mean = string(round(mv; digits=digits))
        s_sd   = string(round(sv; digits=digits))
        s_min  = string(round(mn; digits=digits))
        s_max  = string(round(mx; digits=digits))
        s_iqr  = string(round(iqr_w; digits=digits))
        println("  mean=", s_mean, "  sd=", s_sd,
                "  min=", s_min, "  max=", s_max, "  IQR=", s_iqr)
    end
    println("Other components: windows=", result.windows, ", degrees=", result.degrees, ", jumps=", result.jumps,
            ", inner=", result.inner, ", outer=", result.outer)
    return nothing
end

"""
    plot(result::STLResult; labels, col_range="lightgray", main=nothing, range_bars=true, kwargs...)

Create a multi-panel plot of an `STLResult`.  The first panel
shows the reconstructed data (seasonal + trend + remainder), and
the subsequent panels show the seasonal, trend and remainder
components respectively.  You can supply your own `labels` (an
array of four strings) to label each panel.  The keyword
`range_bars` controls whether small range bars are drawn to the
right of each subplot indicating the relative scale of the series.
Additional keyword arguments are passed through to the individual
plot calls.

This function assumes the Plots.jl package is available.  If you
already have your own base plotting function, you can overload
`plot(::STLResult)` in your own code to forward to that implementation.
"""
function plot(result::STLResult; labels::Vector{String}=["data","seasonal","trend","remainder"],
              col_range::Any="lightgray", main::Union{Nothing,String}=nothing, range_bars::Bool=true,
              kwargs...)
    ts = result.time_series
    n = length(ts.seasonal)
    data = ts.seasonal .+ ts.trend .+ ts.remainder
    series = [data, ts.seasonal, ts.trend, ts.remainder]
    nplot = length(series)
    
    rx = [extrema(s) for s in series]
    rng = [r[2] - r[1] for r in rx]
    mx = range_bars ? minimum(rng) : 0.0
    
    plt = Plots.plot(layout=(nplot, 1), legend=false, kwargs...)
    for i in 1:nplot
        ptype = i < nplot ? :line : :sticks
        Plots.plot!(plt[i], series[i], color=:blue, ylabel=labels[i], seriestype=ptype, kwargs...)
        if range_bars
            
            yrange = rx[i]
            ymid = sum(yrange) / 2
            barhalf = mx / 2
            
            dx = 0.02 * n
            xb = [n - dx, n - dx, n - 0.4 * dx, n - 0.4 * dx]
            yb = [ymid - barhalf, ymid + barhalf, ymid + barhalf, ymid - barhalf]
            Plots.plot!(plt[i], xb, yb, fill=(col_range, 0.5), linecolor=:transparent)
        end
        if i == 1 && main !== nothing
            Plots.plot!(plt[i], title=main)
        end
        if i == nplot
            Plots.hline!(plt[i], [0.0], color=:black, linestyle=:dash)
        end
        
        if i < nplot
            
            Plots.plot!(plt[i], xticks=:none)
        end
    end

    Plots.xlabel!(plt[nplot], "index")
    Plots.display(plt)
    return plt
end