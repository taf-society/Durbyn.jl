using Pkg
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using DocumenterMathJax
using Durbyn

const ON_CI = get(ENV, "CI", "false") == "true"

makedocs(
    sitename = "Durbyn.jl",
    modules  = [Durbyn],
    format   = Documenter.HTML(
        prettyurls = ON_CI,
        assets     = ["assets/theme.css"],
        mathengine = MathJax3(),
    ),
    pages    = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "User Guide" => Any[
            "Exponential Smoothing" => "expsmoothing.md",
            "Intermittent Demand"   => "intermittent.md",
            "ARIMA"                  => "arima.md",
            "ARAR/ARARMA"            => "ararma.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :none,
)

deploydocs(
    repo      = "github.com/taf-society/Durbyn.jl",
    devbranch = "main",
)

