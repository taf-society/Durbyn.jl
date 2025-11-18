using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using Durbyn

const ON_CI = get(ENV, "CI", "false") == "true"

makedocs(
    sitename = "Durbyn.jl",
    modules  = [Durbyn],
    format   = Documenter.HTML(
        prettyurls = ON_CI,
        assets     = ["assets/theme.css"],
        mathengine = Documenter.MathJax3(),   # if does not work try with Documenter.KaTeX()
    ),
    pages    = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "User Guide" => Any[
            "Grammar"                => "grammar.md",
            "Exponential Smoothing" => "expsmoothing.md",
            "BATS"                   => "bats.md",
            "Intermittent Demand"   => "intermittent.md",
            "ARIMA"                  => "arima.md",
            "ARAR/ARARMA"            => "ararma.md",
            "Table Operations"       => "tableops.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :none,
)

deploydocs(
    repo      = "github.com/taf-society/Durbyn.jl",
    devbranch = "main",
)
