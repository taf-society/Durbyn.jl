using Pkg
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




# using Durbyn
# using Documenter

# DocMeta.setdocmeta!(Durbyn, :DocTestSetup, :(using Durbyn); recursive = true)

# makedocs(;
#     sitename = "Durbyn.jl",
#     modules = [Durbyn],
#     authors = "Resul Akay <resul.akay@taf-society.org> and contributors",
#     clean = true,
#     format = Documenter.HTML(;
#         canonical = "https://taf-society.github.io/Durbyn.jl/",
#         assets = String[joinpath(@__DIR__, "src", "assets", "theme.css")],
#         footer = "<div class=\"footer\">Made with ❤ using Documenter.jl • © 2025 Time Series Analysis and Forecasting Society</div>",
#     ),
#     pages = [
#         "Home" => "index.md",
#         "Quick Start" => "quickstart.md",
#         "User Guide" => [
#             "Exponential Smoothing" => "expsmoothing.md",
#             "Intermittent Demand"   => "intermittent.md",
#             "ARIMA"                  => "arima.md",
#             "ARAR/ARARMA"            => "ararma.md",
#         ],
#         "API Reference" => "api.md",
#     ],
# )

# deploydocs(
#     repo = "github.com/taf-society/Durbyn.jl",
#     devbranch = "main",
#     push_preview = true,
# )
