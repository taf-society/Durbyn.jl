using Durbyn
using Documenter

DocMeta.setdocmeta!(Durbyn, :DocTestSetup, :(using Durbyn); recursive=true)

makedocs(;
    modules=[Durbyn],
    authors="Resul Akay <resul.akay@taf-society.org> and contributors",
    sitename="Durbyn.jl",
    format=Documenter.HTML(;
        canonical="https://akai01.github.io/Durbyn.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/akai01/Durbyn.jl",
    devbranch="main",
)
