using DifferentiationInterface
using Documenter

DocMeta.setdocmeta!(DifferentiationInterface, :DocTestSetup, :(using DifferentiationInterface); recursive=true)

makedocs(;
    modules=[DifferentiationInterface],
    authors="Guillaume Dalle, Adrian Hill",
    repo="https://github.com/gdalle/DifferentiationInterface.jl/blob/{commit}{path}#{line}",
    sitename="DifferentiationInterface.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/DifferentiationInterface.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gdalle/DifferentiationInterface.jl",
    devbranch="main",
)
