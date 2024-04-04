using DifferentiationInterface
using DifferentiationInterfaceTest
using Documenter

using BenchmarkTools: BenchmarkTools
using DataFrames: DataFrames
using ForwardDiff: ForwardDiff

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(;
    modules=[DifferentiationInterfaceTest],
    authors="Guillaume Dalle, Adrian Hill",
    sitename="DifferentiationInterfaceTest.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Home" => "index.md", #
        "API reference" => "api.md",
    ],
    warnonly=:missing_docs,  # missing docs for ADTypes.jl are normal
)

deploydocs(;
    repo="github.com/gdalle/DifferentiationInterface.jl",
    devbranch="main",
    dirname="DifferentiationInterfaceTest",
    tag_prefix="DIT-",
)
