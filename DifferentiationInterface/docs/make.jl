using Base: get_extension
using DifferentiationInterface
import DifferentiationInterface as DI
using Documenter
using DocumenterMermaid

using ADTypes: ADTypes
# using Diffractor: Diffractor
using Enzyme: Enzyme
# using FastDifferentiation: FastDifferentiation
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff
# using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Symbolics: Symbolics
# using Tapir: Tapir
using Tracker: Tracker
using Zygote: Zygote

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(;
    modules=[DifferentiationInterface, ADTypes],
    authors="Guillaume Dalle, Adrian Hill",
    sitename="DifferentiationInterface.jl",
    format=Documenter.HTML(; assets=["assets/favicon.ico"]),
    pages=[
        "Home" => "index.md",
        "Start here" => ["tutorial.md", "overview.md", "backends.md"],
        "API reference" => "api.md",
        "Advanced" => ["design.md", "overloads.md"],
    ],
    checkdocs=:exports,
    warnonly=[:missing_docs, :cross_references],
)

deploydocs(;
    repo="github.com/gdalle/DifferentiationInterface.jl",
    devbranch="main",
    dirname="DifferentiationInterface",
    tag_prefix="DifferentiationInterface-",
    push_preview=true,
)
