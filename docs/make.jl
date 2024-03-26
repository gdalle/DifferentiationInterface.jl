using Base: get_extension
using DifferentiationInterface
using DifferentiationInterfaceTest
import DifferentiationInterface as DI
using Documenter
using DocumenterMermaid
using JET
using Random
using Test

using ADTypes
using Chairmarks: Chairmarks
using DataFrames: DataFrames
using Diffractor: Diffractor
using Enzyme: Enzyme
using FastDifferentiation: FastDifferentiation
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Tracker: Tracker
using Zygote: Zygote

open(joinpath(@__DIR__, "src", "index.md"), "w") do io
    println(
        io,
        """
        ```@meta
        EditURL = "https://github.com/gdalle/DifferentiationInterface.jl/blob/main/README.md"
        ```
        """,
    )
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

makedocs(;
    modules=[
        ADTypes,
        DifferentiationInterface,
        DifferentiationInterfaceTest,
        get_extension(DI, :DifferentiationInterfaceChainRulesCoreExt),
        get_extension(DI, :DifferentiationInterfaceDiffractorExt),
        get_extension(DI, :DifferentiationInterfaceEnzymeExt),
        get_extension(DI, :DifferentiationInterfaceFastDifferentiationExt),
        get_extension(DI, :DifferentiationInterfaceFiniteDiffExt),
        get_extension(DI, :DifferentiationInterfaceFiniteDifferencesExt),
        get_extension(DI, :DifferentiationInterfaceForwardDiffExt),
        get_extension(DI, :DifferentiationInterfacePolyesterForwardDiffExt),
        get_extension(DI, :DifferentiationInterfaceReverseDiffExt),
        get_extension(DI, :DifferentiationInterfaceTrackerExt),
        get_extension(DI, :DifferentiationInterfaceZygoteExt),
    ],
    authors="Guillaume Dalle, Adrian Hill",
    sitename="DifferentiationInterface.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/DifferentiationInterface.jl",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md", #
        "tutorial.md", #
        "overview.md", #
        "api.md", #
        "backends.md", #
        "developer.md",
    ],
    warnonly=:missing_docs,  # missing docs for ADTypes.jl are normal
)

deploydocs(; repo="github.com/gdalle/DifferentiationInterface.jl", devbranch="main")
