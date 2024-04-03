using Base: get_extension
using DifferentiationInterface
using DifferentiationInterfaceTest
import DifferentiationInterface as DI
using Documenter
using DocumenterMermaid
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
using Symbolics: Symbolics
using SparseDiffTools: SparseDiffTools
using Tapir: Tapir
using Tracker: Tracker
using Zygote: Zygote

cp(
    joinpath(@__DIR__, "..", "..", "README.md"),
    joinpath(@__DIR__, "src", "index.md");
    force=true,
)

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
        get_extension(DI, :DifferentiationInterfaceSparseDiffToolsExt),
        get_extension(DI, :DifferentiationInterfaceTapirExt),
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
        "Start here" => ["tutorial.md", "overview.md", "backends.md"],
        "Reference" => ["core.md", "testing.md"],
        "Advanced" => ["design.md", "extensions.md"],
    ],
    warnonly=:missing_docs,  # missing docs for ADTypes.jl are normal
)

deploydocs(; repo="github.com/gdalle/DifferentiationInterface.jl", devbranch="main")
