using Base: get_extension
using DifferentiationInterface
import DifferentiationInterface as DI
using Documenter
using DocumenterMermaid

using ADTypes
using Diffractor: Diffractor
using Enzyme: Enzyme
using FastDifferentiation: FastDifferentiation
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
# using Symbolics: Symbolics
using Tapir: Tapir
using Tracker: Tracker
using Zygote: Zygote

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

extensions = [
    get_extension(DI, :DifferentiationInterfaceChainRulesCoreExt),
    get_extension(DI, :DifferentiationInterfaceDiffractorExt),
    get_extension(DI, :DifferentiationInterfaceEnzymeExt),
    get_extension(DI, :DifferentiationInterfaceFastDifferentiationExt),
    get_extension(DI, :DifferentiationInterfaceFiniteDiffExt),
    get_extension(DI, :DifferentiationInterfaceFiniteDifferencesExt),
    get_extension(DI, :DifferentiationInterfaceForwardDiffExt),
    get_extension(DI, :DifferentiationInterfacePolyesterForwardDiffExt),
    get_extension(DI, :DifferentiationInterfaceReverseDiffExt),
    # get_extension(DI, :DifferentiationInterfaceSymbolicsExt),
    get_extension(DI, :DifferentiationInterfaceTapirExt),
    get_extension(DI, :DifferentiationInterfaceTrackerExt),
    get_extension(DI, :DifferentiationInterfaceZygoteExt),
]

makedocs(;
    modules=[DifferentiationInterface, extensions...],
    authors="Guillaume Dalle, Adrian Hill",
    sitename="DifferentiationInterface.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md", #
        "Start here" => ["tutorial.md", "overview.md", "backends.md"],
        "API reference" => "api.md",
        "Advanced" => ["design.md", "extensions.md"],
    ],
    pagesonly=true,
    warnonly=true,
)

deploydocs(;
    repo="github.com/gdalle/DifferentiationInterface.jl",
    devbranch="main",
    dirname="DifferentiationInterface",
    tag_prefix="DifferentiationInterface-",
    push_preview=true,
)
