using Base: get_extension
using DifferentiationInterface
import DifferentiationInterface as DI
using Documenter

using DiffResults: DiffResults
using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote

ChainRulesCoreExt = get_extension(DI, :DifferentiationInterfaceChainRulesCoreExt)
EnzymeExt = get_extension(DI, :DifferentiationInterfaceEnzymeExt)
FiniteDiffExt = get_extension(DI, :DifferentiationInterfaceFiniteDiffExt)
ForwardDiffExt = get_extension(DI, :DifferentiationInterfaceForwardDiffExt)
PolyesterForwardDiffExt = get_extension(
    DI, :DifferentiationInterfacePolyesterForwardDiffExt
)
ReverseDiffExt = get_extension(DI, :DifferentiationInterfaceReverseDiffExt)
ZygoteExt = get_extension(DI, :DifferentiationInterfaceZygoteExt)

DocMeta.setdocmeta!(
    DifferentiationInterface,
    :DocTestSetup,
    :(using DifferentiationInterface);
    recursive=true,
)

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
        DifferentiationInterface,
        ChainRulesCoreExt,
        EnzymeExt,
        FiniteDiffExt,
        ForwardDiffExt,
        PolyesterForwardDiffExt,
        ReverseDiffExt,
        ZygoteExt,
    ],
    authors="Guillaume Dalle, Adrian Hill",
    sitename="DifferentiationInterface.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/DifferentiationInterface.jl",
    ),
    pages=[
        "Home" => "index.md", "Interface" => "interface.md", "Backends" => "backends.md"
    ],
)

deploydocs(; repo="github.com/gdalle/DifferentiationInterface.jl", devbranch="main")
