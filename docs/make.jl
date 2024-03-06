using DifferentiationInterface
import DifferentiationInterface as DI
using Documenter

using Diffractor: Diffractor
using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote

DIChainRulesCoreExt = Base.get_extension(DI, :DifferentiationInterfaceChainRulesCoreExt)
DIEnzymeExt = Base.get_extension(DI, :DifferentiationInterfaceEnzymeExt)
DIFiniteDiffExt = Base.get_extension(DI, :DifferentiationInterfaceFiniteDiffExt)
DIForwardDiffExt = Base.get_extension(DI, :DifferentiationInterfaceForwardDiffExt)
DIReverseDiffExt = Base.get_extension(DI, :DifferentiationInterfaceReverseDiffExt)
DIZygoteExt = Base.get_extension(DI, :DifferentiationInterfaceZygoteExt)

DocMeta.setdocmeta!(
    DifferentiationInterface,
    :DocTestSetup,
    :(using DifferentiationInterface);
    recursive=true,
)

makedocs(;
    modules=[
        DifferentiationInterface,
        DIChainRulesCoreExt,
        DIEnzymeExt,
        DIFiniteDiffExt,
        DIForwardDiffExt,
        DIReverseDiffExt,
        DIZygoteExt,
    ],
    authors="Guillaume Dalle, Adrian Hill",
    sitename="DifferentiationInterface.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/DifferentiationInterface.jl",
    ),
    pages=[
        "Home" => "index.md", "API reference" => "api.md", "Extensions" => "extensions.md"
    ],
    warnonly=true,
)

deploydocs(; repo="github.com/gdalle/DifferentiationInterface.jl", devbranch="main")
