```@meta
CollapsedDocStrings = true
```

# Backends

## Types

```@meta
CurrentModule = ADTypes
```

The possible backend choices are defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

!!! warning
    Only the backends listed here are supported by DifferentiationInterface.jl, even though ADTypes.jl defines more.

```@docs
AutoChainRules
AutoDiffractor
AutoEnzyme
AutoForwardDiff
AutoForwardDiff()
AutoFiniteDiff
AutoPolyesterForwardDiff
AutoPolyesterForwardDiff()
AutoReverseDiff
AutoZygote
```

## Package extensions

```@meta
CurrentModule = DifferentiationInterface
```

Backend-specific extension content is not part of the public API.

```@autodocs
Modules = [
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceChainRulesCoreExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceDiffractorExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceEnzymeExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceFiniteDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceForwardDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfacePolyesterForwardDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceReverseDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceZygoteExt)
]
Filter = t -> !(t <: ADTypes.AbstractADType)
```
