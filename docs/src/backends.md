```@meta
CollapsedDocStrings = true
```

# Backends

## Types

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

## [Mutation compatibility](@id mutcompat)

All backends are compatible with allocating functions `f(x) = y`. Only some are compatible with mutating functions `f!(y, x) = nothing`:

| Backend                                 | Mutating functions |
| :-------------------------------------- | ------------------ |
| `AutoChainRules(ruleconfig)`            | ✗                  |
| `AutoDiffractor()`                      | ✗                  |
| `AutoEnzyme(Enzyme.Forward)`            | ✓                  |
| `AutoEnzyme(Enzyme.Reverse)`            | ✓                  |
| `AutoFiniteDiff()`                      | soon               |
| `AutoForwardDiff()`                     | ✓                  |
| `AutoPolyesterForwardDiff(; chunksize)` | ✓                  |
| `AutoReverseDiff()`                     | ✓                  |
| `AutoZygote()`                          | ✗                  |

## [Second order combinations](@id secondcombin)

For hessian computations, in theory we can combine any pair of backends into a [`SecondOrder`](@ref).
In practice, many combinations will fail.
Here are the ones we tested for you:

| Reverse backend     | Forward backend              | Hessian tested |
| :------------------ | :--------------------------- | -------------- |
| `AutoZygote()`      | `AutoForwardDiff()`          | ✓              |
| `AutoReverseDiff()` | `AutoForwardDiff()`          | ✓              |
| `AutoZygote()`      | `AutoEnzyme(Enzyme.Forward)` | ✓              |

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
