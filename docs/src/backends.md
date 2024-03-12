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

### ChainRulesCore

```@docs
AutoChainRules
```

### Diffractor

```@docs
AutoDiffractor
```

### Enzyme

```@docs
AutoEnzyme
```

### ForwardDiff

```@docs
AutoForwardDiff
AutoForwardDiff()
```

### FiniteDiff

```@docs
AutoFiniteDiff
```

### PolyesterForwardDiff

```@docs
AutoPolyesterForwardDiff
AutoPolyesterForwardDiff()
```

### ReverseDiff

```@docs
AutoReverseDiff
```

### Zygote

```@docs
AutoZygote
```

## Package extensions

```@meta
CurrentModule = DifferentiationInterface
```

What follows is not part of the public API.

### ChainRulesCoreExt

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceChainRulesCoreExt)]
Filter = t -> !(t <: ADTypes.AbstractADType)
```

### DiffractorExt

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceDiffractorExt)]
Filter = t -> !(t <: ADTypes.AbstractADType)
```

### EnzymeExt

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceEnzymeExt)]
Filter = t -> !(t <: ADTypes.AbstractADType)
```

### FiniteDiffExt

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceFiniteDiffExt)]
Filter = t -> !(t <: ADTypes.AbstractADType)
```

### ForwardDiffExt

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceForwardDiffExt)]
Filter = t -> !(t <: ADTypes.AbstractADType)
```

### PolyesterForwardDiffExt

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfacePolyesterForwardDiffExt)]
Filter = t -> !(t <: ADTypes.AbstractADType)
```

### ReverseDiffExt

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceReverseDiffExt)]
Filter = t -> !(t <: ADTypes.AbstractADType)
```

### ZygoteExt

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceZygoteExt)]
Filter = t -> !(t <: ADTypes.AbstractADType)
```
