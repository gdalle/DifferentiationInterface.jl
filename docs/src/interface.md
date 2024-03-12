```@meta
CurrentModule = DifferentiationInterface
CollapsedDocStrings = true
```

# Interface

```@docs
DifferentiationInterface
```

In every function below, the `extras` argument is meant to contain a backend-specific cache for optimal memory reuse.
This will be implemented in a future release.

## Utilities

### Scalar to scalar

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["scalar_scalar.jl"]
```

### Scalar to array

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["scalar_array.jl"]
```

### Array to scalar

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["array_scalar.jl"]
```

### Array to array

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["array_array.jl"]
```

## Primitives

### Pushforward

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["pushforward.jl"]
```

### Pullback

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["pullback.jl"]
```

## Internals

These are not part of the public API.

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["backends.jl", "implem.jl", "mode.jl", "utils.jl"]
Public = false
```

## Package extensions

These are not part of the public API.

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceChainRulesCoreExt)]
```

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceDiffractorExt)]
```

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceEnzymeExt)]
```

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceFiniteDiffExt)]
```

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceForwardDiffExt)]
```

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfacePolyesterForwardDiffExt)]
```

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceReverseDiffExt)]
```

```@autodocs
Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceZygoteExt)]
```
