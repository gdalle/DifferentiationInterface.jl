```@meta
CurrentModule = DifferentiationInterface
CollapsedDocStrings = true
```

# Interface

```@docs
DifferentiationInterface
```

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

## Backends

### ADTypes.jl

```@meta
CurrentModule = ADTypes
```

The following backends are defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

```@autodocs
Modules = [ADTypes]
```

Only a subset is supported by DifferentiationInterface.jl at the moment.

### DifferentiationInterface.jl

```@meta
CurrentModule = DifferentiationInterface
```

The following backends are defined by DifferentiationInterface.jl:

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["backends.jl"]
Order   = [:type]
Private = false
```

### Input / output types

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["backends.jl"]
Order   = [:function]
Private = false
```

## Internals

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["implem.jl", "mode.jl", "utils.jl", "backends.jl"]
Public = false
```
