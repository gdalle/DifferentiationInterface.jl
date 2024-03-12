```@meta
CurrentModule = DifferentiationInterface
CollapsedDocStrings = true
```

# API reference

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

## Preparation

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["prepare.jl"]
```

## Internals

These are not part of the public API.

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["backends.jl", "mode.jl", "utils.jl"]
Public = false
```
