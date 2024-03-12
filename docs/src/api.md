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
Pages = ["derivative.jl"]
```

### Scalar to array

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["multiderivative.jl"]
```

### Array to scalar

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["gradient.jl"]
```

### Array to array

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["jacobian.jl"]
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
