```@meta
CurrentModule = DifferentiationInterface
CollapsedDocStrings = true
```

# API reference

```@docs
DifferentiationInterface
```

## Derivative

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["src/derivative.jl"]
```

## Gradient

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["gradient.jl"]
```

## Jacobian

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["jacobian.jl"]
```

## Primitives

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["pushforward.jl", "pullback.jl"]
```

## Backend queries

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["backends.jl"]
```

## Testing & benchmarking

```@autodocs
Modules = [DifferentiationTest]
Private = false
```

## Internals

This is not part of the public API.

```@autodocs
Modules = [DifferentiationInterface]
Public = false
Order = [:function, :type]
```

```@autodocs
Modules = [DifferentiationTest]
Public = false
```
