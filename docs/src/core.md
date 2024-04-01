```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# Core API

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

## Second order

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["second_order.jl", "second_derivative.jl", "hessian.jl", "hvp.jl"]
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

## Internals

This is not part of the public API.

```@autodocs
Modules = [DifferentiationInterface]
Public = false
Order = [:function, :type]
Filter = t -> !(t isa Type && t <: ADTypes.AbstractADType)
```
