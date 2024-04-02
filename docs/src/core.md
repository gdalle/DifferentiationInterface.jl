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
Private = false
```

## Gradient

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["gradient.jl"]
Private = false
```

## Jacobian

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["jacobian.jl"]
Private = false
```

## Second order

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["second_order.jl", "second_derivative.jl", "hessian.jl", "hvp.jl"]
Private = false
```

## Primitives

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["pushforward.jl", "pullback.jl"]
Private = false
```

## Backend queries

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["backends.jl"]
Private = false
```

## Internals

This is not part of the public API.

```@autodocs
Modules = [DifferentiationInterface]
Public = false
```
