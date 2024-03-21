```@meta
CurrentModule = DifferentiationInterface
CollapsedDocStrings = true
```

# API reference

```@docs
DifferentiationInterface
```

## Scalar to scalar

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["src/derivative.jl", "src/second_derivative.jl"]
```

## Scalar to array

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["multiderivative.jl"]
```

## Array to scalar

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["gradient.jl", "hessian.jl", "hessian_vector_product.jl"]
```

## Array to array

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["jacobian.jl"]
```

## Lower-level operators

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["pushforward.jl", "pullback.jl"]
```

## Preparation

```@autodocs
Modules = [DifferentiationInterface]
Pages = ["prepare.jl"]
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
```

## Testing

This is not part of the public API.

```@autodocs
Modules = [DifferentiationTest]
```
