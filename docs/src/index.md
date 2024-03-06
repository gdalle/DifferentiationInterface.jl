```@meta
CurrentModule = DifferentiationInterface
CollapsedDocStrings = true
```

# DifferentiationInterface

Documentation for [DifferentiationInterface](https://github.com/gdalle/DifferentiationInterface.jl).

This is an interface to various autodiff backends for differentiating functions of the form `f(x) = y`, where `x` and `y` are either numbers or abstract arrays.

## Terminology

|              | scalar output | vector output   |
| ------------ | ------------- | --------------- |
| scalar input | derivative    | multiderivative |
| vector input | gradient      | jacobian        |
