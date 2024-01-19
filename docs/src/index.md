```@meta
CurrentModule = DifferentiationInterface
```

# DifferentiationInterface

Documentation for [DifferentiationInterface](https://github.com/gdalle/DifferentiationInterface.jl).

This is an interface to various autodiff backends for differentiating functions of the form `f(x) = y`, where `x` and `y` are either numbers or arrays.

## Public

```@autodocs
Modules = [DifferentiationInterface]
Private = false
```

## Internals

```@autodocs
Modules = [DifferentiationInterface]
Public = false
```

## Math

Some implementation reminders:

|                  | pushforward                                                      | pullback                                                           |
| ---------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| scalar -> scalar | derivative multiplied by the tangent                             | derivative multiplied by the cotangent                             |
| scalar -> vector | derivative vector multiplied componentwise by the tangent vector | dot product between the derivative vector and the cotangent vector |
| vector -> scalar | dot product between the gradient vector and the tangent vector   | gradient vector multiplied componentwise by the cotangent          |
| vector -> vector | Jacobian matrix multiplied by the tangent vector                 | transposed Jacobian matrix multiplied by the cotangent vector      |

## Index

```@index
```
