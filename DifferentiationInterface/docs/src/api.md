```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# API reference

```@docs
DifferentiationInterface
```

## Derivative

```@docs
prepare_derivative
derivative
derivative!
value_and_derivative
value_and_derivative!
```

## Gradient

```@docs
prepare_gradient
gradient
gradient!
value_and_gradient
value_and_gradient!
```

## Jacobian

```@docs
prepare_jacobian
jacobian
jacobian!
value_and_jacobian
value_and_jacobian!
```

## Second order

```@docs
SecondOrder
```

```@docs
prepare_second_derivative
second_derivative
second_derivative!
```

```@docs
prepare_hvp
hvp
hvp!
```

```@docs
prepare_hessian
hessian
hessian!
```

## Primitives

```@docs
prepare_pushforward
pushforward
pushforward!
value_and_pushforward
value_and_pushforward!
```

```@docs
prepare_pullback
pullback
pullback!
value_and_pullback
value_and_pullback!
value_and_pullback_split
value_and_pullback!_split
```

## Backends

The following backend types and their documentation have been re-exported from [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

```@docs
AutoChainRules
AutoDiffractor
AutoEnzyme
AutoFastDifferentiation
AutoFiniteDiff
AutoFiniteDifferences
AutoForwardDiff
AutoPolyesterForwardDiff
AutoReverseDiff
AutoSymbolics
AutoTapir
AutoTracker
AutoZygote
```

## Backend queries

```@docs
check_available
check_twoarg
check_hessian
```

## Internals

This is not part of the public API.

```@autodocs
Modules = [DifferentiationInterface]
Public = false
```
