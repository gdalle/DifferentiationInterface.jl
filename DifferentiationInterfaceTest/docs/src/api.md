```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# API reference

```@docs
DifferentiationInterfaceTest
```

## Entry points

```@docs
test_differentiation
benchmark_differentiation
```

## Pre-made scenario lists

```@docs
default_scenarios
sparse_scenarios
component_scenarios
gpu_scenarios
static_scenarios
```

## Scenario types

```@docs
AbstractScenario
PushforwardScenario
PullbackScenario
DerivativeScenario
GradientScenario
JacobianScenario
SecondDerivativeScenario
HVPScenario
HessianScenario
```

## Internals

This is not part of the public API.

```@autodocs
Modules = [DifferentiationInterfaceTest]
Public = false
```
