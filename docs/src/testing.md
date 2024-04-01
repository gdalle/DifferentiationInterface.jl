```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# Testing API

```@docs
DifferentiationInterfaceTest
```

## Entry points

```@docs
test_differentiation
benchmark_differentiation
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

## Scenario lists

```@docs
default_scenarios
component_scenarios
gpu_scenarios
static_scenarios
```

## Internals

This is not part of the public API.

```@autodocs
Modules = [DifferentiationInterfaceTest]
Public = false
```
