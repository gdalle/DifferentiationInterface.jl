# API reference

```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

```@docs
DifferentiationInterfaceTest
```

## Entry points

```@docs
test_differentiation
benchmark_differentiation
DifferentiationBenchmarkDataRow
```

## Pre-made scenario lists

The precise contents of the scenario lists are not part of the API, only their existence.

```@docs
default_scenarios
sparse_scenarios
component_scenarios
gpu_scenarios
static_scenarios
```

## Scenario types

```@docs
Scenario
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
