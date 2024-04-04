# DifferentiationInterface

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/dev/)
[![Build Status](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

An interface to various automatic differentiation (AD) backends in Julia.

## Goal

This package provides a backend-agnostic syntax to differentiate functions of the following types:

- _allocating_: `f(x) = y`
- _mutating_: `f!(y, x) = nothing`

## Features

- First and second order operators
- In-place and out-of-place differentiation
- Preparation mechanism (e.g. to create a config or tape)
- Thorough validation on standard inputs and outputs (scalars, vectors, matrices)
- Testing and benchmarking utilities accessible to users with DifferentiationInterfaceTest.jl

## Compatibility

We support most of the backends defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

| backend                                                                         | object                                                     |
| :------------------------------------------------------------------------------ | :--------------------------------------------------------- |
| [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)             | `AutoChainRules(ruleconfig)`                               |
| [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl)                     | `AutoDiffractor()`                                         |
| [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)                              | `AutoEnzyme(Enzyme.Forward)`, `AutoEnzyme(Enzyme.Reverse)` |
| [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl)                     | `AutoFiniteDiff()`                                         |
| [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)       | `AutoFiniteDifferences(fdm)`                               |
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)                   | `AutoForwardDiff()`                                        |
| [PolyesterForwardDiff.jl](https://github.com/JuliaDiff/PolyesterForwardDiff.jl) | `AutoPolyesterForwardDiff(; chunksize)`                    |
| [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)                   | `AutoReverseDiff()`                                        |
| [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl)           | `AutoSparseForwardDiff()`, `AutoSparseFiniteDiff()`        |
| [Tracker.jl](https://github.com/FluxML/Tracker.jl)                              | `AutoTracker()`                                            |
| [Zygote.jl](https://github.com/FluxML/Zygote.jl)                                | `AutoZygote()`                                             |

We also provide some experimental backends ourselves:

| backend                                                                          | object                                                         |
| :------------------------------------------------------------------------------- | :------------------------------------------------------------- |
| [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl) | `AutoFastDifferentiation()`, `AutoSparseFastDifferentiation()` |
| [Tapir.jl](https://github.com/withbayes/Tapir.jl)                                | `AutoTapir()`                                                  |

## Example

```jldoctest readme
julia> import ForwardDiff

julia> using DifferentiationInterface

julia> backend = AutoForwardDiff();

julia> f(x) = sum(abs2, x);

julia> value_and_gradient(f, backend, [1., 2., 3.])
(14.0, [2.0, 4.0, 6.0])
```

## Related packages

- [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) is the original inspiration for DifferentiationInterface.jl.
- [AutoDiffOperators.jl](https://github.com/oschulz/AutoDiffOperators.jl) is an attempt to bridge ADTypes.jl with AbstractDifferentiation.jl.
