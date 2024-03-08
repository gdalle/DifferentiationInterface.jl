```@meta
EditURL = "https://github.com/gdalle/DifferentiationInterface.jl/blob/main/README.md"
```

# DifferentiationInterface

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/dev/)
[![Build Status](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gdalle/DifferentiationInterface.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/gdalle/DifferentiationInterface.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

An interface to various automatic differentiation backends in Julia.

## Goal

This package provides a backend-agnostic syntax to differentiate functions `f(x) = y`, where `x` and `y` are either numbers or abstract arrays.

It started out as an experimental redesign for [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl).

## Supported backends

We support some of the backends defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) with `AutoEnzyme(Val(:forward))`
- [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) with `AutoFiniteDiff()`
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) with `AutoForwardDiff()`
- [PolyesterForwardDiff.jl](https://github.com/JuliaDiff/PolyesterForwardDiff.jl) with `AutoPolyesterForwardDiff(; chunksize=C)`
- [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) with `AutoReverseDiff()`
- [Zygote.jl](https://github.com/FluxML/Zygote.jl) with `AutoZygote()`

We also support one more backend which is not yet part of ADTypes.jl (see [ADTypes.jl#21](https://github.com/SciML/ADTypes.jl/pull/21)):

- [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) with `AutoChainRules(ruleconfig)`

## Design

Each backend must implement only one primitive:

- forward mode: the pushforward, computing a Jacobian-vector product
- reverse mode: the pullback, computing a vector-Jacobian product

From these primitives, several utilities are defined, depending on the type of the input and output:

|              | scalar output | array output    |
| ------------ | ------------- | --------------- |
| scalar input | derivative    | multiderivative |
| array input  | gradient      | jacobian        |

## Example

```jldoctest
julia> import DifferentiationInterface, ADTypes, ForwardDiff

julia> backend = ADTypes.AutoForwardDiff();

julia> f(x) = sum(abs2, x);

julia> DifferentiationInterface.value_and_gradient(backend, f, [1., 2., 3.])
(14.0, [2.0, 4.0, 6.0])
```
