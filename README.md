# DifferentiationInterface

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/dev/)
[![Build Status](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gdalle/DifferentiationInterface.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/gdalle/DifferentiationInterface.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

An interface to various automatic differentiation backends in Julia.

## Goal

This package provides a backend-agnostic syntax to differentiate functions of two types:

- `f(x) = y` where `x` and `y` are either real numbers or abstract arrays
- `f!(y, x) = nothing` where `y` is an abstract array and `x` can be a real number or an abstract array

It supports in-place versions of every operator, and ensures type stability whenever possible.

## Compatibility

We support some of the backends defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

| Backend                                                                         | Type                                                       |
| :------------------------------------------------------------------------------ | :--------------------------------------------------------- |
| [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)             | `AutoChainRules(ruleconfig)`                               |
| [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl)                     | `AutoDiffractor()`                                         |
| [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)                              | `AutoEnzyme(Val(:forward))` or `AutoEnzyme(Val(:reverse))` |
| [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl)                     | `AutoFiniteDiff()`                                         |
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)                   | `AutoForwardDiff()`                                        |
| [PolyesterForwardDiff.jl](https://github.com/JuliaDiff/PolyesterForwardDiff.jl) | `AutoPolyesterForwardDiff(; chunksize=C)`                  |
| [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)                   | `AutoReverseDiff()`                                        |
| [Zygote.jl](https://github.com/FluxML/Zygote.jl)                                | `AutoZygote()`                                             |

## Example

Setup:

```jldoctest readme
julia> import DifferentiationInterface, ADTypes, ForwardDiff

julia> backend = ADTypes.AutoForwardDiff();

julia> f(x) = sum(abs2, x);
```

Out-of-place gradient:

```jldoctest readme
julia> DifferentiationInterface.value_and_gradient(backend, f, [1., 2., 3.])
(14.0, [2.0, 4.0, 6.0])
```

In-place gradient:

```jldoctest readme
julia> grad = zeros(3);

julia> DifferentiationInterface.value_and_gradient!(grad, backend, f, [1., 2., 3.])
(14.0, [2.0, 4.0, 6.0])

julia> grad
3-element Vector{Float64}:
 2.0
 4.0
 6.0
```

## Related packages

- [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) is the original inspiration for DifferentiationInterface.jl. We aim to be less generic (one input, one output, first order only) but more efficient (type stability, memory reuse).
- [AutoDiffOperators.jl](https://github.com/oschulz/AutoDiffOperators.jl) is an attempt to bridge ADTypes.jl with AbstractDifferentiation.jl. We provide similar functionality (except for the matrix-like behavior) but cover more backends.

## Roadmap

Goals for future releases:

- implement backend-specific cache objects
- support in-place functions `f!(y, x)`
- define user-facing functions to test and benchmark backends against each other
