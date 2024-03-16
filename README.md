# DifferentiationInterface

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/dev/)
[![Build Status](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gdalle/DifferentiationInterface.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/gdalle/DifferentiationInterface.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

An interface to various automatic differentiation backends in Julia.

## Goal

This package provides a backend-agnostic syntax to differentiate functions of the following types:

- **Allocating**: `f(x) = y` where `x` and `y` can be real numbers or abstract arrays
- **Mutating**: `f!(y, x) = nothing` where `y` is an abstract array and `x` can be a real number or an abstract array

## Compatibility

We support some of the first order backends defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

| Backend                                                                         | Object                                                       |
| :------------------------------------------------------------------------------ | :----------------------------------------------------------- |
| [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)             | `AutoChainRules(ruleconfig)`                                 |
| [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl)                     | `AutoDiffractor()`                                           |
| [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)                              | `AutoEnzyme(Enzyme.Forward)` or `AutoEnzyme(Enzyme.Reverse)` |
| [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl)                     | `AutoFiniteDiff()`                                           |
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)                   | `AutoForwardDiff()`                                          |
| [PolyesterForwardDiff.jl](https://github.com/JuliaDiff/PolyesterForwardDiff.jl) | `AutoPolyesterForwardDiff(; chunksize)`                      |
| [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)                   | `AutoReverseDiff()`                                          |
| [Zygote.jl](https://github.com/FluxML/Zygote.jl)                                | `AutoZygote()`                                               |

We also provide a second order backend `SecondOrder(reverse_backend, forward_backend)` for hessian computations.

## Example

Setup:

```jldoctest readme
julia> import ADTypes, ForwardDiff

julia> using DifferentiationInterface

julia> backend = ADTypes.AutoForwardDiff();

julia> f(x) = sum(abs2, x);
```

Out-of-place gradient:

```jldoctest readme
julia> value_and_gradient(backend, f, [1., 2., 3.])
(14.0, [2.0, 4.0, 6.0])
```

In-place gradient:

```jldoctest readme
julia> grad = zeros(3);

julia> value_and_gradient!(grad, backend, f, [1., 2., 3.])
(14.0, [2.0, 4.0, 6.0])

julia> grad
3-element Vector{Float64}:
 2.0
 4.0
 6.0
```

## Related packages

- [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) is the original inspiration for DifferentiationInterface.jl.
- [AutoDiffOperators.jl](https://github.com/oschulz/AutoDiffOperators.jl) is an attempt to bridge ADTypes.jl with AbstractDifferentiation.jl.

## Roadmap

Goals for future releases:

- optimize performance for each backend
- define user-facing functions to test and benchmark backends against each other
