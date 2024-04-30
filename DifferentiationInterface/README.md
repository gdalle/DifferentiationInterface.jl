![DifferentiationInterface Logo](https://raw.githubusercontent.com/gdalle/DifferentiationInterface.jl/main/DifferentiationInterface/docs/src/assets/logo.svg)

# DifferentiationInterface

[![Build Status](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gdalle/DifferentiationInterface.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gdalle/DifferentiationInterface.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

|           Package            |                                                                                                                                                    Docs                                                                                                                                                    |
| :--------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   DifferentiationInterface   |   [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/stable/)     [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/dev/)   |
| DifferentiationInterfaceTest | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterfaceTest/dev/) |

An interface to various automatic differentiation (AD) backends in Julia.

## Goal

This package provides a backend-agnostic syntax to differentiate functions of the following types:

- _one-argument functions_ (allocating): `f(x) = y`
- _two-argument functions_ (mutating): `f!(y, x) = nothing`

## Features

- First- and second-order operators (gradients, Jacobians, Hessians and more)
- In-place and out-of-place differentiation
- Preparation mechanism (e.g. to create a config or tape)
- Built-in sparsity handling
- Thorough validation on standard inputs and outputs (numbers, vectors, matrices)
- Testing and benchmarking utilities accessible to users with [DifferentiationInterfaceTest](https://github.com/gdalle/DifferentiationInterface.jl/tree/main/DifferentiationInterfaceTest)

## Compatibility

We support all of the backends defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl) v1.0:

- [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)
- [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl)
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)
- [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl)
- [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl)
- [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- [PolyesterForwardDiff.jl](https://github.com/JuliaDiff/PolyesterForwardDiff.jl)
- [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)
- [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)
- [Tapir.jl](https://github.com/withbayes/Tapir.jl)
- [Tracker.jl](https://github.com/FluxML/Tracker.jl)
- [Zygote.jl](https://github.com/FluxML/Zygote.jl)

## Installation

To install the stable version of the package, run the following code in a Julia REPL:

```julia
julia> using Pkg

julia> Pkg.add("DifferentiationInterface")
```

To install the development version, run this instead:

```julia
julia> using Pkg

julia> Pkg.add(
        url="https://github.com/gdalle/DifferentiationInterface.jl",
        subdir="DifferentiationInterface"
    )
```

## Example

```julia
using DifferentiationInterface
import ForwardDiff, Enzyme, Zygote  # AD backends you want to use 

f(x) = sum(abs2, x)

x = [1.0, 2.0]

value_and_gradient(f, AutoForwardDiff(), x) # returns (5.0, [2.0, 4.0]) with ForwardDiff.jl
value_and_gradient(f, AutoEnzyme(),      x) # returns (5.0, [2.0, 4.0]) with Enzyme.jl
value_and_gradient(f, AutoZygote(),      x) # returns (5.0, [2.0, 4.0]) with Zygote.jl
```

For more performance, take a look at the [DifferentiationInterface tutorial](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/stable/tutorial/).

## Related packages

- [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) is the original inspiration for DifferentiationInterface.jl.
- [AutoDiffOperators.jl](https://github.com/oschulz/AutoDiffOperators.jl) is an attempt to bridge ADTypes.jl with AbstractDifferentiation.jl.
