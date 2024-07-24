# DifferentiationInterfaceTest

[![Build Status](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gdalle/DifferentiationInterface.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gdalle/DifferentiationInterface.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![DOI](https://zenodo.org/badge/740973714.svg)](https://zenodo.org/doi/10.5281/zenodo.11092033)

|           Package            |                                                                                                                                                    Docs                                                                                                                                                    |
| :--------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   DifferentiationInterface   |   [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/stable/)     [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/dev/)   |
| DifferentiationInterfaceTest | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterfaceTest/dev/) |

Testing and benchmarking utilities for automatic differentiation (AD) in Julia, based on [DifferentiationInterface](https://github.com/gdalle/DifferentiationInterface.jl/tree/main/DifferentiationInterface).

## Goal

Make it easy to know, for a given function:

- which AD backends can differentiate it
- how fast they can do it

## Features

- Predefined or custom test scenarios
- Correctness tests
- Type stability tests
- Count calls to the function
- Benchmark runtime and allocations
- Scenarios with weird array types (GPU, static, components) in package extensions

## Installation

To install the stable version of the package, run the following code in a Julia REPL:

```julia
using Pkg

Pkg.add("DifferentiationInterfaceTest")
```

To install the development version, run this instead:

```julia
using Pkg

Pkg.add(
    url="https://github.com/gdalle/DifferentiationInterface.jl",
    subdir="DifferentiationInterface"
)
    
Pkg.add(
    url="https://github.com/gdalle/DifferentiationInterface.jl",
    subdir="DifferentiationInterfaceTest"
)
```
