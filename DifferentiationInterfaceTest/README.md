# DifferentiationInterfaceTest

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterfaceTest/dev/)
[![Build Status](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gdalle/DifferentiationInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Testing and benchmarking utilities for automatic differentiation (AD) in Julia, based on [DifferentiationInterface](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/).

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
- Weird array types (GPU, static, components)