"""
    SymbolicsSparsityDetector <: ADTypes.AbstractSparsityDetector

Sparsity detection algorithm based on the [Symbolics.jl tracing system](https://symbolics.juliasymbolics.org/stable/manual/sparsity_detection/).

Compatible with the [ADTypes.jl sparsity detection framework](https://sciml.github.io/ADTypes.jl/stable/#Sparsity-detector).

!!! danger
    This functionality is in a package extension, and requires [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) to be loaded.

# Implements

- [`ADTypes.jacobian_sparsity`](@extref ADTypes)
- [`ADTypes.hessian_sparsity`](@extref ADTypes)

# Reference

> [Sparsity Programming: Automated Sparsity-Aware Optimizations in Differentiable Programming](https://openreview.net/forum?id=rJlPdcY38B), Gowda et al. (2019)
"""
struct SymbolicsSparsityDetector <: ADTypes.AbstractSparsityDetector end
