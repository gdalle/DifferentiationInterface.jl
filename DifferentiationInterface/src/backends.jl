
"""
    AutoFastDifferentiation

Chooses [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl).
"""
struct AutoFastDifferentiation <: AbstractSymbolicDifferentiationMode end

"""
    AutoSparseFastDifferentiation

Chooses [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl) leveraging sparsity.
"""
struct AutoSparseFastDifferentiation <: AbstractSymbolicDifferentiationMode end

"""
    AutoSymbolics

Chooses [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl).
"""
struct AutoSymbolics <: AbstractSymbolicDifferentiationMode end

"""
    AutoSparseSymbolics

Chooses [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) leveraging sparsity.
"""
struct AutoSparseSymbolics <: AbstractSymbolicDifferentiationMode end

"""
    AutoTapir

Chooses [Tapir.jl](https://github.com/withbayes/Tapir.jl).
"""
struct AutoTapir <: AbstractReverseMode end
