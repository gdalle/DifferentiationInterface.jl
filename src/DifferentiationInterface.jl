"""
    DifferentiationInterface

An experimental redesign for [AbstractDifferentiation.jl]
(https://github.com/JuliaDiff/AbstractDifferentiation.jl).

# Exports

$(EXPORTS)
"""
module DifferentiationInterface

using DocStringExtensions

abstract type AbstractBackend end
abstract type AbstractForwardBackend <: AbstractBackend end
abstract type AbstractReverseBackend <: AbstractBackend end

include("backends.jl")
include("forward.jl")
include("reverse.jl")

export ChainRulesReverseBackend,
    ChainRulesForwardBackend,
    EnzymeReverseBackend,
    EnzymeForwardBackend,
    FiniteDiffBackend,
    ForwardDiffBackend,
    ReverseDiffBackend
export pushforward!, value_and_pushforward!
export pullback!, value_and_pullback!

end # module
