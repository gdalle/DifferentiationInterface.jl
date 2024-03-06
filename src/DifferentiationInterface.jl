"""
    DifferentiationInterface

An experimental redesign for [AbstractDifferentiation.jl]
(https://github.com/JuliaDiff/AbstractDifferentiation.jl).

# Exports

$(EXPORTS)
"""
module DifferentiationInterface

using DocStringExtensions
using FillArrays: OneElement

include("backends.jl")
include("utils.jl")
include("pushforward.jl")
include("pullback.jl")
include("scalar_scalar.jl")
include("scalar_array.jl")
include("array_scalar.jl")
include("array_array.jl")

export ChainRulesReverseBackend,
    ChainRulesForwardBackend,
    EnzymeReverseBackend,
    EnzymeForwardBackend,
    FiniteDiffBackend,
    ForwardDiffBackend,
    ReverseDiffBackend

export value_and_pushforward!, value_and_pushforward
export value_and_pullback!, value_and_pullback

export value_and_derivative
export value_and_multiderivative!, value_and_multiderivative
export value_and_gradient!, value_and_gradient
export value_and_jacobian!, value_and_jacobian

end # module
