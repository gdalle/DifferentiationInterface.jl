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
include("basis.jl")
include("forward.jl")
include("reverse.jl")
include("derivative.jl")
include("multiderivative.jl")
include("gradient.jl")
include("jacobian.jl")

export ChainRulesReverseBackend,
    ChainRulesForwardBackend,
    EnzymeReverseBackend,
    EnzymeForwardBackend,
    FiniteDiffBackend,
    ForwardDiffBackend,
    ReverseDiffBackend

export value_and_pushforward!, value_and_pushforward
export pushforward!, pushforward
export value_and_pullback!, value_and_pullback
export pullback!, pullback

export value_and_derivative
export value_and_multiderivative!, value_and_multiderivative
export value_and_gradient!, value_and_gradient
export value_and_jacobian!, value_and_jacobian

end # module
