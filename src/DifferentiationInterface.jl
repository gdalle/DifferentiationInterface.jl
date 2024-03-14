"""
    DifferentiationInterface

An interface to various automatic differentiation backends in Julia.

# Exports

$(EXPORTS)
"""
module DifferentiationInterface

using ADTypes:
    AbstractADType, AbstractForwardMode, AbstractReverseMode, AbstractFiniteDifferencesMode
using DocStringExtensions
using FillArrays: OneElement

const NumberOrArray = Union{Number,AbstractArray}

include("backends.jl")
include("mode.jl")
include("utils.jl")
include("pushforward.jl")
include("pullback.jl")
include("derivative.jl")
include("multiderivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("mode_trait.jl")
include("prepare.jl")

export value_and_pushforward!, value_and_pushforward
export pushforward!, pushforward
export prepare_pushforward

export value_and_pullback!, value_and_pullback
export pullback!, pullback
export prepare_pullback

export value_and_derivative
export derivative
export prepare_derivative

export value_and_multiderivative!, value_and_multiderivative
export multiderivative!, multiderivative
export prepare_multiderivative

export value_and_gradient!, value_and_gradient
export gradient!, gradient
export prepare_gradient

export value_and_jacobian!, value_and_jacobian
export jacobian!, jacobian
export prepare_jacobian

end # module
