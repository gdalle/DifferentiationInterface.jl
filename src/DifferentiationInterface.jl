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

include("backends.jl")
include("implem.jl")
include("mode.jl")
include("utils.jl")
include("pushforward.jl")
include("pullback.jl")
include("scalar_scalar.jl")
include("scalar_array.jl")
include("array_scalar.jl")
include("array_array.jl")
include("custom.jl")

export AutoChainRules, AutoDiffractor

export handles_input_type, handles_output_type, handles_types

export value_and_pushforward!, value_and_pushforward
export pushforward!, pushforward

export value_and_pullback!, value_and_pullback
export pullback!, pullback

export value_and_derivative
export derivative

export value_and_multiderivative!, value_and_multiderivative
export multiderivative!, multiderivative

export value_and_gradient!, value_and_gradient
export gradient!, gradient

export value_and_jacobian!, value_and_jacobian
export jacobian!, jacobian

end # module
