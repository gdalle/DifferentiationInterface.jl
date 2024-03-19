"""
    DifferentiationInterface

An interface to various automatic differentiation backends in Julia.

# Exports

$(EXPORTS)
"""
module DifferentiationInterface

using ADTypes: ADTypes, AbstractADType
using DocStringExtensions
using FillArrays: OneElement
using LinearAlgebra: dot
using Test: Test

"""
    AutoFastDifferentiation

Chooses [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl).
"""
struct AutoFastDifferentiation <: ADTypes.AbstractSymbolicDifferentiationMode end

include("mode.jl")
include("mutation.jl")
include("utils.jl")
include("prepare.jl")

include("pushforward.jl")
include("pullback.jl")
include("zero.jl")

include("derivative.jl")
include("multiderivative.jl")
include("gradient.jl")
include("jacobian.jl")

include("second_order.jl")
include("second_derivative.jl")
include("hessian_vector_product.jl")
include("hessian.jl")

export AutoFastDifferentiation
export SecondOrder

export value_and_pushforward!, value_and_pushforward
export value_and_pullback!, value_and_pullback

export value_and_derivative
export value_and_multiderivative!, value_and_multiderivative
export value_and_gradient!, value_and_gradient
export value_and_jacobian!, value_and_jacobian

export gradient_and_hessian_vector_product!, gradient_and_hessian_vector_product
export hessian_vector_product!, hessian_vector_product

export value_derivative_and_second_derivative
export value_gradient_and_hessian!, value_gradient_and_hessian

export pushforward!, pushforward
export pullback!, pullback

export derivative
export multiderivative!, multiderivative
export gradient!, gradient
export jacobian!, jacobian

export second_derivative
export hessian!, hessian

export prepare_pushforward
export prepare_pullback

export prepare_derivative
export prepare_multiderivative
export prepare_gradient
export prepare_jacobian

export prepare_second_derivative
export prepare_hessian
export prepare_hessian_vector_product

# submodules
include("DifferentiationTest/DifferentiationTest.jl")

end # module
