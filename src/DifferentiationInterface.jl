"""
    DifferentiationInterface

An interface to various automatic differentiation backends in Julia.

# Exports

$(EXPORTS)
"""
module DifferentiationInterface

using ADTypes:
    ADTypes,
    AbstractADType,
    AbstractForwardMode,
    AbstractFiniteDifferencesMode,
    AbstractReverseMode,
    AbstractSymbolicDifferentiationMode
using DocStringExtensions
using FillArrays: OneElement
using LinearAlgebra: dot
using Test: Test, @test

"""
    AutoFastDifferentiation

Chooses [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl).
"""
struct AutoFastDifferentiation <: AbstractSymbolicDifferentiationMode end

"""
    AutoTaped

Chooses [Taped.jl](https://github.com/withbayes/Taped.jl).

!!! danger
    This backend is experimental, use at your own risk.
"""
struct AutoTaped <: AbstractReverseMode end

include("second_order.jl")
include("traits.jl")
include("utils.jl")
include("prepare.jl")

include("pushforward.jl")
include("pullback.jl")

include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")

include("second_derivative.jl")
include("hvp.jl")
include("hessian.jl")

include("backends.jl")

export AutoFastDifferentiation
export SecondOrder

export value_and_pushforward!!, value_and_pushforward
export value_and_pullback!!, value_and_pullback

export value_and_derivative!!, value_and_derivative
export value_and_gradient!!, value_and_gradient
export value_and_jacobian!!, value_and_jacobian

export pushforward!!, pushforward
export pullback!!, pullback

export derivative!!, derivative
export gradient!!, gradient
export jacobian!!, jacobian

export second_derivative
export hvp
export hessian

export prepare_pushforward, prepare_pullback
export prepare_derivative, prepare_gradient, prepare_jacobian
export prepare_second_derivative, prepare_hvp, prepare_hessian

export check_available, check_mutation, check_hessian

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        f_name = string(exc.f)
        if (
            f_name == "mode" ||
            contains(f_name, "pushforward") ||
            contains(f_name, "pullback") ||
            contains(f_name, "derivative") ||
            contains(f_name, "gradient") ||
            contains(f_name, "jacobian") ||
            contains(f_name, "hvp") ||
            contains(f_name, "hessian")
        )
            for T in argtypes
                if T <: AbstractADType
                    print(
                        io,
                        """\n
                        HINT: One of DifferentiationInterface's functions is missing a method. Some possible fixes:
                        - switch to another backend
                        - if you don't want to switch, load the package extension corresponding to backend `$T`
                        - if the package is already loaded, define the method `$f_name` for the right combination of argument types
                        """,
                    )
                    return nothing
                end
            end
        end
    end
end

end # module
