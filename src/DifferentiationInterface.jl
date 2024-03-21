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
using Test: Test

"""
    AutoFastDifferentiation

Chooses [FastDifferentiation.jl](https://github.com/brianguenter/FastDifferentiation.jl).
"""
struct AutoFastDifferentiation <: AbstractSymbolicDifferentiationMode end

include("traits.jl")
include("utils.jl")

include("pushforward.jl")
include("pullback.jl")

include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")

include("backends.jl")

export AutoFastDifferentiation

export value_and_pushforward!, value_and_pushforward
export value_and_pullback!, value_and_pullback

export value_and_derivative!, value_and_derivative
export value_and_gradient!, value_and_gradient
export value_and_jacobian!, value_and_jacobian

export check_available, check_mutation

# submodules
include("DifferentiationTest/DifferentiationTest.jl")

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
                        - if the package is already loaded, define the method `$f_name` for arguments `$(Tuple(argtypes))`
                        """,
                    )
                    return nothing
                end
            end
        end
    end
end

end # module
