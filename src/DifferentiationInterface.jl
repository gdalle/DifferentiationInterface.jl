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
include("mode.jl")
include("utils.jl")
include("pushforward.jl")
include("pullback.jl")
include("derivative.jl")
include("multiderivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("additional_args.jl")
include("prepare.jl")

# submodules
include("DifferentiationTest.jl")

export value_and_pushforward!, value_and_pushforward
export value_and_pullback!, value_and_pullback
export value_and_derivative
export value_and_multiderivative!, value_and_multiderivative
export value_and_gradient!, value_and_gradient
export value_and_jacobian!, value_and_jacobian

export pushforward!, pushforward
export pullback!, pullback
export derivative
export multiderivative!, multiderivative
export gradient!, gradient
export jacobian!, jacobian

export prepare_pushforward
export prepare_pullback
export prepare_derivative
export prepare_multiderivative
export prepare_gradient
export prepare_jacobian

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        f_name = string(exc.f)
        if (
            f_name == "autodiff_mode" ||
            contains(f_name, "pushforward") ||
            contains(f_name, "pullback") ||
            contains(f_name, "derivative") ||
            contains(f_name, "gradient") ||
            contains(f_name, "jacobian")
        )
            for T in argtypes
                if T <: AbstractADType
                    print(
                        io,
                        """\n
HINT: To use `DifferentiationInterface` with backend `$T`, you need to load the corresponding package extension.
            """,
                    )
                    return nothing
                end
            end
        end
    end
end

end # module
