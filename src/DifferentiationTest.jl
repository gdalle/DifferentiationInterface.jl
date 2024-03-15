"""
    DifferentiationInterface.DifferentiationTest

Testing utilities for [`DifferentiationInterface`](@ref).

The functions defined here are empty shells, their implementation is in the package extension `DifferentiationInterfaceTestExt`.

To load this extension, run the following command in your REPL:

    import ForwardDiff, JET, Random, Test
"""
module DifferentiationTest

using ADTypes: AbstractADType, AbstractForwardMode, AbstractReverseMode
using DifferentiationInterface: ForwardMode, ReverseMode, autodiff_mode, zero!
import DifferentiationInterface as DI
using DocStringExtensions

"""
    Scenario

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct Scenario{
    F,
    X<:Union{Number,AbstractArray},
    Y<:Union{Number,AbstractArray},
    D1<:Union{Nothing,Number},
    D2<:Union{Nothing,AbstractArray},
    D3<:Union{Nothing,AbstractArray},
    D4<:Union{Nothing,AbstractArray},
}
    "function"
    f::F
    "argument"
    x::X
    "primal value"
    y::Y
    "pushforward seed"
    dx::X
    "pullback seed"
    dy::Y
    "pullback result"
    dx_true::X
    "pushforward result"
    dy_true::Y
    "derivative result"
    der_true::D1 = nothing
    "multiderivative result"
    multider_true::D2 = nothing
    "gradient result"
    grad_true::D3 = nothing
    "Jacobian result"
    jac_true::D4 = nothing
    "mutation"
    mutating::Bool = false
end

function default_scenarios end
function test_operators_allocating end
function test_operators_mutating end

"""
    AutoZeroForward <: ADTypes.AbstractForwardMode

Trivial backend that sets all derivatives to zero. Used in testing and benchmarking.
"""
struct AutoZeroForward <: AbstractForwardMode end

"""
    AutoZeroReverse <: ADTypes.AbstractReverseMode

Trivial backend that sets all derivatives to zero. Used in testing and benchmarking.
"""
struct AutoZeroReverse <: AbstractReverseMode end

function DI.value_and_pushforward!(
    dy::Union{Number,AbstractArray}, ::AutoZeroForward, f, x, dx, extras=nothing
)
    return f(x), zero!(dy)
end

function DI.value_and_pullback!(
    dx::Union{Number,AbstractArray}, ::AutoZeroReverse, f, x, dy, extras=nothing
)
    return f(x), zero!(dx)
end

function DI.value_and_pushforward!(
    y::AbstractArray,
    dy::Union{Number,AbstractArray},
    ::AutoZeroForward,
    f!,
    x,
    dx,
    extras=nothing,
)
    f!(y, x)
    return y, zero!(dy)
end

function DI.value_and_pullback!(
    y::AbstractArray,
    dx::Union{Number,AbstractArray},
    ::AutoZeroReverse,
    f!,
    x,
    dy,
    extras=nothing,
)
    f!(y, x)
    return y, zero!(dx)
end

export Scenario, default_scenarios
export test_operators_allocating, test_operators_mutating

# see https://docs.julialang.org/en/v1/base/base/#Base.Experimental.register_error_hint

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        f_name = string(exc.f)
        if (contains(f_name, "scenario") || contains(f_name, "test_"))
            print(
                io,
                """\n
HINT: To use the `DifferentiationInterface.DifferentiationTest` submodule, you need to load the `DifferentiationInterfaceTestExt` package extension. Run the following command in your REPL:

    import ForwardDiff, JET, Random, Test
""",
            )
        end
    end
end

end
