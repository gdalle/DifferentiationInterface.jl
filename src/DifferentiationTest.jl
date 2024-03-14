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

@kwdef struct Scenario{F,X,Y,D1,D2,D3,D4}
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
end

function make_scenario end
function default_scenarios end
function test_pushforward end
function test_pullback end
function test_derivative end
function test_multiderivative end
function test_gradient end
function test_jacobian end
function test_all_operators end

struct AutoZeroForward <: AbstractForwardMode end
struct AutoZeroReverse <: AbstractReverseMode end

function DI.value_and_pushforward!(dy, ::AutoZeroForward, f, x, dx, extras=nothing)
    return f(x), zero!(dy)
end

function DI.value_and_pullback!(dx, ::AutoZeroReverse, f, x, dy, extras=nothing)
    return f(x), zero!(dx)
end

export Scenario, make_scenario, default_scenarios
export test_pushforward, test_pullback
export test_derivative, test_multiderivative, test_gradient, test_jacobian
export test_all_operators

# see https://docs.julialang.org/en/v1/base/base/#Base.Experimental.register_error_hint

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f in [
            make_scenario,
            default_scenarios,
            test_pushforward,
            test_pullback,
            test_derivative,
            test_multiderivative,
            test_gradient,
            test_jacobian,
            test_all_operators,
        ]
            println(
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
