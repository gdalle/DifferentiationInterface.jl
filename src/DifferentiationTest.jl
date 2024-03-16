"""
    DifferentiationInterface.DifferentiationTest

Testing utilities for [`DifferentiationInterface`](@ref).

The functions defined here are empty shells, their implementation is in the package extension `DifferentiationInterfaceTestExt`.

To load this extension, run the following command in your REPL:

    import ForwardDiff, JET, Test
"""
module DifferentiationTest

using DocStringExtensions

"""
    Scenario

Store a testing scenario composed of a function and its input + output + tangents.

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct Scenario{F,X<:Union{Number,AbstractArray},Y<:Union{Number,AbstractArray}}
    "function"
    f::F
    "mutation"
    mutating::Bool
    "input"
    x::X
    "output"
    y::Y
    "pushforward seed"
    dx::X
    "pullback seed"
    dy::Y
end

similar_random(z::Number) = randn(eltype(z))

function similar_random(z::AbstractArray)
    zz = similar(z)
    zz .= randn(eltype(zz), size(zz))
    return zz
end

function Scenario(f, x::Union{Number,AbstractArray})
    y = f(x)
    dx = similar_random(x)
    dy = similar_random(y)
    return Scenario(; f, x, y, dx, dy, mutating=false)
end

function Scenario(f!, x::Union{Number,AbstractArray}, s::NTuple{N,<:Integer}) where {N}
    y = zeros(eltype(x), s...)
    f!(y, x)
    dx = similar_random(x)
    dy = similar_random(y)
    return Scenario(; f=f!, x, y, dx, dy, mutating=true)
end

function default_scenarios end
function test_operators_allocating end
function test_operators_mutating end

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

    import ForwardDiff, JET, Test
""",
            )
        end
    end
end

end
