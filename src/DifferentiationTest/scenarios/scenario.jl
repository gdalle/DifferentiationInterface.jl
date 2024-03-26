@kwdef struct Reference
    pushforward=nothing
    pullback=nothing
    derivative=nothing
    gradient=nothing
    jacobian=nothing
    second_derivative=nothing
    hvp=nothing
    hessian=nothing
end

"""
    Scenario{mutating}

Store a testing scenario composed of a function and its input + output + tangents.

# Fields

$(TYPEDFIELDS)
"""
struct Scenario{mutating,F,X,Y,DX,DY,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "pushforward seed"
    dx::DX
    "pullback seed"
    dy::DY
    "reference to compare against"
    ref::R
end

function Scenario{mutating}(
    f::F, x::X, y::Y, dx::DX, dy::DY, ref::R
) where {mutating,F,X,Y,DX,DY,R}
    return Scenario{mutating,F,X,Y,DX,DY,R}(f, x, y, dx, dy, ref)
end

function Base.string(scen::Scenario{mutating}) where {mutating}
    return "$(string(scen.f)): $(typeof(scen.x)) -> $(typeof(scen.y))"
end

is_mutating(::Scenario{mutating}) where {mutating} = mutating

function change_ref(scen::Scenario{mutating}, new_ref::AbstractADType) where {mutating}
    return Scenario{mutating}(scen.f, scen.x, scen.y, scen.dx, scen.dy, new_ref)
end

## Scenario constructors

function Scenario(f, x, ref)
    y = f(x)
    dx = mysimilar_random(x)
    dy = mysimilar_random(y)
    return Scenario{false}(f, x, y, dx, dy, ref)
end

function Scenario(f!, y, x, ref)
    f!(y, x)
    dx = mysimilar_random(x)
    dy = mysimilar_random(y)
    return Scenario{true}(f!, x, y, dx, dy, ref)
end

function Scenario(f; x, y=nothing, ref=AutoForwardDiff())
    if isnothing(y)
        return Scenario(f, x, ref)
    else
        return Scenario(f, y, x, ref)
    end
end
