
"""
    Scenario{mutating}

Store a testing scenario composed of a function and its input + output + tangents.

# Fields

$(TYPEDFIELDS)
"""
struct Scenario{mutating,F,X,Y,DX,DY}
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
end

function Scenario{mutating}(f::F, x::X, y::Y, dx::DX, dy::DY) where {mutating,F,X,Y,DX,DY}
    return Scenario{mutating,F,X,Y,DX,DY}(f, x, y, dx, dy)
end

function Base.string(scen::Scenario{mutating}) where {mutating}
    return "$(string(scen.f)): $(typeof(scen.x)) -> $(typeof(scen.y))"
end

is_mutating(::Scenario{mutating}) where {mutating} = mutating

## Scenario constructors

mysimilar_random(z::Number) = randn(eltype(z))

function mysimilar_random(z::AbstractArray)
    zz = mysimilar(z)
    zz .= randn(eltype(zz), size(zz))
    return zz
end

function Scenario(f, x::Union{Number,AbstractArray})
    y = f(x)
    dx = mysimilar_random(x)
    dy = mysimilar_random(y)
    return Scenario{false}(f, x, y, dx, dy)
end

function Scenario(f!, x::Union{Number,AbstractArray}, s::NTuple{N,<:Integer}) where {N}
    y = zeros(eltype(x), s...)
    f!(y, x)
    dx = mysimilar_random(x)
    dy = mysimilar_random(y)
    return Scenario{true}(f!, x, y, dx, dy)
end
