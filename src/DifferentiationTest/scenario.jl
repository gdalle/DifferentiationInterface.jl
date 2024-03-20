
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

Base.string(scen::Scenario) = "$(string(scen.f)): $(typeof(scen.x)) -> $(typeof(scen.y))"

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

allocating(scenarios::Vector{<:Scenario}) = filter(s -> !s.mutating, scenarios)
mutating(scenarios::Vector{<:Scenario}) = filter(s -> s.mutating, scenarios)

function scalar_scalar(scenarios::Vector{<:Scenario})
    return filter(scenarios) do s
        typeof(s.x) <: Number && typeof(s.y) <: Number
    end
end

function scalar_array(scenarios::Vector{<:Scenario})
    return filter(scenarios) do s
        typeof(s.x) <: Number && typeof(s.y) <: AbstractArray
    end
end

function array_scalar(scenarios::Vector{<:Scenario})
    return filter(scenarios) do s
        typeof(s.x) <: AbstractArray && typeof(s.y) <: Number
    end
end

function array_array(scenarios::Vector{<:Scenario})
    return filter(scenarios) do s
        typeof(s.x) <: AbstractArray && typeof(s.y) <: AbstractArray
    end
end
