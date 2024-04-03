"""
    AbstractScenario

Store a testing scenario composed of a function and its input + output.

This abstract type should never be used directly: construct one of the subtypes corresponding to the operator you want to test.
    
# Subtypes

- [`PushforwardScenario`](@ref)
- [`PullbackScenario`](@ref)
- [`DerivativeScenario`](@ref)
- [`GradientScenario`](@ref)
- [`JacobianScenario`](@ref)
- [`SecondDerivativeScenario`](@ref)
- [`HVPScenario`](@ref)
- [`HessianScenario`](@ref)
    
# Fields

All subtypes have the following fields:

- `f`: function to apply
- `x`: primal input
- `y`: primal output
- `ref`: reference to compare against

In addition, some subtypes contain an additional seed (`dx` or `dy`).

# Constructor

When no seed is needed, the constructor looks like

    GradientScenario(f; x, y=nothing, ref=nothing)

When a seed is needed, the constructor looks like

    PushforwardScenario(f; x, y=nothing, ref=nothing, dx=nothing)

If `y` is provided, `f` is interpreted as a _mutating_ function `f!(y, x) = nothing`.
Otherwise, `f` is interpreted as an _allocating_ function `f(x) = y`.

The reference `ref` should be a function that takes `x` (and a potential seed `dx` or `dy`) to return the correct object.
"""
abstract type AbstractScenario{mut,F,X,Y,R} end

abstract type AbstractFirstOrderScenario{mut,F,X,Y,R} <: AbstractScenario{mut,F,X,Y,R} end
abstract type AbstractSecondOrderScenario{mut,F,X,Y,R} <: AbstractScenario{mut,F,X,Y,R} end

ismutating(::AbstractScenario{mut}) where {mut} = mut

function Base.string(scen::S) where {mut,F,X,Y,S<:AbstractScenario{mut,F,X,Y}}
    return "$(S.name.name) $(string(scen.f)) : $X -> $Y"
end

## Struct definitions

"""
    PushforwardScenario(f; x, y, ref, dx)

See [`AbstractScenario`](@ref) for details.
"""
struct PushforwardScenario{mut,F,X,Y,R,DX} <: AbstractFirstOrderScenario{mut,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "reference pushforward operator or backend to compare against"
    ref::R
    "pushforward seed"
    dx::DX
end

"""
    PullbackScenario(f; x, y, ref, dy)

See [`AbstractScenario`](@ref) for details.
"""
struct PullbackScenario{mut,F,X,Y,R,DY} <: AbstractFirstOrderScenario{mut,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "reference pullback operator or backend to compare against"
    ref::R
    "pullback seed"
    dy::DY
end

"""
    DerivativeScenario(f; x, y, ref)

See [`AbstractScenario`](@ref) for details.
"""
struct DerivativeScenario{mut,F,X<:Number,Y,R} <: AbstractFirstOrderScenario{mut,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "reference derivative operator or backend to compare against"
    ref::R
end

"""
    GradientScenario(f; x, y, ref)

See [`AbstractScenario`](@ref) for details.
"""
struct GradientScenario{mut,F,X,Y<:Number,R} <: AbstractFirstOrderScenario{mut,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "reference gradient operator or backend to compare against"
    ref::R
end

"""
    JacobianScenario(f; x, y, ref)

See [`AbstractScenario`](@ref) for details.
"""
struct JacobianScenario{mut,F,X<:AbstractArray,Y<:AbstractArray,R} <:
       AbstractFirstOrderScenario{mut,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "reference Jacobian operator or backend to compare against"
    ref::R
end

"""
    SecondDerivativeScenario(f; x, y, ref)

See [`AbstractScenario`](@ref) for details.
"""
struct SecondDerivativeScenario{mut,F,X<:Number,Y,R} <:
       AbstractSecondOrderScenario{mut,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "reference second derivative operator or backend to compare against"
    ref::R
end

"""
    HVPScenario(f; x, y, ref, dx)

See [`AbstractScenario`](@ref) for details.
"""
struct HVPScenario{mut,F,X,Y<:Number,R,DX} <: AbstractSecondOrderScenario{mut,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "reference Hessian-vector product operator or backend to compare against"
    ref::R
    "Hessian-vector product seed"
    dx::DX
end

"""
    HessianScenario(f; x, y, ref)

See [`AbstractScenario`](@ref) for details.
"""
struct HessianScenario{mut,F,X<:AbstractArray,Y<:Number,R} <:
       AbstractSecondOrderScenario{mut,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "reference Hessian operator or backend to compare against"
    ref::R
end

## Constructors

for S in (
    :DerivativeScenario,
    :GradientScenario,
    :JacobianScenario,
    :SecondDerivativeScenario,
    :HessianScenario,
)
    @eval begin
        function $S(f::F; x::X, y=nothing, ref::R=nothing) where {F,X,R}
            mut = !isnothing(y)
            if mut
                f(y, x)
            else
                y = f(x)
            end
            return ($S){mut,F,X,typeof(y),R}(f, x, y, ref)
        end
    end
end

for S in (:PushforwardScenario, :HVPScenario)
    @eval begin
        function $S(f::F; x::X, y=nothing, ref::R=nothing, dx=nothing) where {F,X,R}
            mut = !isnothing(y)
            if mut
                f(y, x)
            else
                y = f(x)
            end
            if isnothing(dx)
                dx = mysimilar_random(x)
            end
            return ($S){mut,F,X,typeof(y),R,typeof(dx)}(f, x, y, ref, dx)
        end
    end
end

for S in (:PullbackScenario,)
    @eval begin
        function $S(f::F; x::X, y=nothing, ref::R=nothing, dy=nothing) where {F,X,R}
            mut = !isnothing(y)
            if mut
                f(y, x)
            else
                y = f(x)
            end
            if isnothing(dy)
                dy = mysimilar_random(y)
            end
            return ($S){mut,F,X,typeof(y),R,typeof(dy)}(f, x, y, ref, dy)
        end
    end
end
