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

If `y` is provided, `f` is interpreted as a 2-argument function `f!(y, x) = nothing`.
Otherwise, `f` is interpreted as an 1-argument function `f(x) = y`.

The reference keyword `ref` should be a function that takes `x` (and a potential seed `dx` or `dy`) to return the correct object.

The operator behavior keyword `operator` should be either `:inplace` or `:outofplace` depending on what must be tested.
"""
abstract type AbstractScenario{A,O,F,X,Y,R} end

abstract type AbstractFirstOrderScenario{A,O,F,X,Y,R} <: AbstractScenario{A,O,F,X,Y,R} end
abstract type AbstractSecondOrderScenario{A,O,F,X,Y,R} <: AbstractScenario{A,O,F,X,Y,R} end

function compatible(backend::AbstractADType, ::AbstractScenario{A}) where {A}
    if A == 2
        return Bool(supports_mutation(backend))
    end
    return true
end

function Base.string(scen::S) where {A,O,F,X,Y,S<:AbstractScenario{A,O,F,X,Y}}
    return "$(S.name.name){$A,$O} $(string(scen.f)) : $X -> $Y"
end

## Struct definitions

"""
    PushforwardScenario(f; x, y, dx, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct PushforwardScenario{A,O,F,X,Y,DX,R} <: AbstractFirstOrderScenario{A,O,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "pushforward seed"
    dx::DX
    "reference pushforward operator or backend to compare against"
    ref::R
end

"""
    PullbackScenario(f; x, y, dy, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct PullbackScenario{A,O,F,X,Y,DY,R} <: AbstractFirstOrderScenario{A,O,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "pullback seed"
    dy::DY
    "reference pullback operator or backend to compare against"
    ref::R
end

"""
    DerivativeScenario(f; x, y, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct DerivativeScenario{A,O,F,X<:Number,Y,R} <: AbstractFirstOrderScenario{A,O,F,X,Y,R}
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
    GradientScenario(f; x, y, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct GradientScenario{A,O,F,X,Y<:Number,R} <: AbstractFirstOrderScenario{A,O,F,X,Y,R}
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
    JacobianScenario(f; x, y, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct JacobianScenario{A,O,F,X<:AbstractArray,Y<:AbstractArray,R} <:
       AbstractFirstOrderScenario{A,O,F,X,Y,R}
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
    SecondDerivativeScenario(f; x, y, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct SecondDerivativeScenario{A,O,F,X<:Number,Y,R} <:
       AbstractSecondOrderScenario{A,O,F,X,Y,R}
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
    HVPScenario(f; x, y, dx, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct HVPScenario{A,O,F,X,Y<:Number,DX,R} <: AbstractSecondOrderScenario{A,O,F,X,Y,R}
    "function"
    f::F
    "input"
    x::X
    "output"
    y::Y
    "Hessian-vector product seed"
    dx::DX
    "reference Hessian-vector product operator or backend to compare against"
    ref::R
end

"""
    HessianScenario(f; x, y, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct HessianScenario{A,O,F,X<:AbstractArray,Y<:Number,R} <:
       AbstractSecondOrderScenario{A,O,F,X,Y,R}
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
        function $S(f::F; x::X, y=nothing, ref::R=nothing, operator=:inplace) where {F,X,R}
            A = isnothing(y) ? 1 : 2
            if A == 2
                f(y, x)
            else
                y = f(x)
            end
            return ($S){A,operator,F,X,typeof(y),R}(f, x, y, ref)
        end
    end
end

for S in (:PushforwardScenario, :HVPScenario)
    @eval begin
        function $S(
            f::F; x::X, y=nothing, ref::R=nothing, dx=nothing, operator=:inplace
        ) where {F,X,R}
            A = isnothing(y) ? 1 : 2
            if A == 2
                f(y, x)
            else
                y = f(x)
            end
            if isnothing(dx)
                dx = mysimilar_random(x)
            end
            return ($S){A,operator,F,X,typeof(y),typeof(dx),R}(f, x, y, dx, ref)
        end
    end
end

for S in (:PullbackScenario,)
    @eval begin
        function $S(
            f::F; x::X, y=nothing, ref::R=nothing, dy=nothing, operator=:inplace
        ) where {F,X,R}
            A = isnothing(y) ? 1 : 2
            if A == 2
                f(y, x)
            else
                y = f(x)
            end
            if isnothing(dy)
                dy = mysimilar_random(y)
            end
            return ($S){A,operator,F,X,typeof(y),typeof(dy),R}(f, x, y, dy, ref)
        end
    end
end
