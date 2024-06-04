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

Some scenarios have the following fields:

- `dx` or `dy`: seed for the low-level operators
- `first_order_ref`: reference to compare against for second order scenarios when the operator also computes a first-order quantity

# Constructor

If `y` is provided, `f` is interpreted as a 2-argument function `f!(y, x) = nothing`.
Otherwise, `f` is interpreted as an 1-argument function `f(x) = y`.

The reference keyword `ref` (as well as `first_order_ref`) should be a function that takes `x` (and a potential seed `dx` or `dy`) to return the correct object.

The keyword `place` should be either `:inplace` or `:outofplace` depending on what kind of operator must be tested.
"""
abstract type AbstractScenario{args,place,F,X,Y} end

abstract type AbstractFirstOrderScenario{args,place,F,X,Y,R} <:
              AbstractScenario{args,place,F,X,Y} end
abstract type AbstractSecondOrderScenario{args,place,F,X,Y,R2,R1} <:
              AbstractScenario{args,place,F,X,Y} end

scen_type(scenario::AbstractScenario) = nameof(typeof(scenario))
nb_args(::AbstractScenario{args}) where {args} = args
operator_place(::AbstractScenario{args,place}) where {args,place} = place

function compatible(backend::AbstractADType, scen::AbstractScenario)
    if nb_args(scen) == 2
        return Bool(twoarg_support(backend))
    end
    return true
end

function group_by_scen_type(scenarios)
    return Dict(
        st => filter(s -> scen_type(s) == st, scenarios) for
        st in unique(scen_type.(scenarios))
    )
end

function Base.print(
    io::IO, scen::S
) where {args,place,F,X,Y,S<:AbstractScenario{args,place,F,X,Y}}
    return print(io, "$(nameof(S)){$args,$place}($(string(scen.f)) : $X -> $Y)")
end

## Struct definitions

"""
    PushforwardScenario(f; x, y, dx, ref, place)

See [`AbstractScenario`](@ref) for details.
"""
struct PushforwardScenario{args,place,F,X,Y,DX,R} <:
       AbstractFirstOrderScenario{args,place,F,X,Y,R}
    f::F
    x::X
    y::Y
    dx::DX
    ref::R
end

"""
    PullbackScenario(f; x, y, dy, ref, place)

See [`AbstractScenario`](@ref) for details.
"""
struct PullbackScenario{args,place,F,X,Y,DY,R} <:
       AbstractFirstOrderScenario{args,place,F,X,Y,R}
    f::F
    x::X
    y::Y
    dy::DY
    ref::R
end

"""
    DerivativeScenario(f; x, y, ref, place)

See [`AbstractScenario`](@ref) for details.
"""
struct DerivativeScenario{args,place,F,X<:Number,Y,R} <:
       AbstractFirstOrderScenario{args,place,F,X,Y,R}
    f::F
    x::X
    y::Y
    ref::R
end

"""
    GradientScenario(f; x, y, ref, place)

See [`AbstractScenario`](@ref) for details.
"""
struct GradientScenario{args,place,F,X,Y<:Number,R} <:
       AbstractFirstOrderScenario{args,place,F,X,Y,R}
    f::F
    x::X
    y::Y
    ref::R
end

"""
    JacobianScenario(f; x, y, ref, place)

See [`AbstractScenario`](@ref) for details.
"""
struct JacobianScenario{args,place,F,X<:AbstractArray,Y<:AbstractArray,R} <:
       AbstractFirstOrderScenario{args,place,F,X,Y,R}
    f::F
    x::X
    y::Y
    ref::R
end

"""
    SecondDerivativeScenario(f; x, y, ref, first_order_ref, place)

See [`AbstractScenario`](@ref) for details.
"""
struct SecondDerivativeScenario{args,place,F,X<:Number,Y,R2,R1} <:
       AbstractSecondOrderScenario{args,place,F,X,Y,R2,R1}
    f::F
    x::X
    y::Y
    ref::R2
    first_order_ref::R1
end

"""
    HVPScenario(f; x, y, dx, ref, first_order_ref, place)

See [`AbstractScenario`](@ref) for details.
"""
struct HVPScenario{args,place,F,X,Y<:Number,DX,R2,R1} <:
       AbstractSecondOrderScenario{args,place,F,X,Y,R2,R1}
    f::F
    x::X
    y::Y
    dx::DX
    ref::R2
    first_order_ref::R1
end

"""
    HessianScenario(f; x, y, ref, first_order_ref, place)

See [`AbstractScenario`](@ref) for details.
"""
struct HessianScenario{args,place,F,X<:AbstractArray,Y<:Number,R2,R1} <:
       AbstractSecondOrderScenario{args,place,F,X,Y,R2,R1}
    f::F
    x::X
    y::Y
    ref::R2
    first_order_ref::R1
end

## Constructors

### First order

for S in (:DerivativeScenario, :GradientScenario, :JacobianScenario)
    @eval function $S(f::F; x::X, y=nothing, ref::R=nothing, place=:inplace) where {F,X,R}
        args = isnothing(y) ? 1 : 2
        if args == 2
            f(y, x)
        else
            y = f(x)
        end
        return ($S){args,place,F,X,typeof(y),R}(f, x, y, ref)
    end
end

function PushforwardScenario(
    f::F; x::X, y=nothing, ref::R=nothing, dx=nothing, place=:inplace
) where {F,X,R}
    args = isnothing(y) ? 1 : 2
    if args == 2
        f(y, x)
    else
        y = f(x)
    end
    if isnothing(dx)
        dx = mycopy_random(x)
    end
    return PushforwardScenario{args,place,F,X,typeof(y),typeof(dx),R}(f, x, y, dx, ref)
end

function PullbackScenario(
    f::F; x::X, y=nothing, ref::R=nothing, dy=nothing, place=:inplace
) where {F,X,R}
    args = isnothing(y) ? 1 : 2
    if args == 2
        f(y, x)
    else
        y = f(x)
    end
    if isnothing(dy)
        dy = mycopy_random(y)
    end
    return PullbackScenario{args,place,F,X,typeof(y),typeof(dy),R}(f, x, y, dy, ref)
end

### Second order

for S in (:SecondDerivativeScenario, :HessianScenario)
    @eval function $S(
        f::F; x::X, y=nothing, ref::R2=nothing, first_order_ref::R1=nothing, place=:inplace
    ) where {F,X,R2,R1}
        args = isnothing(y) ? 1 : 2
        if args == 2
            f(y, x)
        else
            y = f(x)
        end
        return ($S){args,place,F,X,typeof(y),R2,R1}(f, x, y, ref, first_order_ref)
    end
end

function HVPScenario(
    f::F;
    x::X,
    y=nothing,
    ref::R2=nothing,
    first_order_ref::R1=nothing,
    dx=nothing,
    place=:inplace,
) where {F,X,R2,R1}
    args = isnothing(y) ? 1 : 2
    if args == 2
        f(y, x)
    else
        y = f(x)
    end
    if isnothing(dx)
        dx = mycopy_random(x)
    end
    return HVPScenario{args,place,F,X,typeof(y),typeof(dx),R2,R1}(
        f, x, y, dx, ref, first_order_ref
    )
end

## Change function

for S in (
    :DerivativeScenario,
    :GradientScenario,
    :JacobianScenario,
    :PullbackScenario,
    :PushforwardScenario,
)
    @eval function change_function(s::($S), f)
        return ($S)(
            f;
            x=s.x,
            y=(nb_args(s) == 1 ? nothing : s.y),
            ref=s.ref,
            place=operator_place(s),
        )
    end
end

for S in (:SecondDerivativeScenario, :HessianScenario, :HVPScenario)
    @eval function change_function(s::($S), f)
        return ($S)(
            f;
            x=s.x,
            y=(nb_args(s) == 1 ? nothing : s.y),
            ref=s.ref,
            first_order_ref=s.first_order_ref,
            place=operator_place(s),
        )
    end
end
