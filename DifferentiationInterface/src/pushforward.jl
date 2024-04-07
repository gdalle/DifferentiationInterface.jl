## Preparation

"""
    PushforwardExtras

Abstract type for additional information needed by pushforward operators.
"""
abstract type PushforwardExtras <: Extras end

struct NoPushforwardExtras <: PushforwardExtras end

struct PullbackPushforwardExtras{E} <: PushforwardExtras
    pullback_extras::E
end

"""
    prepare_pushforward(f, backend, x) -> extras
    prepare_pushforward(f!, backend, y, x) -> extras

Create an `extras` object subtyping [`PushforwardExtras`](@ref) that can be given to pushforward operators.
"""
function prepare_pushforward(f, backend::AbstractADType, x)
    return prepare_pushforward_aux(f, backend, x, pushforward_performance(backend))
end

function prepare_pushforward(f!, backend::AbstractADType, y, x)
    return prepare_pushforward_aux(f!, backend, y, x, pushforward_performance(backend))
end

function prepare_pushforward_aux(f, backend, x, ::PushforwardSlow)
    return PullbackPushforwardExtras(prepare_pullback(f, backend, x))
end

function prepare_pushforward_aux(f!, backend, y, x, ::PushforwardSlow)
    return PullbackPushforwardExtras(prepare_pullback(f!, backend, y, x))
end

## Allocating

"""
    value_and_pushforward(f, backend, x, dx, [extras]) -> (y, dy)

!!! info
    Required primitive for forward mode backends to support allocating functions.
"""
function value_and_pushforward(
    f,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x),
)
    return value_and_pushforward_aux(f, backend, x, dx, extras)
end

function value_and_pushforward_aux(f, backend, x, dx, extras::PullbackPushforwardExtras)
    (; pullback_extras) = extras
    y, pullbackfunc = value_and_pullback_split(f, backend, x, pullback_extras)
    dy = if x isa Number && y isa Number
        dx * pullbackfunc(one(y))
    elseif x isa AbstractArray && y isa Number
        dot(dx, pullbackfunc(one(y)))
    elseif x isa Number && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dx * pullbackfunc(basis(backend, y, i))
        end
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dot(dx, pullbackfunc(basis(backend, y, i)))
        end
    end
    return y, dy
end

"""
    value_and_pushforward!!(f, dy, backend, x, dx, [extras]) -> (y, dy)
"""
function value_and_pushforward!!(
    f,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x),
)
    return value_and_pushforward(f, backend, x, dx, extras)
end

"""
    pushforward(f, backend, x, dx, [extras]) -> (y, dy)
"""
function pushforward(
    f,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x),
)
    return value_and_pushforward(f, backend, x, dx, extras)[2]
end

"""
    pushforward!!(f, dy, backend, x, dx, [extras]) -> (y, dy)
"""
function pushforward!!(
    f,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x),
)
    return value_and_pushforward!!(f, dy, backend, x, dx, extras)[2]
end

## Mutating

"""
    value_and_pushforward!!(f!, y, dy, backend, x, dx, [extras]) -> (y, dy)

!!! info
    Required primitive for forward mode backends to support mutating functions.
"""
function value_and_pushforward!!(
    f!,
    y,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f!, backend, y, x),
)
    return value_and_pushforward_aux!!(f!, y, dy, backend, x, dx, extras)
end

function value_and_pushforward_aux!!(
    f!, y, dy, backend, x, dx, extras::PullbackPushforwardExtras
)
    (; pullback_extras) = extras
    new_dy = if x isa Number && y isa AbstractArray
        map!(dy, CartesianIndices(y)) do i
            dx * pullback!!(f!, y, zero(x), backend, x, basis(backend, y, i), pullback_extras)
        end
    elseif x isa AbstractArray && y isa AbstractArray
        map!(dy, CartesianIndices(y)) do i
            dot(
                dx,
                pullback!!(
                    f!, y, similar(x), backend, x, basis(backend, y, i), pullback_extras
                ),
            )
        end
    end
    f!(y, x)
    return y, new_dy
end

"""
    pushforward!!(f!, y, dy, backend, x, dx, [extras]) -> dy
"""
function pushforward!!(
    f!,
    y,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f!, backend, y, x),
)
    return value_and_pushforward!!(f!, y, dy, backend, x, dx, extras)[2]
end
