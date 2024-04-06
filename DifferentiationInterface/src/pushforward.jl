## Preparation

"""
    PushforwardExtras

Abstract type for additional information needed by pushforward operators.
"""
abstract type PushforwardExtras <: Extras end

struct NoPushforwardExtras <: PushforwardExtras end

"""
    prepare_pushforward(f, backend, x) -> extras
    prepare_pushforward(f!, backend, y, x) -> extras

Create an `extras` object subtyping [`PushforwardExtras`](@ref) that can be given to pushforward operators.
"""
prepare_pushforward(f, ::AbstractADType, x) = NoPushforwardExtras()
prepare_pushforward(f!, ::AbstractADType, y, x) = NoPushforwardExtras()

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
    return value_and_pushforward_aux(
        f, backend, x, dx, extras, pushforward_performance(backend)
    )
end

function value_and_pushforward_aux(f, backend, x, dx, extras, ::PushforwardSlow)
    pullback_extras = prepare_pullback(f, backend, x)
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
    new_extras = prepare_pullback(f!, backend, y, x)
    dy = if x isa Number && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dx * value_and_pullback!!(
                f!, y, zero(x), backend, x, basis(backend, y, i), new_extras
            )[2]
        end
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dot(
                dx,
                value_and_pullback!!(
                    f!, y, similar(x), backend, x, basis(backend, y, i), new_extras
                )[2],
            )
        end
    end
    return y, dy
end
