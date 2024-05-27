## Docstrings

"""
    prepare_pushforward(f,     backend, x, dx) -> extras
    prepare_pushforward(f!, y, backend, x, dx) -> extras

Create an `extras` object that can be given to [`pushforward`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    In the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_pushforward end

"""
    prepare_pushforward_same_point(f,     backend, x, dx) -> extras_same
    prepare_pushforward_same_point(f!, y, backend, x, dx) -> extras_same

Create an `extras_same` object that can be given to [`pushforward`](@ref) and its variants _if they are applied at the same point `x`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    In the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_pushforward_same_point end

"""
    value_and_pushforward(f,     backend, x, dx, [extras]) -> (y, dy)
    value_and_pushforward(f!, y, backend, x, dx, [extras]) -> (y, dy)

Compute the value and the pushforward of the function `f` at point `x` with seed `dx`.

!!! info
    Required primitive for forward mode backends.
"""
function value_and_pushforward end

"""
    value_and_pushforward!(f,     dy, backend, x, dx, [extras]) -> (y, dy)
    value_and_pushforward!(f!, y, dy, backend, x, dx, [extras]) -> (y, dy)

Compute the value and the pushforward of the function `f` at point `x` with seed `dx`, overwriting `dy`.
"""
function value_and_pushforward! end

"""
    pushforward(f,     backend, x, dx, [extras]) -> dy
    pushforward(f!, y, backend, x, dx, [extras]) -> dy

Compute the pushforward of the function `f` at point `x` with seed `dx`.
"""
function pushforward end

"""
    pushforward!(f,     dy, backend, x, dx, [extras]) -> dy
    pushforward!(f!, y, dy, backend, x, dx, [extras]) -> dy

Compute the pushforward of the function `f` at point `x` with seed `dx`, overwriting `dy`.
"""
function pushforward! end

## Preparation

"""
    PushforwardExtras

Abstract type for additional information needed by [`pushforward`](@ref) and its variants.
"""
abstract type PushforwardExtras <: Extras end

struct NoPushforwardExtras <: PushforwardExtras end

struct PullbackPushforwardExtras{E} <: PushforwardExtras
    pullback_extras::E
end

function prepare_pushforward(f::F, backend::AbstractADType, x, dx) where {F}
    return prepare_pushforward_aux(f, backend, x, dx, pushforward_performance(backend))
end

function prepare_pushforward(f!::F, y, backend::AbstractADType, x, dx) where {F}
    return prepare_pushforward_aux(f!, y, backend, x, dx, pushforward_performance(backend))
end

function prepare_pushforward_aux(f::F, backend, x, dx, ::PushforwardSlow) where {F}
    y = f(x)
    dy = y isa Number ? one(y) : basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f, backend, x, dy)
    return PullbackPushforwardExtras(pullback_extras)
end

function prepare_pushforward_aux(f!::F, y, backend, x, dx, ::PushforwardSlow) where {F}
    dy = y isa Number ? one(y) : basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f!, y, backend, x, dy)
    return PullbackPushforwardExtras(pullback_extras)
end

# Throw error if backend is missing
prepare_pushforward_aux(f::F, backend, x, dy, ::PushforwardFast) where {F}     = throw(MissingBackendError(backend))
prepare_pushforward_aux(f!::F, y, backend, x, dy, ::PushforwardFast) where {F} = throw(MissingBackendError(backend))

## Preparation (same point)

function prepare_pushforward_same_point(
    f::F, backend::AbstractADType, x, dx, extras::PushforwardExtras
) where {F}
    return extras
end

function prepare_pushforward_same_point(
    f!::F, y, backend::AbstractADType, x, dx, extras::PushforwardExtras
) where {F}
    return extras
end

function prepare_pushforward_same_point(f::F, backend::AbstractADType, x, dx) where {F}
    extras = prepare_pushforward(f, backend, x, dx)
    return prepare_pushforward_same_point(f, backend, x, dx, extras)
end

function prepare_pushforward_same_point(f!::F, y, backend::AbstractADType, x, dx) where {F}
    extras = prepare_pushforward(f!, y, backend, x, dx)
    return prepare_pushforward_same_point(f!, y, backend, x, dx, extras)
end

## One argument

function value_and_pushforward(
    f::F,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x, dx),
) where {F}
    return value_and_pushforward_onearg_aux(f, backend, x, dx, extras)
end

function value_and_pushforward_onearg_aux(
    f::F, backend, x, dx, extras::PullbackPushforwardExtras
) where {F}
    @compat (; pullback_extras) = extras
    y = f(x)
    dy = if x isa Number && y isa Number
        dx * pullback(f, backend, x, one(y), pullback_extras)
    elseif x isa AbstractArray && y isa Number
        dot(dx, pullback(f, backend, x, one(y), pullback_extras))
    elseif x isa Number && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dx * pullback(f, backend, x, basis(backend, y, i), pullback_extras)
        end
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dot(dx, pullback(f, backend, x, basis(backend, y, i), pullback_extras))
        end
    end
    return y, dy
end

function value_and_pushforward!(
    f::F,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x, dx),
) where {F}
    y, new_dy = value_and_pushforward(f, backend, x, dx, extras)
    return y, copyto!(dy, new_dy)
end

function pushforward(
    f::F,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x, dx),
) where {F}
    return value_and_pushforward(f, backend, x, dx, extras)[2]
end

function pushforward!(
    f::F,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x, dx),
) where {F}
    return value_and_pushforward!(f, dy, backend, x, dx, extras)[2]
end

## Two arguments

function value_and_pushforward(
    f!::F,
    y,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f!, y, backend, x, dx),
) where {F}
    return value_and_pushforward_twoarg_aux(f!, y, backend, x, dx, extras)
end

function value_and_pushforward_twoarg_aux(
    f!::F, y, backend, x, dx, extras::PullbackPushforwardExtras
) where {F}
    @compat (; pullback_extras) = extras
    dy = if x isa Number && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dx * pullback(f!, y, backend, x, basis(backend, y, i), pullback_extras)
        end
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(y)) do i
            dot(dx, pullback(f!, y, backend, x, basis(backend, y, i), pullback_extras))
        end
    end
    f!(y, x)
    return y, dy
end

function value_and_pushforward!(
    f!::F,
    y,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f!, y, backend, x, dx),
) where {F}
    y, new_dy = value_and_pushforward(f!, y, backend, x, dx, extras)
    return y, copyto!(dy, new_dy)
end

function pushforward(
    f!::F,
    y,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f!, y, backend, x, dx),
) where {F}
    return value_and_pushforward(f!, y, backend, x, dx, extras)[2]
end

function pushforward!(
    f!::F,
    y,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f!, y, backend, x, dx),
) where {F}
    return value_and_pushforward!(f!, y, dy, backend, x, dx, extras)[2]
end
