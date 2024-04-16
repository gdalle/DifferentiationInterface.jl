## Docstrings

"""
    prepare_pushforward(f,     backend, x, dx) -> extras
    prepare_pushforward(f!, y, backend, x, dx) -> extras

Create an `extras` object subtyping [`PushforwardExtras`](@ref) that can be given to pushforward operators.

Beware that in the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_pushforward end

"""
    value_and_pushforward(f,     backend, x, dx, [extras]) -> (y, dy)
    value_and_pushforward(f!, y, backend, x, dx, [extras]) -> (y, dy)

!!! info
    Required primitive for forward mode backends.
"""
function value_and_pushforward end

"""
    value_and_pushforward!(f,     dy, backend, x, dx, [extras]) -> (y, dy)
    value_and_pushforward!(f!, y, dy, backend, x, dx, [extras]) -> (y, dy)
"""
function value_and_pushforward! end

"""
    pushforward(f,     backend, x, dx, [extras]) -> dy
    pushforward(f!, y, backend, x, dx, [extras]) -> dy
"""
function pushforward end

"""
    pushforward!(f,     dy, backend, x, dx, [extras]) -> dy
    pushforward!(f!, y, dy, backend, x, dx, [extras]) -> dy
"""
function pushforward! end

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

function prepare_pushforward(f, backend::AbstractADType, x, dx)
    return prepare_pushforward_aux(f, backend, x, dx, pushforward_performance(backend))
end

function prepare_pushforward(f!, y, backend::AbstractADType, x, dx)
    return prepare_pushforward_aux(f!, y, backend, x, dx, pushforward_performance(backend))
end

function prepare_pushforward_aux(f, backend, x, dx, ::PushforwardSlow)
    y = f(x)
    dy = y isa Number ? one(y) : basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f, backend, x, dy)
    return PullbackPushforwardExtras(pullback_extras)
end

function prepare_pushforward_aux(f!, y, backend, x, dx, ::PushforwardSlow)
    dy = y isa Number ? one(y) : basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f!, y, backend, x, dy)
    return PullbackPushforwardExtras(pullback_extras)
end

## One argument

function value_and_pushforward(
    f,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x, dx),
)
    return value_and_pushforward_onearg_aux(f, backend, x, dx, extras)
end

function value_and_pushforward_onearg_aux(
    f, backend, x, dx, extras::PullbackPushforwardExtras
)
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

function value_and_pushforward!(
    f,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x, dx),
)
    y, new_dy = value_and_pushforward(f, backend, x, dx, extras)
    return y, copyto!(dy, new_dy)
end

function pushforward(
    f,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x, dx),
)
    return value_and_pushforward(f, backend, x, dx, extras)[2]
end

function pushforward!(
    f,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f, backend, x, dx),
)
    return value_and_pushforward!(f, dy, backend, x, dx, extras)[2]
end

## Two arguments

function value_and_pushforward(
    f!,
    y,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f!, y, backend, x, dx),
)
    return value_and_pushforward_twoarg_aux(f!, y, backend, x, dx, extras)
end

function value_and_pushforward_twoarg_aux(
    f!, y, backend, x, dx, extras::PullbackPushforwardExtras
)
    (; pullback_extras) = extras
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
    f!,
    y,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f!, y, backend, x, dx),
)
    y, new_dy = value_and_pushforward(f!, y, backend, x, dx, extras)
    return y, copyto!(dy, new_dy)
end

function pushforward(
    f!,
    y,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f!, y, backend, x, dx),
)
    return value_and_pushforward(f!, y, backend, x, dx, extras)[2]
end

function pushforward!(
    f!,
    y,
    dy,
    backend::AbstractADType,
    x,
    dx,
    extras::PushforwardExtras=prepare_pushforward(f!, y, backend, x, dx),
)
    return value_and_pushforward!(f!, y, dy, backend, x, dx, extras)[2]
end
