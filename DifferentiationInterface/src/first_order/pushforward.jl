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

function prepare_pushforward_batched end

"""
    prepare_pushforward_same_point(f,     backend, x, dx) -> extras_same
    prepare_pushforward_same_point(f!, y, backend, x, dx) -> extras_same

Create an `extras_same` object that can be given to [`pushforward`](@ref) and its variants _if they are applied at the same point `x`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    In the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_pushforward_same_point end

function prepare_pushforward_batched_same_point end

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

function pushforward_batched end

"""
    pushforward!(f,     dy, backend, x, dx, [extras]) -> dy
    pushforward!(f!, y, dy, backend, x, dx, [extras]) -> dy

Compute the pushforward of the function `f` at point `x` with seed `dx`, overwriting `dy`.
"""
function pushforward! end

function pushforward_batched! end

## Preparation

### Extras types

"""
    PushforwardExtras

Abstract type for additional information needed by [`pushforward`](@ref) and its variants.
"""
abstract type PushforwardExtras <: Extras end

struct NoPushforwardExtras <: PushforwardExtras end

struct PullbackPushforwardExtras{E} <: PushforwardExtras
    pullback_extras::E
end

### Standard

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

function prepare_pushforward_aux(f, backend, x, dy, ::PushforwardFast)
    throw(MissingBackendError(backend))
end

function prepare_pushforward_aux(f!, y, backend, x, dy, ::PushforwardFast)
    throw(MissingBackendError(backend))
end

### Standard, same point

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

### Batched

function prepare_pushforward_batched(f::F, backend::AbstractADType, x, dx::Batch) where {F}
    return prepare_pushforward(f, backend, x, first(dx.elements))
end

function prepare_pushforward_batched(
    f!::F, y, backend::AbstractADType, x, dx::Batch
) where {F}
    return prepare_pushforward(f!, y, backend, x, first(dx.elements))
end

function prepare_pushforward_batched_same_point(
    f::F, backend::AbstractADType, x, dx::Batch
) where {F}
    extras = prepare_pushforward_batched(f, backend, x, dx)
    return prepare_pushforward_batched_same_point(f, backend, x, dx, extras)
end

function prepare_pushforward_batched_same_point(
    f!::F, y, backend::AbstractADType, x, dx::Batch
) where {F}
    extras = prepare_pushforward_batched(f!, y, backend, x, dx)
    return prepare_pushforward_batched_same_point(f!, y, backend, x, dx, extras)
end

### Batched, same point

function prepare_pushforward_batched_same_point(
    f::F, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    return prepare_pushforward_same_point(f, backend, x, first(dx.elements), extras)
end

function prepare_pushforward_batched_same_point(
    f!::F, y, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    return prepare_pushforward_same_point(f!, y, backend, x, first(dx.elements), extras)
end

## One argument

### Standard

function value_and_pushforward(f::F, backend::AbstractADType, x, dx) where {F}
    return value_and_pushforward(f, backend, x, dx, prepare_pushforward(f, backend, x, dx))
end

function value_and_pushforward!(f::F, dy, backend::AbstractADType, x, dx) where {F}
    return value_and_pushforward!(
        f, dy, backend, x, dx, prepare_pushforward(f, backend, x, dx)
    )
end

function pushforward(f::F, backend::AbstractADType, x, dx) where {F}
    return pushforward(f, backend, x, dx, prepare_pushforward(f, backend, x, dx))
end

function pushforward!(f::F, dy, backend::AbstractADType, x, dx) where {F}
    return pushforward!(f, dy, backend, x, dx, prepare_pushforward(f, backend, x, dx))
end

function value_and_pushforward(
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
    f::F, dy, backend::AbstractADType, x, dx, extras::PushforwardExtras
) where {F}
    y, new_dy = value_and_pushforward(f, backend, x, dx, extras)
    return y, copyto!(dy, new_dy)
end

function pushforward(
    f::F, backend::AbstractADType, x, dx, extras::PushforwardExtras
) where {F}
    return value_and_pushforward(f, backend, x, dx, extras)[2]
end

function pushforward!(
    f::F, dy, backend::AbstractADType, x, dx, extras::PushforwardExtras
) where {F}
    return value_and_pushforward!(f, dy, backend, x, dx, extras)[2]
end

### Batched

function pushforward_batched(
    f::F, backend::AbstractADType, x, dx::Batch{B}, extras::PushforwardExtras
) where {F,B}
    dy_elements = ntuple(Val(B)) do b
        pushforward(f, backend, x, dx.elements[b], extras)
    end
    return Batch(dy_elements)
end

function pushforward_batched!(
    f::F, dy::Batch, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    for b in eachindex(dy.elements, dx.elements)
        pushforward!(f, dy.elements[b], backend, x, dx.elements[b], extras)
    end
    return dy
end

## Two arguments

### Standard

function value_and_pushforward(f!::F, y, backend::AbstractADType, x, dx) where {F}
    return value_and_pushforward(
        f!, y, backend, x, dx, prepare_pushforward(f!, y, backend, x, dx)
    )
end

function value_and_pushforward!(f!::F, y, dy, backend::AbstractADType, x, dx) where {F}
    return value_and_pushforward!(
        f!, y, dy, backend, x, dx, prepare_pushforward(f!, y, backend, x, dx)
    )
end

function pushforward(f!::F, y, backend::AbstractADType, x, dx) where {F}
    return pushforward(f!, y, backend, x, dx, prepare_pushforward(f!, y, backend, x, dx))
end

function pushforward!(f!::F, y, dy, backend::AbstractADType, x, dx) where {F}
    return pushforward!(
        f!, y, dy, backend, x, dx, prepare_pushforward(f!, y, backend, x, dx)
    )
end

function value_and_pushforward(
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
    f!::F, y, dy, backend::AbstractADType, x, dx, extras::PushforwardExtras
) where {F}
    y, new_dy = value_and_pushforward(f!, y, backend, x, dx, extras)
    return y, copyto!(dy, new_dy)
end

function pushforward(
    f!::F, y, backend::AbstractADType, x, dx, extras::PushforwardExtras
) where {F}
    return value_and_pushforward(f!, y, backend, x, dx, extras)[2]
end

function pushforward!(
    f!::F, y, dy, backend::AbstractADType, x, dx, extras::PushforwardExtras
) where {F}
    return value_and_pushforward!(f!, y, dy, backend, x, dx, extras)[2]
end

### Batched

function pushforward_batched(
    f!::F, y, backend::AbstractADType, x, dx::Batch{B}, extras::PushforwardExtras
) where {F,B}
    dy_elements = ntuple(Val(B)) do b
        pushforward(f!, y, backend, x, dx.elements[b], extras)
    end
    return Batch(dy_elements)
end

function pushforward_batched!(
    f!::F, y, dy::Batch, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    for b in eachindex(dy.elements, dx.elements)
        pushforward!(f!, y, dy.elements[b], backend, x, dx.elements[b], extras)
    end
    return dy
end
