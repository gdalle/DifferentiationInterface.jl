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

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `value_and_jvp`.

!!! info
    Required primitive for forward mode backends.
"""
function value_and_pushforward end

"""
    value_and_pushforward!(f,     dy, backend, x, dx, [extras]) -> (y, dy)
    value_and_pushforward!(f!, y, dy, backend, x, dx, [extras]) -> (y, dy)

Compute the value and the pushforward of the function `f` at point `x` with seed `dx`, overwriting `dy`.

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `value_and_jvp!`.
"""
function value_and_pushforward! end

"""
    pushforward(f,     backend, x, dx, [extras]) -> dy
    pushforward(f!, y, backend, x, dx, [extras]) -> dy

Compute the pushforward of the function `f` at point `x` with seed `dx`.

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `jvp`.
"""
function pushforward end

"""
    pushforward!(f,     dy, backend, x, dx, [extras]) -> dy
    pushforward!(f!, y, dy, backend, x, dx, [extras]) -> dy

Compute the pushforward of the function `f` at point `x` with seed `dx`, overwriting `dy`.

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `jvp!`.
"""
function pushforward! end

## Preparation

struct PullbackPushforwardExtras{E} <: PushforwardExtras
    pullback_extras::E
end

function prepare_pushforward(f::F, backend::AbstractADType, x, tx::Tangents) where {F}
    return prepare_pushforward_aux(f, backend, x, tx, pushforward_performance(backend))
end

function prepare_pushforward(f!::F, y, backend::AbstractADType, x, tx::Tangents) where {F}
    return prepare_pushforward_aux(f!, y, backend, x, tx, pushforward_performance(backend))
end

function prepare_pushforward_aux(
    f::F, backend::AbstractADType, x, tx::Tangents, ::PushforwardSlow
) where {F}
    y = f(x)
    dy = y isa Number ? one(y) : basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f, backend, x, SingleTangent(dy))
    return PullbackPushforwardExtras(pullback_extras)
end

function prepare_pushforward_aux(
    f!::F, y, backend::AbstractADType, x, tx::Tangents, ::PushforwardSlow
) where {F}
    dy = y isa Number ? one(y) : basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f!, y, backend, x, SingleTangent(dy))
    return PullbackPushforwardExtras(pullback_extras)
end

function prepare_pushforward_aux(
    f, backend::AbstractADType, x, tx::Tangents, ::PushforwardFast
)
    throw(MissingBackendError(backend))
end

function prepare_pushforward_aux(
    f!, y, backend::AbstractADType, x, tx::Tangents, ::PushforwardFast
)
    throw(MissingBackendError(backend))
end

## One argument

function _pushforward_via_pullback(
    f::F, backend::AbstractADType, x, dx, pullback_extras::PullbackExtras, y::Number
) where {F}
    t1 = pullback(f, backend, x, SingleTangent(dx), pullback_extras)
    dy = dot(dx, only(t1))
    return dy
end

function _pushforward_via_pullback(
    f::F, backend::AbstractADType, x, dx, pullback_extras::PullbackExtras, y::AbstractArray
) where {F}
    dy = map(CartesianIndices(y)) do i
        t1 = pullback(f, backend, x, SingleTangent(basis(backend, y, i)), pullback_extras)
        dot(dx, only(t1))
    end
    return dy
end

function value_and_pushforward(
    f::F, backend::AbstractADType, x, tx::Tangents{B}, extras::PullbackPushforwardExtras
) where {F,B}
    @compat (; pullback_extras) = extras
    y = f(x)
    if B == 1
        dx = _pushforward_via_pullback(f, backend, x, only(tx), pullback_extras, y)
        return y, SingleTangent(dx)
    else
        dxs = ntuple(
            b -> _pushforward_via_pullback(f, backend, x, tx.d[b], pullback_extras, y),
            Val(B),
        )
        return y, Tangents(dxs)
    end
end

function value_and_pushforward!(
    f::F, ty::Tangents, backend::AbstractADType, x, tx::Tangents, extras::PushforwardExtras
) where {F}
    y, new_ty = value_and_pushforward(f, backend, x, tx, extras)
    return y, copyto!(ty, new_ty)
end

function pushforward(
    f::F, backend::AbstractADType, x, tx::Tangents, extras::PushforwardExtras
) where {F}
    return value_and_pushforward(f, backend, x, tx, extras)[2]
end

function pushforward!(
    f::F, ty::Tangents, backend::AbstractADType, x, tx::Tangents, extras::PushforwardExtras
) where {F}
    return value_and_pushforward!(f, ty, backend, x, tx, extras)[2]
end

## Two arguments

function _pushforward_via_pullback(
    f!::F, y::AbstractArray, backend::AbstractADType, x, dx, pullback_extras::PullbackExtras
) where {F}
    dy = map(CartesianIndices(y)) do i
        t1 = pullback(f!, y, backend, x, SingleTangent(basis(backend, y, i)), pullback_extras)
        dot(dx, only(t1))
    end
    return dy
end

function value_and_pushforward(
    f!::F, y, backend::AbstractADType, x, tx::Tangents{B}, extras::PullbackPushforwardExtras
) where {F,B}
    @compat (; pullback_extras) = extras
    if B == 1
        dy = _pushforward_via_pullback(f!, y, backend, x, only(tx), pullback_extras)
        f!(y, x)
        return y, SingleTangent(dy)
    else
        dys = ntuple(
            b -> _pushforward_via_pullback(f!, y, backend, x, tx.d[b], pullback_extras),
            Val(B),
        )
        f!(y, x)
        return y, Tangents(dys)
    end
end

function value_and_pushforward!(
    f!::F,
    y,
    ty::Tangents,
    backend::AbstractADType,
    x,
    tx::Tangents,
    extras::PushforwardExtras,
) where {F}
    y, new_ty = value_and_pushforward(f!, y, backend, x, tx, extras)
    return y, copyto!(ty, new_ty)
end

function pushforward(
    f!::F, y, backend::AbstractADType, x, tx::Tangents, extras::PushforwardExtras
) where {F}
    return value_and_pushforward(f!, y, backend, x, tx, extras)[2]
end

function pushforward!(
    f!::F,
    y,
    ty::Tangents,
    backend::AbstractADType,
    x,
    tx::Tangents,
    extras::PushforwardExtras,
) where {F}
    return value_and_pushforward!(f!, y, ty, backend, x, tx, extras)[2]
end

## Functors

struct PushforwardFixedSeed{F,B,TX,E}
    f::F
    backend::B
    tx::TX
    extras::E
end

function PushforwardFixedSeed(f, backend::AbstractADType, tx)
    return PushforwardFixedSeed(f, backend, tx, nothing)
end

function (pfs::PushforwardFixedSeed{F,B,TX,Nothing})(x) where {F,B,TX}
    @compat (; f, backend, tx) = pfs
    return pushforward(f, backend, x, tx)
end
