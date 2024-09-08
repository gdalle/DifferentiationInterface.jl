## Docstrings

"""
    prepare_pushforward(f,     backend, x, tx) -> extras
    prepare_pushforward(f!, y, backend, x, tx) -> extras

Create an `extras` object that can be given to [`pushforward`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_pushforward end

"""
    prepare_pushforward_same_point(f,     backend, x, tx) -> extras_same
    prepare_pushforward_same_point(f!, y, backend, x, tx) -> extras_same

Create an `extras_same` object that can be given to [`pushforward`](@ref) and its variants _if they are applied at the same point `x`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_pushforward_same_point end

"""
    value_and_pushforward(f,     [extras,] backend, x, tx) -> (y, ty)
    value_and_pushforward(f!, y, [extras,] backend, x, tx) -> (y, ty)

Compute the value and the pushforward of the function `f` at point `x` with tangent `tx` of type [`Tangents`](@ref).

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `value_and_jvp`.

!!! info
    Required primitive for forward mode backends.
"""
function value_and_pushforward end

"""
    value_and_pushforward!(f,     dy, [extras,] backend, x, tx) -> (y, ty)
    value_and_pushforward!(f!, y, dy, [extras,] backend, x, tx) -> (y, ty)

Compute the value and the pushforward of the function `f` at point `x` with tangent `tx` of type [`Tangents`](@ref), overwriting `ty`.

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `value_and_jvp!`.
"""
function value_and_pushforward! end

"""
    pushforward(f,     [extras,] backend, x, tx) -> ty
    pushforward(f!, y, [extras,] backend, x, tx) -> ty

Compute the pushforward of the function `f` at point `x` with tangent `tx` of type [`Tangents`](@ref).

$(document_preparation("pushforward"; same_point=true))

!!! tip 
    Pushforwards are also commonly called Jacobian-vector products or JVPs.
    This function could have been named `jvp`.
"""
function pushforward end

"""
    pushforward!(f,     dy, [extras,] backend, x, tx) -> ty
    pushforward!(f!, y, dy, [extras,] backend, x, tx) -> ty

Compute the pushforward of the function `f` at point `x` with tangent `tx` of type [`Tangents`](@ref), overwriting `ty`.

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
    return _prepare_pushforward_aux(f, backend, x, tx, pushforward_performance(backend))
end

function prepare_pushforward(f!::F, y, backend::AbstractADType, x, tx::Tangents) where {F}
    return _prepare_pushforward_aux(f!, y, backend, x, tx, pushforward_performance(backend))
end

function _prepare_pushforward_aux(
    f::F, backend::AbstractADType, x, tx::Tangents, ::PushforwardSlow
) where {F}
    y = f(x)
    dy = y isa Number ? one(y) : basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f, backend, x, Tangents(dy))
    return PullbackPushforwardExtras(pullback_extras)
end

function _prepare_pushforward_aux(
    f!::F, y, backend::AbstractADType, x, tx::Tangents, ::PushforwardSlow
) where {F}
    dy = y isa Number ? one(y) : basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f!, y, backend, x, Tangents(dy))
    return PullbackPushforwardExtras(pullback_extras)
end

function _prepare_pushforward_aux(
    f, backend::AbstractADType, x, tx::Tangents, ::PushforwardFast
)
    throw(MissingBackendError(backend))
end

function _prepare_pushforward_aux(
    f!, y, backend::AbstractADType, x, tx::Tangents, ::PushforwardFast
)
    throw(MissingBackendError(backend))
end

## One argument

function _pushforward_via_pullback(
    f::F, pullback_extras::PullbackExtras, backend::AbstractADType, x, dx, y::Number
) where {F}
    t1 = pullback(f, pullback_extras, backend, x, Tangents(one(y)))
    dy = dot(dx, only(t1))
    return dy
end

function _pushforward_via_pullback(
    f::F, pullback_extras::PullbackExtras, backend::AbstractADType, x, dx, y::AbstractArray
) where {F}
    dy = map(CartesianIndices(y)) do i
        t1 = pullback(f, pullback_extras, backend, x, Tangents(basis(backend, y, i)))
        dot(dx, only(t1))
    end
    return dy
end

function value_and_pushforward(
    f::F, extras::PullbackPushforwardExtras, backend::AbstractADType, x, tx::Tangents{B}
) where {F,B}
    @compat (; pullback_extras) = extras
    y = f(x)
    if B == 1
        dx = _pushforward_via_pullback(f, pullback_extras, backend, x, only(tx), y)
        return y, Tangents(dx)
    else
        dxs = ntuple(
            b -> _pushforward_via_pullback(f, pullback_extras, backend, x, tx.d[b], y),
            Val(B),
        )
        return y, Tangents(dxs...)
    end
end

function value_and_pushforward!(
    f::F, ty::Tangents, extras::PushforwardExtras, backend::AbstractADType, x, tx::Tangents
) where {F}
    y, new_ty = value_and_pushforward(f, extras, backend, x, tx)
    return y, copyto!(ty, new_ty)
end

function pushforward(
    f::F, extras::PushforwardExtras, backend::AbstractADType, x, tx::Tangents
) where {F}
    return value_and_pushforward(f, extras, backend, x, tx)[2]
end

function pushforward!(
    f::F, ty::Tangents, extras::PushforwardExtras, backend::AbstractADType, x, tx::Tangents
) where {F}
    return value_and_pushforward!(f, ty, extras, backend, x, tx)[2]
end

## Two arguments

function _pushforward_via_pullback(
    f!::F, y::AbstractArray, pullback_extras::PullbackExtras, backend::AbstractADType, x, dx
) where {F}
    dy = map(CartesianIndices(y)) do i  # preserve shape
        t1 = pullback(f!, y, pullback_extras, backend, x, Tangents(basis(backend, y, i)))
        dot(dx, only(t1))
    end
    return dy
end

function value_and_pushforward(
    f!::F, y, extras::PullbackPushforwardExtras, backend::AbstractADType, x, tx::Tangents{B}
) where {F,B}
    @compat (; pullback_extras) = extras
    if B == 1
        dy = _pushforward_via_pullback(f!, y, pullback_extras, backend, x, only(tx))
        f!(y, x)
        return y, Tangents(dy)
    else
        dys = ntuple(
            b -> _pushforward_via_pullback(f!, y, pullback_extras, backend, x, tx.d[b]),
            Val(B),
        )
        f!(y, x)
        return y, Tangents(dys...)
    end
end

function value_and_pushforward!(
    f!::F,
    y,
    ty::Tangents,
    extras::PushforwardExtras,
    backend::AbstractADType,
    x,
    tx::Tangents,
) where {F}
    y, new_ty = value_and_pushforward(f!, y, extras, backend, x, tx)
    return y, copyto!(ty, new_ty)
end

function pushforward(
    f!::F, y, extras::PushforwardExtras, backend::AbstractADType, x, tx::Tangents
) where {F}
    return value_and_pushforward(f!, y, extras, backend, x, tx)[2]
end

function pushforward!(
    f!::F,
    y,
    ty::Tangents,
    extras::PushforwardExtras,
    backend::AbstractADType,
    x,
    tx::Tangents,
) where {F}
    return value_and_pushforward!(f!, y, ty, extras, backend, x, tx)[2]
end
