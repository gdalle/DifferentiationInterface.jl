## Docstrings

"""
    prepare_pullback(f,     backend, x, dy) -> extras
    prepare_pullback(f!, y, backend, x, dy) -> extras

Create an `extras` object that can be given to [`pullback`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    In the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_pullback end

"""
    prepare_pullback_same_point(f,     backend, x, dy) -> extras_same
    prepare_pullback_same_point(f!, y, backend, x, dy) -> extras_same

Create an `extras_same` object that can be given to [`pullback`](@ref) and its variants _if they are applied at the same point `x`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    In the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_pullback_same_point end

"""
    value_and_pullback(f,     backend, x, dy, [extras]) -> (y, dx)
    value_and_pullback(f!, y, backend, x, dy, [extras]) -> (y, dx)

Compute the value and the pullback of the function `f` at point `x` with seed `dy`.

$(document_preparation("pullback"; same_point=true))

!!! tip 
    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `value_and_vjp`.

!!! info
    Required primitive for reverse mode backends.
"""
function value_and_pullback end

"""
    value_and_pullback!(f,     dx, backend, x, dy, [extras]) -> (y, dx)
    value_and_pullback!(f!, y, dx, backend, x, dy, [extras]) -> (y, dx)

Compute the value and the pullback of the function `f` at point `x` with seed `dy`, overwriting `dx`.

$(document_preparation("pullback"; same_point=true))

!!! tip 
    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `value_and_vjp!`.
"""
function value_and_pullback! end

"""
    pullback(f,     backend, x, dy, [extras]) -> dx
    pullback(f!, y, backend, x, dy, [extras]) -> dx

Compute the pullback of the function `f` at point `x` with seed `dy`.

$(document_preparation("pullback"; same_point=true))

!!! tip 
    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `vjp`.
"""
function pullback end

"""
    pullback!(f,     dx, backend, x, dy, [extras]) -> dx
    pullback!(f!, y, dx, backend, x, dy, [extras]) -> dx

Compute the pullback of the function `f` at point `x` with seed `dy`, overwriting `dx`.

$(document_preparation("pullback"; same_point=true))

!!! tip 
    Pullbacks are also commonly called vector-Jacobian products or VJPs.
    This function could have been named `vjp!`.
"""
function pullback! end

## Preparation

### Extras types

struct PushforwardPullbackExtras{E<:PushforwardExtras} <: PullbackExtras
    pushforward_extras::E
end

## Different point

function prepare_pullback(f::F, backend::AbstractADType, x, dy) where {F}
    return _prepare_pullback_aux(f, backend, x, dy, pullback_performance(backend))
end

function prepare_pullback(f!::F, y, backend::AbstractADType, x, dy) where {F}
    return _prepare_pullback_aux(f!, y, backend, x, dy, pullback_performance(backend))
end

function _prepare_pullback_aux(
    f::F, backend::AbstractADType, x, dy, ::PullbackSlow
) where {F}
    dx = x isa Number ? one(x) : basis(backend, x, first(CartesianIndices(x)))
    pushforward_extras = prepare_pushforward(f, backend, x, dx)
    return PushforwardPullbackExtras(pushforward_extras)
end

function _prepare_pullback_aux(
    f!::F, y, backend::AbstractADType, x, dy, ::PullbackSlow
) where {F}
    dx = x isa Number ? one(x) : basis(backend, x, first(CartesianIndices(x)))
    pushforward_extras = prepare_pushforward(f!, y, backend, x, dx)
    return PushforwardPullbackExtras(pushforward_extras)
end

function _prepare_pullback_aux(f, backend::AbstractADType, x, dy, ::PullbackFast)
    throw(MissingBackendError(backend))
end

function _prepare_pullback_aux(f!, y, backend::AbstractADType, x, dy, ::PullbackFast)
    throw(MissingBackendError(backend))
end

### Same point

function prepare_pullback_same_point(
    f::F, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    return extras
end

function prepare_pullback_same_point(
    f!::F, y, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    return extras
end

function prepare_pullback_same_point(f::F, backend::AbstractADType, x, dy) where {F}
    extras = prepare_pullback(f, backend, x, dy)
    return prepare_pullback_same_point(f, backend, x, dy, extras)
end

function prepare_pullback_same_point(f!::F, y, backend::AbstractADType, x, dy) where {F}
    extras = prepare_pullback(f!, y, backend, x, dy)
    return prepare_pullback_same_point(f!, y, backend, x, dy, extras)
end

## One argument

function _pullback_via_pushforward(
    f::F, backend::AbstractADType, x::Number, dy, pushforward_extras::PushforwardExtras
) where {F}
    dx = dot(dy, pushforward(f, backend, x, one(x), pushforward_extras))
    return dx
end

function _pullback_via_pushforward(
    f::F,
    backend::AbstractADType,
    x::AbstractArray,
    dy,
    pushforward_extras::PushforwardExtras,
) where {F}
    dx = map(CartesianIndices(x)) do j
        dot(dy, pushforward(f, backend, x, basis(backend, x, j), pushforward_extras))
    end
    return dx
end

function value_and_pullback(
    f::F, backend::AbstractADType, x, dy, extras::PushforwardPullbackExtras
) where {F}
    @compat (; pushforward_extras) = extras
    y = f(x)
    dx = _pullback_via_pushforward(f, backend, x, dy, pushforward_extras)
    return y, dx
end

function value_and_pullback!(
    f::F, dx, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    y, new_dx = value_and_pullback(f, backend, x, dy, extras)
    return y, copyto!(dx, new_dx)
end

function pullback(f::F, backend::AbstractADType, x, dy, extras::PullbackExtras) where {F}
    return value_and_pullback(f, backend, x, dy, extras)[2]
end

function pullback!(
    f::F, dx, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    return value_and_pullback!(f, dx, backend, x, dy, extras)[2]
end

## Two arguments

function _pullback_via_pushforward(
    f!::F, y, backend::AbstractADType, x::Number, dy, pushforward_extras::PushforwardExtras
) where {F}
    dx = dot(dy, pushforward(f!, y, backend, x, one(x), pushforward_extras))
    return dx
end

function _pullback_via_pushforward(
    f!::F,
    y,
    backend::AbstractADType,
    x::AbstractArray,
    dy,
    pushforward_extras::PushforwardExtras,
) where {F}
    dx = map(CartesianIndices(x)) do j
        dot(dy, pushforward(f!, y, backend, x, basis(backend, x, j), pushforward_extras))
    end
    return dx
end

function value_and_pullback(
    f!::F, y, backend::AbstractADType, x, dy, extras::PushforwardPullbackExtras
) where {F}
    @compat (; pushforward_extras) = extras
    dx = _pullback_via_pushforward(f!, y, backend, x, dy, pushforward_extras)
    f!(y, x)
    return y, dx
end

function value_and_pullback!(
    f!::F, y, dx, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    y, new_dx = value_and_pullback(f!, y, backend, x, dy, extras)
    return y, copyto!(dx, new_dx)
end

function pullback(
    f!::F, y, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    return value_and_pullback(f!, y, backend, x, dy, extras)[2]
end

function pullback!(
    f!::F, y, dx, backend::AbstractADType, x, dy, extras::PullbackExtras
) where {F}
    return value_and_pullback!(f!, y, dx, backend, x, dy, extras)[2]
end
