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

### Extras types

struct PullbackPushforwardExtras{E<:PullbackExtras} <: PushforwardExtras
    pullback_extras::E
end

function prepare_pushforward(f::F, backend::AbstractADType, x, dx) where {F}
    return _prepare_pushforward_aux(f, backend, x, dx, pushforward_performance(backend))
end

function prepare_pushforward(f!::F, y, backend::AbstractADType, x, dx) where {F}
    return _prepare_pushforward_aux(f!, y, backend, x, dx, pushforward_performance(backend))
end

function _prepare_pushforward_aux(
    f::F, backend::AbstractADType, x, dx, ::PushforwardSlow
) where {F}
    y = f(x)
    dy = y isa Number ? one(y) : basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f, backend, x, dy)
    return PullbackPushforwardExtras(pullback_extras)
end

function _prepare_pushforward_aux(
    f!::F, y, backend::AbstractADType, x, dx, ::PushforwardSlow
) where {F}
    dy = y isa Number ? one(y) : basis(backend, y, first(CartesianIndices(y)))
    pullback_extras = prepare_pullback(f!, y, backend, x, dy)
    return PullbackPushforwardExtras(pullback_extras)
end

function _prepare_pushforward_aux(f, backend::AbstractADType, x, dx, ::PushforwardFast)
    throw(MissingBackendError(backend))
end

function _prepare_pushforward_aux(f!, y, backend::AbstractADType, x, dx, ::PushforwardFast)
    throw(MissingBackendError(backend))
end

## One argument

function _pushforward_via_pullback(
    f::F, backend::AbstractADType, x, dx, pullback_extras::PullbackExtras, y::Number
) where {F}
    dy = dot(dx, pullback(f, backend, x, one(y), pullback_extras))
    return dy
end

function _pushforward_via_pullback(
    f::F, backend::AbstractADType, x, dx, pullback_extras::PullbackExtras, y::AbstractArray
) where {F}
    dy = map(CartesianIndices(y)) do i
        dot(dx, pullback(f, backend, x, basis(backend, y, i), pullback_extras))
    end
    return dy
end

function value_and_pushforward(
    f::F, backend::AbstractADType, x, dx, extras::PullbackPushforwardExtras
) where {F}
    @compat (; pullback_extras) = extras
    y = f(x)
    dy = _pushforward_via_pullback(f, backend, x, dx, pullback_extras, y)
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

## Two arguments

function _pushforward_via_pullback(
    f!::F, y::AbstractArray, backend::AbstractADType, x, dx, pullback_extras::PullbackExtras
) where {F}
    dy = map(CartesianIndices(y)) do i
        dot(dx, pullback(f!, y, backend, x, basis(backend, y, i), pullback_extras))
    end
    return dy
end

function value_and_pushforward(
    f!::F, y, backend::AbstractADType, x, dx, extras::PullbackPushforwardExtras
) where {F}
    @compat (; pullback_extras) = extras
    dy = _pushforward_via_pullback(f!, y, backend, x, dx, pullback_extras)
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

## Functors

struct PushforwardFixedSeed{F,B,DX,E}
    f::F
    backend::B
    dx::DX
    extras::E
end

function PushforwardFixedSeed(f, backend::AbstractADType, dx)
    return PushforwardFixedSeed(f, backend, dx, nothing)
end

# not covered but don't remove, Enzyme messes with code coverage
function (pfs::PushforwardFixedSeed{F,B,DX,Nothing})(x) where {F,B,DX}
    @compat (; f, backend, dx) = pfs
    return pushforward(f, backend, x, dx)
end
