## Docstrings

"""
    prepare_derivative(f,     backend, x) -> extras
    prepare_derivative(f!, y, backend, x) -> extras

Create an `extras` object that can be given to [`derivative`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    In the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_derivative end

"""
    value_and_derivative(f,     backend, x, [extras]) -> (y, der)
    value_and_derivative(f!, y, backend, x, [extras]) -> (y, der)

Compute the value and the derivative of the function `f` at point `x`.

$(document_preparation("derivative"))
"""
function value_and_derivative end

"""
    value_and_derivative!(f,     der, backend, x, [extras]) -> (y, der)
    value_and_derivative!(f!, y, der, backend, x, [extras]) -> (y, der)

Compute the value and the derivative of the function `f` at point `x`, overwriting `der`.

$(document_preparation("derivative"))
"""
function value_and_derivative! end

"""
    derivative(f,     backend, x, [extras]) -> der
    derivative(f!, y, backend, x, [extras]) -> der

Compute the derivative of the function `f` at point `x`.

$(document_preparation("derivative"))
"""
function derivative end

"""
    derivative!(f,     der, backend, x, [extras]) -> der
    derivative!(f!, y, der, backend, x, [extras]) -> der

Compute the derivative of the function `f` at point `x`, overwriting `der`.

$(document_preparation("derivative"))
"""
function derivative! end

## Preparation

"""
    DerivativeExtras

Abstract type for additional information needed by [`derivative`](@ref) and its variants.
"""
abstract type DerivativeExtras <: Extras end

struct NoDerivativeExtras <: DerivativeExtras end

struct PushforwardDerivativeExtras{E<:PushforwardExtras} <: DerivativeExtras
    pushforward_extras::E
end

function prepare_derivative(f::F, backend::AbstractADType, x) where {F}
    dx = one(x)
    return PushforwardDerivativeExtras(prepare_pushforward(f, backend, x, Tangents(dx)))
end

function prepare_derivative(f!::F, y, backend::AbstractADType, x) where {F}
    dx = one(x)
    pushforward_extras = prepare_pushforward(f!, y, backend, x, Tangents(dx))
    return PushforwardDerivativeExtras(pushforward_extras)
end

## One argument

function value_and_derivative(
    f::F, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    y, ty = value_and_pushforward(
        f, backend, x, Tangents(one(x)), extras.pushforward_extras
    )
    return y, only(ty)
end

function value_and_derivative!(
    f::F, der, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    y, _ = value_and_pushforward!(
        f, Tangents(der), backend, x, Tangents(one(x)), extras.pushforward_extras
    )
    return y, der
end

function derivative(
    f::F, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    ty = pushforward(f, backend, x, Tangents(one(x)), extras.pushforward_extras)
    return only(ty)
end

function derivative!(
    f::F, der, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    pushforward!(f, Tangents(der), backend, x, Tangents(one(x)), extras.pushforward_extras)
    return der
end

## Two arguments

function value_and_derivative(
    f!::F, y, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    y, ty = value_and_pushforward(
        f!, y, backend, x, Tangents(one(x)), extras.pushforward_extras
    )
    return y, only(ty)
end

function value_and_derivative!(
    f!::F, y, der, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    y, _ = value_and_pushforward!(
        f!, y, Tangents(der), backend, x, Tangents(one(x)), extras.pushforward_extras
    )
    return y, der
end

function derivative(
    f!::F, y, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    ty = pushforward(f!, y, backend, x, Tangents(one(x)), extras.pushforward_extras)
    return only(ty)
end

function derivative!(
    f!::F, y, der, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    pushforward!(
        f!, y, Tangents(der), backend, x, Tangents(one(x)), extras.pushforward_extras
    )
    return der
end
