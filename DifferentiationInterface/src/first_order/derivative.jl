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
"""
function value_and_derivative end

"""
    value_and_derivative!(f,     der, backend, x, [extras]) -> (y, der)
    value_and_derivative!(f!, y, der, backend, x, [extras]) -> (y, der)

Compute the value and the derivative of the function `f` at point `x`, overwriting `der`.
    """
function value_and_derivative! end

"""
    derivative(f,     backend, x, [extras]) -> der
    derivative(f!, y, backend, x, [extras]) -> der

Compute the derivative of the function `f` at point `x`.
"""
function derivative end

"""
    derivative!(f,     der, backend, x, [extras]) -> der
    derivative!(f!, y, der, backend, x, [extras]) -> der

Compute the derivative of the function `f` at point `x`, overwriting `der`.
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
    return PushforwardDerivativeExtras(prepare_pushforward(f, backend, x, dx))
end

function prepare_derivative(f!::F, y, backend::AbstractADType, x) where {F}
    dx = one(x)
    pushforward_extras = prepare_pushforward(f!, y, backend, x, dx)
    return PushforwardDerivativeExtras(pushforward_extras)
end

## One argument

function value_and_derivative(f::F, backend::AbstractADType, x) where {F}
    return value_and_derivative(f, backend, x, prepare_derivative(f, backend, x))
end

function value_and_derivative!(f::F, der, backend::AbstractADType, x) where {F}
    return value_and_derivative!(f, der, backend, x, prepare_derivative(f, backend, x))
end

function derivative(f::F, backend::AbstractADType, x) where {F}
    return derivative(f, backend, x, prepare_derivative(f, backend, x))
end

function derivative!(f::F, der, backend::AbstractADType, x) where {F}
    return derivative!(f, der, backend, x, prepare_derivative(f, backend, x))
end

function value_and_derivative(
    f::F, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    return value_and_pushforward(f, backend, x, one(x), extras.pushforward_extras)
end

function value_and_derivative!(
    f::F, der, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    return value_and_pushforward!(f, der, backend, x, one(x), extras.pushforward_extras)
end

function derivative(
    f::F, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    return pushforward(f, backend, x, one(x), extras.pushforward_extras)
end

function derivative!(
    f::F, der, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    return pushforward!(f, der, backend, x, one(x), extras.pushforward_extras)
end

## Two arguments

function value_and_derivative(f!::F, y, backend::AbstractADType, x) where {F}
    return value_and_derivative(f!, y, backend, x, prepare_derivative(f!, y, backend, x))
end

function value_and_derivative!(f!::F, y, der, backend::AbstractADType, x) where {F}
    return value_and_derivative!(
        f!, y, der, backend, x, prepare_derivative(f!, y, backend, x)
    )
end

function derivative(f!::F, y, backend::AbstractADType, x) where {F}
    return derivative(f!, y, backend, x, prepare_derivative(f!, y, backend, x))
end

function derivative!(f!::F, y, der, backend::AbstractADType, x) where {F}
    return derivative!(f!, y, der, backend, x, prepare_derivative(f!, y, backend, x))
end

function value_and_derivative(
    f!::F, y, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    return value_and_pushforward(f!, y, backend, x, one(x), extras.pushforward_extras)
end

function value_and_derivative!(
    f!::F, y, der, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    return value_and_pushforward!(f!, y, der, backend, x, one(x), extras.pushforward_extras)
end

function derivative(
    f!::F, y, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    return pushforward(f!, y, backend, x, one(x), extras.pushforward_extras)
end

function derivative!(
    f!::F, y, der, backend::AbstractADType, x, extras::PushforwardDerivativeExtras
) where {F}
    return pushforward!(f!, y, der, backend, x, one(x), extras.pushforward_extras)
end
