## Preparation

"""
    DerivativeExtras

Abstract type for additional information needed by derivative operators.
"""
abstract type DerivativeExtras <: Extras end

struct NoDerivativeExtras <: DerivativeExtras end

struct PushforwardDerivativeExtras{E<:PushforwardExtras} <: DerivativeExtras
    pushforward_extras::E
end

"""
    prepare_derivative([other_extras], f, backend, x) -> extras
    prepare_derivative([other_extras], f!, backend, y, x) -> extras

Create an `extras` object subtyping [`DerivativeExtras`](@ref) that can be given to derivative operators.
"""
function prepare_derivative(::Extras, f_or_f!, backend::AbstractADType, args...)
    return prepare_derivative(f_or_f!, backend, args...)
end

function prepare_derivative(f, backend::AbstractADType, x)
    return PushforwardDerivativeExtras(prepare_pushforward(f, backend, x))
end

function prepare_derivative(f!, backend::AbstractADType, y, x)
    return PushforwardDerivativeExtras(prepare_pushforward(f!, backend, y, x))
end

function prepare_pushforward(
    extras::PushforwardDerivativeExtras, f_or_f!, backend::AbstractADType, args...
)
    return extras.pushforward_extras
end

## Allocating

"""
    value_and_derivative(f, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative(
    f,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f, backend, x),
)
    new_extras = prepare_pushforward(extras, f, backend, x)
    return value_and_pushforward(f, backend, x, one(x), new_extras)
end

"""
    value_and_derivative!!(f, der, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative!!(
    f,
    der,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f, backend, x),
)
    new_extras = prepare_pushforward(extras, f, backend, x)
    return value_and_pushforward!!(f, der, backend, x, one(x), new_extras)
end

"""
    derivative(f, backend, x, [extras]) -> der
"""
function derivative(
    f,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f, backend, x),
)
    return value_and_derivative(f, backend, x, extras)[2]
end

"""
    derivative!!(f, der, backend, x, [extras]) -> der
"""
function derivative!!(
    f,
    der,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f, backend, x),
)
    return value_and_derivative!!(f, der, backend, x, extras)[2]
end

## Mutating

"""
    value_and_derivative!!(f!, y, der, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative!!(
    f!,
    y,
    der,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f!, backend, y, x),
)
    new_extras = prepare_pushforward(extras, f!, backend, y, x)
    return value_and_pushforward!!(f!, y, der, backend, x, one(x), new_extras)
end
