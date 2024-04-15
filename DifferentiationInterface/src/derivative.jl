## Docstrings

"""
    prepare_derivative(f,     backend, x) -> extras
    prepare_derivative(f!, y, backend, x) -> extras

Create an `extras` object subtyping [`DerivativeExtras`](@ref) that can be given to derivative operators.

Beware that in the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_derivative end

"""
    value_and_derivative(f,     backend, x, [extras]) -> (y, der)
    value_and_derivative(f!, y, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative end

"""
    value_and_derivative!(f,     der, backend, x, [extras]) -> (y, der)
    value_and_derivative!(f!, y, der, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative! end

"""
    derivative(f,     backend, x, [extras]) -> der
    derivative(f!, y, backend, x, [extras]) -> der
"""
function derivative end

"""
    derivative!(f,     der, backend, x, [extras]) -> der
    derivative!(f!, y, der, backend, x, [extras]) -> der
"""
function derivative! end

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

function prepare_derivative(f, backend::AbstractADType, x)
    return PushforwardDerivativeExtras(prepare_pushforward(f, backend, x))
end

function prepare_derivative(f!, y, backend::AbstractADType, x)
    return PushforwardDerivativeExtras(prepare_pushforward(f!, y, backend, x))
end

## One argument

function value_and_derivative(
    f,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f, backend, x),
)
    return value_and_pushforward(f, backend, x, one(x), extras.pushforward_extras)
end

function value_and_derivative!(
    f,
    der,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f, backend, x),
)
    return value_and_pushforward!(f, der, backend, x, one(x), extras.pushforward_extras)
end

function derivative(
    f,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f, backend, x),
)
    return pushforward(f, backend, x, one(x), extras.pushforward_extras)
end

function derivative!(
    f,
    der,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f, backend, x),
)
    return pushforward!(f, der, backend, x, one(x), extras.pushforward_extras)
end

## Two arguments

function value_and_derivative(
    f!,
    y,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f!, y, backend, x),
)
    return value_and_pushforward(f!, y, backend, x, one(x), extras.pushforward_extras)
end

function value_and_derivative!(
    f!,
    y,
    der,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f!, y, backend, x),
)
    return value_and_pushforward!(f!, y, der, backend, x, one(x), extras.pushforward_extras)
end

function derivative(
    f!,
    y,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f!, y, backend, x),
)
    return pushforward(f!, y, backend, x, one(x), extras.pushforward_extras)
end

function derivative!(
    f!,
    y,
    der,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f!, y, backend, x),
)
    return pushforward!(f!, y, der, backend, x, one(x), extras.pushforward_extras)
end
