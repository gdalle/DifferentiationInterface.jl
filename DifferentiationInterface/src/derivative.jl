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
    prepare_derivative(f, backend, x) -> extras
    prepare_derivative(f!, backend, y, x) -> extras

Create an `extras` object subtyping [`DerivativeExtras`](@ref) that can be given to derivative operators.
"""
function prepare_derivative(f, backend::AbstractADType, x)
    return PushforwardDerivativeExtras(prepare_pushforward(f, backend, x))
end

function prepare_derivative(f!, backend::AbstractADType, y, x)
    return PushforwardDerivativeExtras(prepare_pushforward(f!, backend, y, x))
end

## One argument

"""
    value_and_derivative(f, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative(
    f,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f, backend, x),
)
    return value_and_pushforward(f, backend, x, one(x), extras.pushforward_extras)
end

"""
    value_and_derivative!(f, der, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative!(
    f,
    der,
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f, backend, x),
)
    return value_and_pushforward!(f, der, backend, x, one(x), extras.pushforward_extras)
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
    return pushforward(f, backend, x, one(x), extras.pushforward_extras)
end

"""
    derivative!(f, der, backend, x, [extras]) -> der
"""
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

"""
    value_and_derivative!(f!, (y, der), backend, x, [extras]) -> (y, der)
"""
function value_and_derivative!(
    f!,
    y_and_der::Tuple{<:Any,<:Any},
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f!, backend, y_and_der[1], x),
)
    return value_and_pushforward!(
        f!, y_and_der, backend, x, one(x), extras.pushforward_extras
    )
end

"""
    derivative!(f!, (y, der), backend, x, [extras]) -> der
"""
function derivative!(
    f!,
    y_and_der::Tuple{<:Any,<:Any},
    backend::AbstractADType,
    x,
    extras::DerivativeExtras=prepare_derivative(f!, backend, y_and_der[1], x),
)
    return pushforward!(f!, y_and_der, backend, x, one(x), extras.pushforward_extras)
end
