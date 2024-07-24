## Docstrings

"""
    prepare_second_derivative(f, backend, x) -> extras

Create an `extras` object that can be given to [`second_derivative`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_second_derivative end

"""
    second_derivative(f, backend, x, [extras]) -> der2

Compute the second derivative of the function `f` at point `x`.

$(document_preparation("second_derivative"))
"""
function second_derivative end

"""
    second_derivative!(f, der2, backend, x, [extras]) -> der2

Compute the second derivative of the function `f` at point `x`, overwriting `der2`.

$(document_preparation("second_derivative"))
"""
function second_derivative! end

"""
    value_derivative_and_second_derivative(f, backend, x, [extras]) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`.

$(document_preparation("second_derivative"))
"""
function value_derivative_and_second_derivative end

"""
    value_derivative_and_second_derivative!(f, der, der2, backend, x, [extras]) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`, overwriting `der` and `der2`.

$(document_preparation("second_derivative"))
"""
function value_derivative_and_second_derivative! end

## Preparation

"""
    SecondDerivativeExtras

Abstract type for additional information needed by [`second_derivative`](@ref) and its variants.
"""
abstract type SecondDerivativeExtras <: Extras end

struct NoSecondDerivativeExtras <: SecondDerivativeExtras end

struct InnerDerivative{F,B}
    f::F
    backend::B
end

function (id::InnerDerivative)(x)
    @compat (; f, backend) = id
    return derivative(f, backend, x)
end

struct ClosureSecondDerivativeExtras{ID<:InnerDerivative,E<:DerivativeExtras} <:
       SecondDerivativeExtras
    inner_derivative::ID
    outer_derivative_extras::E
end

function prepare_second_derivative(f::F, backend::AbstractADType, x) where {F}
    return prepare_second_derivative(f, SecondOrder(backend, backend), x)
end

function prepare_second_derivative(f::F, backend::SecondOrder, x) where {F}
    inner_derivative = InnerDerivative(f, nested(inner(backend)))
    outer_derivative_extras = prepare_derivative(inner_derivative, outer(backend), x)
    return ClosureSecondDerivativeExtras(inner_derivative, outer_derivative_extras)
end

## One argument

### Without extras

function value_derivative_and_second_derivative(f::F, backend::AbstractADType, x) where {F}
    return value_derivative_and_second_derivative(
        f, backend, x, prepare_second_derivative(f, backend, x)
    )
end

function value_derivative_and_second_derivative!(
    f::F, der, der2, backend::AbstractADType, x
) where {F}
    return value_derivative_and_second_derivative!(
        f, der, der2, backend, x, prepare_second_derivative(f, backend, x)
    )
end

function second_derivative(f::F, backend::AbstractADType, x) where {F}
    return second_derivative(f, backend, x, prepare_second_derivative(f, backend, x))
end

function second_derivative!(f::F, der2, backend::AbstractADType, x) where {F}
    return second_derivative!(f, der2, backend, x, prepare_second_derivative(f, backend, x))
end

### With extras

function second_derivative(
    f::F, backend::AbstractADType, x, extras::SecondDerivativeExtras
) where {F}
    return second_derivative(f, SecondOrder(backend, backend), x, extras)
end

function second_derivative(
    f::F, backend::SecondOrder, x, extras::ClosureSecondDerivativeExtras
) where {F}
    @compat (; inner_derivative, outer_derivative_extras) = extras
    return derivative(inner_derivative, outer(backend), x, outer_derivative_extras)
end

function value_derivative_and_second_derivative(
    f::F, backend::AbstractADType, x, extras::SecondDerivativeExtras
) where {F}
    return value_derivative_and_second_derivative(
        f, SecondOrder(backend, backend), x, extras
    )
end

function value_derivative_and_second_derivative(
    f::F, backend::SecondOrder, x, extras::ClosureSecondDerivativeExtras
) where {F}
    @compat (; inner_derivative, outer_derivative_extras) = extras
    y = f(x)
    der, der2 = value_and_derivative(
        inner_derivative, outer(backend), x, outer_derivative_extras
    )
    return y, der, der2
end

function second_derivative!(
    f::F, der2, backend::AbstractADType, x, extras::SecondDerivativeExtras
) where {F}
    return second_derivative!(f, der2, SecondOrder(backend, backend), x, extras)
end

function second_derivative!(
    f::F, der2, backend::SecondOrder, x, extras::SecondDerivativeExtras
) where {F}
    @compat (; inner_derivative, outer_derivative_extras) = extras
    return derivative!(inner_derivative, der2, outer(backend), x, outer_derivative_extras)
end

function value_derivative_and_second_derivative!(
    f::F, der, der2, backend::AbstractADType, x, extras::SecondDerivativeExtras
) where {F}
    return value_derivative_and_second_derivative!(
        f, der, der2, SecondOrder(backend, backend), x, extras
    )
end

function value_derivative_and_second_derivative!(
    f::F, der, der2, backend::SecondOrder, x, extras::SecondDerivativeExtras
) where {F}
    @compat (; inner_derivative, outer_derivative_extras) = extras
    y = f(x)
    new_der, _ = value_and_derivative!(
        inner_derivative, der2, outer(backend), x, outer_derivative_extras
    )
    return y, copyto!(der, new_der), der2
end
