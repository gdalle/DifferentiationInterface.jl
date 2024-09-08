## Docstrings

"""
    prepare_second_derivative(f, backend, x) -> extras

Create an `extras` object that can be given to [`second_derivative`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_second_derivative end

"""
    second_derivative(f, [extras,] backend, x) -> der2

Compute the second derivative of the function `f` at point `x`.

$(document_preparation("second_derivative"))
"""
function second_derivative end

"""
    second_derivative!(f, der2, [extras,] backend, x) -> der2

Compute the second derivative of the function `f` at point `x`, overwriting `der2`.

$(document_preparation("second_derivative"))
"""
function second_derivative! end

"""
    value_derivative_and_second_derivative(f, [extras,] backend, x) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`.

$(document_preparation("second_derivative"))
"""
function value_derivative_and_second_derivative end

"""
    value_derivative_and_second_derivative!(f, der, der2, [extras,] backend, x) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`, overwriting `der` and `der2`.

$(document_preparation("second_derivative"))
"""
function value_derivative_and_second_derivative! end

## Preparation

struct InnerDerivative{F,B,C}
    f::F
    backend::B
    contexts::C
end

function (id::InnerDerivative)(x)
    @compat (; f, backend, contexts) = id
    return derivative(f, backend, x, contexts...)
end

struct ClosureSecondDerivativeExtras{ID<:InnerDerivative,E<:DerivativeExtras} <:
       SecondDerivativeExtras
    inner_derivative::ID
    outer_derivative_extras::E
end

function prepare_second_derivative(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return prepare_second_derivative(f, SecondOrder(backend, backend), x, contexts...)
end

function prepare_second_derivative(
    f::F, backend::SecondOrder, x, contexts::Vararg{Context,C}
) where {F,C}
    inner_derivative = InnerDerivative(f, nested(inner(backend)), contexts)
    outer_derivative_extras = prepare_derivative(
        inner_derivative, outer(backend), x, contexts...
    )
    return ClosureSecondDerivativeExtras(inner_derivative, outer_derivative_extras)
end

## One argument

function second_derivative(
    f::F,
    extras::SecondDerivativeExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return second_derivative(f, extras, SecondOrder(backend, backend), x, contexts...)
end

function second_derivative(
    f::F,
    extras::ClosureSecondDerivativeExtras,
    backend::SecondOrder,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_derivative, outer_derivative_extras) = extras
    return derivative(
        inner_derivative, outer_derivative_extras, outer(backend), x, contexts...
    )
end

function value_derivative_and_second_derivative(
    f::F,
    extras::SecondDerivativeExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_derivative_and_second_derivative(
        f, extras, SecondOrder(backend, backend), x, contexts...
    )
end

function value_derivative_and_second_derivative(
    f::F,
    extras::ClosureSecondDerivativeExtras,
    backend::SecondOrder,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_derivative, outer_derivative_extras) = extras
    y = f(x)
    der, der2 = value_and_derivative(
        inner_derivative, outer_derivative_extras, outer(backend), x, contexts...
    )
    return y, der, der2
end

function second_derivative!(
    f::F,
    der2,
    extras::SecondDerivativeExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return second_derivative!(
        f, der2, extras, SecondOrder(backend, backend), x, contexts...
    )
end

function second_derivative!(
    f::F,
    der2,
    extras::SecondDerivativeExtras,
    backend::SecondOrder,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_derivative, outer_derivative_extras) = extras
    return derivative!(
        inner_derivative, der2, outer_derivative_extras, outer(backend), x, contexts...
    )
end

function value_derivative_and_second_derivative!(
    f::F,
    der,
    der2,
    extras::SecondDerivativeExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return value_derivative_and_second_derivative!(
        f, der, der2, extras, SecondOrder(backend, backend), x, contexts...
    )
end

function value_derivative_and_second_derivative!(
    f::F,
    der,
    der2,
    extras::SecondDerivativeExtras,
    backend::SecondOrder,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_derivative, outer_derivative_extras) = extras
    y = f(x)
    new_der, _ = value_and_derivative!(
        inner_derivative, der2, outer_derivative_extras, outer(backend), x, contexts...
    )
    return y, copyto!(der, new_der), der2
end
