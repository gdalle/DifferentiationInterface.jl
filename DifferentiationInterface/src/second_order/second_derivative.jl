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
"""
function second_derivative end

"""
    second_derivative!(f, der2, backend, x, [extras]) -> der2

Compute the second derivative of the function `f` at point `x`, overwriting `der2`.
"""
function second_derivative! end

"""
    value_derivative_and_second_derivative(f, backend, x, [extras]) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`.
"""
function value_derivative_and_second_derivative end

"""
    value_derivative_and_second_derivative!(f, der, der2, backend, x, [extras]) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`, overwriting `der` and `der2`.
"""
function value_derivative_and_second_derivative! end

## Preparation

"""
    SecondDerivativeExtras

Abstract type for additional information needed by [`second_derivative`](@ref) and its variants.
"""
abstract type SecondDerivativeExtras <: Extras end

struct NoSecondDerivativeExtras <: SecondDerivativeExtras end

struct ClosureSecondDerivativeExtras{C,E} <: SecondDerivativeExtras
    inner_derivative_closure::C
    outer_derivative_extras::E
end

function prepare_second_derivative(f::F, backend::AbstractADType, x) where {F}
    return prepare_second_derivative(f, SecondOrder(backend, backend), x)
end

function prepare_second_derivative(f::F, backend::SecondOrder, x) where {F}
    inner_backend = nested(inner(backend))
    inner_derivative_closure(z) = derivative(f, inner_backend, z)
    outer_derivative_extras = prepare_derivative(
        inner_derivative_closure, outer(backend), x
    )
    return ClosureSecondDerivativeExtras(inner_derivative_closure, outer_derivative_extras)
end

## One argument

function second_derivative(
    f::F,
    backend::AbstractADType,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    return second_derivative(f, SecondOrder(backend, backend), x, extras)
end

function second_derivative(
    f::F,
    backend::SecondOrder,
    x,
    extras::ClosureSecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    @compat (; inner_derivative_closure, outer_derivative_extras) = extras
    return derivative(inner_derivative_closure, outer(backend), x, outer_derivative_extras)
end

function value_derivative_and_second_derivative(
    f::F,
    backend::AbstractADType,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    return value_derivative_and_second_derivative(
        f, SecondOrder(backend, backend), x, extras
    )
end

function value_derivative_and_second_derivative(
    f::F,
    backend::SecondOrder,
    x,
    extras::ClosureSecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    @compat (; inner_derivative_closure, outer_derivative_extras) = extras
    y = f(x)
    der, der2 = value_and_derivative(
        inner_derivative_closure, outer(backend), x, outer_derivative_extras
    )
    return y, der, der2
end

function second_derivative!(
    f::F,
    der2,
    backend::AbstractADType,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    return second_derivative!(f, der2, SecondOrder(backend, backend), x, extras)
end

function second_derivative!(
    f::F,
    der2,
    backend::SecondOrder,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    @compat (; inner_derivative_closure, outer_derivative_extras) = extras
    return derivative!(
        inner_derivative_closure, der2, outer(backend), x, outer_derivative_extras
    )
end

function value_derivative_and_second_derivative!(
    f::F,
    der,
    der2,
    backend::AbstractADType,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    return value_derivative_and_second_derivative!(
        f, der, der2, SecondOrder(backend, backend), x, extras
    )
end

function value_derivative_and_second_derivative!(
    f::F,
    der,
    der2,
    backend::SecondOrder,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    @compat (; inner_derivative_closure, outer_derivative_extras) = extras
    y = f(x)
    new_der, _ = value_and_derivative!(
        inner_derivative_closure, der2, outer(backend), x, outer_derivative_extras
    )
    return y, copyto!(der, new_der), der2
end
