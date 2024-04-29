## Docstrings

"""
    prepare_second_derivative(f, backend, x) -> extras

Create an `extras` object subtyping [`SecondDerivativeExtras`](@ref) that can be given to second derivative operators.
"""
function prepare_second_derivative end

"""
    second_derivative(f, backend, x, [extras]) -> der2
"""
function second_derivative end

"""
    second_derivative!(f, der2, backend, x, [extras]) -> der2
"""
function second_derivative! end

## Preparation

"""
    SecondDerivativeExtras

Abstract type for additional information needed by second derivative operators.
"""
abstract type SecondDerivativeExtras <: Extras end

struct NoSecondDerivativeExtras <: SecondDerivativeExtras end

struct ClosureSecondDerivativeExtras{C,E} <: SecondDerivativeExtras
    inner_derivative_closure::C
    outer_derivative_extras::E
end

prepare_second_derivative(f::F, ::AbstractADType, x) where {F} = NoSecondDerivativeExtras()

function prepare_second_derivative(f::F, backend::SecondOrder, x) where {F}
    inner_derivative_closure(z) = derivative(f, inner(backend), z)
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
    new_backend = SecondOrder(backend, backend)
    new_extras = prepare_second_derivative(f, new_backend, x)
    return second_derivative(f, new_backend, x, new_extras)
end

function second_derivative(
    f::F,
    backend::SecondOrder,
    x,
    extras::ClosureSecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    (; inner_derivative_closure, outer_derivative_extras) = extras
    return derivative(inner_derivative_closure, outer(backend), x, outer_derivative_extras)
end

function second_derivative!(
    f::F,
    der2,
    backend::AbstractADType,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    new_backend = SecondOrder(backend, backend)
    new_extras = prepare_second_derivative(f, new_backend, x)
    return second_derivative!(f, der2, new_backend, x, new_extras)
end

function second_derivative!(
    f::F,
    der2,
    backend::SecondOrder,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
) where {F}
    (; inner_derivative_closure, outer_derivative_extras) = extras
    return derivative!(
        inner_derivative_closure, der2, outer(backend), x, outer_derivative_extras
    )
end
