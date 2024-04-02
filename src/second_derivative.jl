## Preparation

"""
    SecondDerivativeExtras

Abstract type for additional information needed by second derivative operators.
"""
abstract type SecondDerivativeExtras <: Extras end

struct NoSecondDerivativeExtras <: SecondDerivativeExtras end

"""
    prepare_second_derivative([other_extras], f, backend, x) -> extras
    prepare_second_derivative([other_extras], f!, backend, y, x) -> extras

Create an `extras` object subtyping [`SecondDerivativeExtras`](@ref) that can be given to second derivative operators.
"""
function prepare_second_derivative(::Extras, f_or_f!, backend::AbstractADType, args...)
    return prepare_second_derivative(f_or_f!, backend, args...)
end

prepare_second_derivative(f, ::AbstractADType, x) = NoSecondDerivativeExtras()
prepare_second_derivative(f!, ::AbstractADType, y, x) = NoSecondDerivativeExtras()

## Allocating

"""
    second_derivative(f, backend, x, [extras]) -> der2
"""
function second_derivative(
    f,
    backend::AbstractADType,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
)
    new_backend = SecondOrder(backend, backend)
    new_extras = prepare_second_derivative(f, new_backend, x)
    return second_derivative(f, new_backend, x, new_extras)
end

function second_derivative(
    f,
    backend::SecondOrder,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
)
    function derivative_closure(z)
        inner_extras = prepare_derivative(extras, f, inner(backend), z)
        return derivative(f, inner(backend), z, inner_extras)
    end
    outer_extras = prepare_derivative(extras, derivative_closure, outer(backend), x)
    der2 = derivative(derivative_closure, outer(backend), x, outer_extras)
    return der2
end

"""
    second_derivative!!(f, der2, backend, x, [extras]) -> der2
"""
function second_derivative!!(
    f,
    der2,
    backend::AbstractADType,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
)
    new_backend = SecondOrder(backend, backend)
    new_extras = prepare_second_derivative(f, new_backend, x)
    return second_derivative!!(f, der2, new_backend, x, new_extras)
end

function second_derivative!!(
    f,
    der2,
    backend::SecondOrder,
    x,
    extras::SecondDerivativeExtras=prepare_second_derivative(f, backend, x),
)
    function derivative_closure(z)
        inner_extras = prepare_derivative(extras, f, inner(backend), z)
        return derivative(f, inner(backend), z, inner_extras)
    end
    outer_extras = prepare_derivative(extras, derivative_closure, outer(backend), x)
    der2 = derivative!!(derivative_closure, der2, outer(backend), x, outer_extras)
    return der2
end

## Mutating
