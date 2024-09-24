## Docstrings

"""
    prepare_second_derivative(f, backend, x, [contexts...]) -> prep

Create an `prep` object that can be given to [`second_derivative`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_second_derivative end

"""
    second_derivative(f, [prep,] backend, x, [contexts...]) -> der2

Compute the second derivative of the function `f` at point `x`.

$(document_preparation("second_derivative"))
"""
function second_derivative end

"""
    second_derivative!(f, der2, [prep,] backend, x, [contexts...]) -> der2

Compute the second derivative of the function `f` at point `x`, overwriting `der2`.

$(document_preparation("second_derivative"))
"""
function second_derivative! end

"""
    value_derivative_and_second_derivative(f, [prep,] backend, x, [contexts...]) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`.

$(document_preparation("second_derivative"))
"""
function value_derivative_and_second_derivative end

"""
    value_derivative_and_second_derivative!(f, der, der2, [prep,] backend, x, [contexts...]) -> (y, der, der2)

Compute the value, first derivative and second derivative of the function `f` at point `x`, overwriting `der` and `der2`.

$(document_preparation("second_derivative"))
"""
function value_derivative_and_second_derivative! end

## Preparation

struct ClosureSecondDerivativePrep{ID,E<:DerivativePrep} <: SecondDerivativePrep
    inner_derivative::ID
    outer_derivative_prep::E
end

function prepare_second_derivative(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    rewrap = Rewrap(contexts...)
    function inner_derivative(_x, unannotated_contexts...)
        annotated_contexts = rewrap(unannotated_contexts...)
        return derivative(f, nested(inner(backend)), _x, annotated_contexts...)
    end
    outer_derivative_prep = prepare_derivative(
        inner_derivative, outer(backend), x, contexts...
    )
    return ClosureSecondDerivativePrep(inner_derivative, outer_derivative_prep)
end

## One argument

function second_derivative(
    f::F,
    prep::ClosureSecondDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_derivative, outer_derivative_prep) = prep
    return derivative(
        inner_derivative, outer_derivative_prep, outer(backend), x, contexts...
    )
end

function value_derivative_and_second_derivative(
    f::F,
    prep::ClosureSecondDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_derivative, outer_derivative_prep) = prep
    y = f(x, map(unwrap, contexts)...)
    der, der2 = value_and_derivative(
        inner_derivative, outer_derivative_prep, outer(backend), x, contexts...
    )
    return y, der, der2
end

function second_derivative!(
    f::F,
    der2,
    prep::SecondDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_derivative, outer_derivative_prep) = prep
    return derivative!(
        inner_derivative, der2, outer_derivative_prep, outer(backend), x, contexts...
    )
end

function value_derivative_and_second_derivative!(
    f::F,
    der,
    der2,
    prep::SecondDerivativePrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    @compat (; inner_derivative, outer_derivative_prep) = prep
    y = f(x, map(unwrap, contexts)...)
    new_der, _ = value_and_derivative!(
        inner_derivative, der2, outer_derivative_prep, outer(backend), x, contexts...
    )
    return y, copyto!(der, new_der), der2
end
