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

## Preparation

struct SelfPreparingDerivative{F,B}
    f::F
    backend::B
    extras_dict::Dict{Type,DerivativeExtras}

    function SelfPreparingDerivative(f::F, backend::B) where {F,B}
        return new{F,B}(f, backend, Dict{Type,DerivativeExtras}())
    end
end

function (self_prep_derivative::SelfPreparingDerivative)(x::X) where {X}
    @compat (; f, backend, extras_dict) = self_prep_derivative
    if !haskey(extras_dict, X)
        extras_dict[X] = prepare_derivative(f, backend, x)
    end
    return derivative(f, backend, x, extras_dict[X])
end

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
    inner_derivative_closure = SelfPreparingDerivative(f, inner_backend)
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
    @compat (; inner_derivative_closure, outer_derivative_extras) = extras
    return derivative!(
        inner_derivative_closure, der2, outer(backend), x, outer_derivative_extras
    )
end
