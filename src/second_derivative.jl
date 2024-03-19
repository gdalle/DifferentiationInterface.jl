"""
    value_derivative_and_second_derivative(backend, f, x, [extras]) -> (y, der, derder)

Compute the primal value `y = f(x)`, the derivative `der = f'(x)` and the second derivative `derder = f''(x)` of a scalar-to-scalar function.
"""
function value_derivative_and_second_derivative(
    backend::AbstractADType,
    f::F,
    x::Number,
    extras=prepare_second_derivative(backend, f, x),
) where {F}
    return value_derivative_and_second_derivative(
        SecondOrder(backend, backend), f, x, extras
    )
end

function value_derivative_and_second_derivative(
    backend::SecondOrder, f::F, x::Number, extras=prepare_second_derivative(backend, f, x)
) where {F}
    y = f(x)
    der_aux(x) = derivative(inner(backend), f, x, extras)
    der, derder = value_and_derivative(outer(backend), der_aux, x, extras)
    return y, der, derder
end

"""
    second_derivative(backend, f, x, [extras]) -> derder

Compute the second derivative `derder = f''(x)` of a scalar-to-scalar function.
"""
function second_derivative(
    backend::AbstractADType,
    f::F,
    x::Number,
    extras=prepare_second_derivative(backend, f, x),
) where {F}
    return second_derivative(SecondOrder(backend, backend), f, x, extras)
end

function second_derivative(
    backend::SecondOrder, f::F, x::Number, extras=prepare_second_derivative(backend, f, x)
) where {F}
    der_aux(x) = derivative(inner(backend), f, x, extras)
    derder = derivative(outer(backend), der_aux, x, extras)
    return derder
end
