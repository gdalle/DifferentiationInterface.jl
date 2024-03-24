## Allocating

"""
    second_derivative(f, backend, x, [extras]) -> der2
"""
function second_derivative(
    f::F,
    backend::AbstractADType,
    x::Number,
    extras=prepare_second_derivative(f, backend, x),
) where {F}
    new_backend = SecondOrder(backend, backend)
    new_extras = prepare_second_derivative(f, new_backend, x)
    return second_derivative(f, new_backend, x, new_extras)
end

function second_derivative(
    f::F, backend::SecondOrder, x::Number, extras=prepare_second_derivative(f, backend, x)
) where {F}
    derivative_closure(z) = derivative(f, inner(backend), z, inner(extras))
    der2 = derivative(derivative_closure, outer(backend), x, outer(extras))
    return der2
end

## Mutating
