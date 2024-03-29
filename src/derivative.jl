## Allocating

"""
    value_and_derivative(f, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative(
    f, backend::AbstractADType, x, extras=prepare_derivative(f, backend, x)
)
    return value_and_pushforward(f, backend, x, one(x), extras)
end

"""
    value_and_derivative!!(f, der, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative!!(
    f, der, backend::AbstractADType, x, extras=prepare_derivative(f, backend, x)
)
    return value_and_pushforward!!(f, der, backend, x, one(x), extras)
end

"""
    derivative(f, backend, x, [extras]) -> der
"""
function derivative(f, backend::AbstractADType, x, extras=prepare_derivative(f, backend, x))
    return value_and_derivative(f, backend, x, extras)[2]
end

"""
    derivative!!(f, der, backend, x, [extras]) -> der
"""
function derivative!!(
    f, der, backend::AbstractADType, x, extras=prepare_derivative(f, backend, x)
)
    return value_and_derivative!!(f, der, backend, x, extras)[2]
end

## Mutating

"""
    value_and_derivative!!(f!, y, der, backend, x, [extras]) -> (y, der)
"""
function value_and_derivative!!(
    f!, y, der, backend::AbstractADType, x, extras=prepare_derivative(f!, backend, y, x)
)
    return value_and_pushforward!!(f!, y, der, backend, x, one(x), extras)
end
