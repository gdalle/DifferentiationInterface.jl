## Allocating

"""
    value_and_gradient(f, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient(
    f, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
)
    return value_and_pullback(f, backend, x, true, extras)
end

"""
    value_and_gradient!!(f, grad, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient!!(
    f, grad, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
)
    return value_and_pullback!!(f, grad, backend, x, true, extras)
end

"""
    gradient(f, backend, x, [extras]) -> grad
"""
function gradient(f, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x))
    return value_and_gradient(f, backend, x, extras)[2]
end

"""
    gradient!!(f, grad, backend, x, [extras]) -> grad
"""
function gradient!!(
    f, grad, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
)
    return value_and_gradient!!(f, grad, backend, x, extras)[2]
end
