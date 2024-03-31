"""
    prepare_gradient([other_extras], f, backend, x) -> extras

Create an `extras` object that can be given to gradient operators.
"""
function prepare_gradient(extras, f_or_f!, backend::AbstractADType, args...)
    return prepare_gradient(f_or_f!, backend, args...)
end

prepare_gradient(f, backend::AbstractADType, x) = prepare_pullback(f, backend, x)

## Allocating

"""
    value_and_gradient(f, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient(
    f, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
)
    new_extras = prepare_pullback(extras, f, backend, x)
    return value_and_pullback(f, backend, x, one(eltype(x)), new_extras)
end

"""
    value_and_gradient!!(f, grad, backend, x, [extras]) -> (y, grad)
"""
function value_and_gradient!!(
    f, grad, backend::AbstractADType, x, extras=prepare_gradient(f, backend, x)
)
    new_extras = prepare_pullback(extras, f, backend, x)
    return value_and_pullback!!(f, grad, backend, x, one(eltype(x)), new_extras)
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
