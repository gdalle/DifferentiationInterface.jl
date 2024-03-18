"""
    value_and_gradient!(grad, backend, f, x, [extras]) -> (y, grad)

Compute the primal value `y = f(x)` and the gradient `grad = ∇f(x)` of an array-to-scalar function, overwriting `grad`.
"""
function value_and_gradient!(
    grad::AbstractArray,
    backend::AbstractADType,
    f,
    x::AbstractArray,
    extras=prepare_gradient(backend, f, x),
)
    return value_and_pullback!(grad, backend, f, x, one(eltype(x)), extras)
end

"""
    value_and_gradient(backend, f, x, [extras]) -> (y, grad)

Compute the primal value `y = f(x)` and the gradient `grad = ∇f(x)` of an array-to-scalar function.
"""
function value_and_gradient(
    backend::AbstractADType, f, x::AbstractArray, extras=prepare_gradient(backend, f, x)
)
    return value_and_pullback(backend, f, x, one(eltype(x)), extras)
end

"""
    gradient!(grad, backend, f, x, [extras]) -> grad

Compute the gradient `grad = ∇f(x)` of an array-to-scalar function, overwriting `grad`.
"""
function gradient!(
    grad::AbstractArray,
    backend::AbstractADType,
    f,
    x::AbstractArray,
    extras=prepare_gradient(backend, f, x),
)
    return pullback!(grad, backend, f, x, one(eltype(x)), extras)
end

"""
    gradient(backend, f, x, [extras]) -> grad

Compute the gradient `grad = ∇f(x)` of an array-to-scalar function.
"""
function gradient(
    backend::AbstractADType, f, x::AbstractArray, extras=prepare_gradient(backend, f, x)
)
    return pullback(backend, f, x, one(eltype(x)), extras)
end
