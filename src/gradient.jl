"""
    value_and_gradient!(grad, backend, f, x, [extras]) -> (y, grad)

Compute the primal value `y = f(x)` and the gradient `grad = ∇f(x)` of an array-to-scalar function, overwriting `grad`.
"""
function value_and_gradient!(
    grad::AbstractArray,
    backend::AbstractADType,
    f::F,
    x::AbstractArray,
    extras=prepare_gradient(backend, f, x),
) where {F}
    return value_and_gradient_aux!(grad, backend, f, x, extras, mode(backend))
end

function value_and_gradient_aux!(
    grad, backend::AbstractADType, f::F, x, extras, ::ForwardMode
) where {F}
    y = f(x)
    for j in CartesianIndices(grad)
        dx_j = basisarray(backend, grad, j)
        grad[j] = pushforward(backend, f, x, dx_j, extras)
    end
    return y, grad
end

function value_and_gradient_aux!(grad, backend, f::F, x, extras, ::ReverseMode) where {F}
    return value_and_pullback!(grad, backend, f, x, one(eltype(x)), extras)
end

"""
    value_and_gradient(backend, f, x, [extras]) -> (y, grad)

Compute the primal value `y = f(x)` and the gradient `grad = ∇f(x)` of an array-to-scalar function.
"""
function value_and_gradient(
    backend::AbstractADType, f::F, x::AbstractArray, extras=prepare_gradient(backend, f, x)
) where {F}
    return value_and_gradient_aux(backend, f, x, extras, mode(backend))
end

function value_and_gradient_aux(backend, f::F, x, extras, ::AbstractMode) where {F}
    grad = similar(x)
    return value_and_gradient!(grad, backend, f, x, extras)
end

function value_and_gradient_aux(backend, f::F, x, extras, ::ReverseMode) where {F}
    return value_and_pullback(backend, f, x, one(eltype(x)), extras)
end

"""
    gradient!(grad, backend, f, x, [extras]) -> grad

Compute the gradient `grad = ∇f(x)` of an array-to-scalar function, overwriting `grad`.
"""
function gradient!(
    grad::AbstractArray,
    backend::AbstractADType,
    f::F,
    x::AbstractArray,
    extras=prepare_gradient(backend, f, x),
) where {F}
    return gradient_aux!(grad, backend, f, x, extras, mode(backend))
end

function gradient_aux!(grad, backend, f::F, x, extras, ::AbstractMode) where {F}
    return last(value_and_gradient!(grad, backend, f, x, extras))
end

function gradient_aux!(grad, backend, f::F, x, extras, ::ReverseMode) where {F}
    return pullback!(grad, backend, f, x, one(eltype(x)), extras)
end

"""
    gradient(backend, f, x, [extras]) -> grad

Compute the gradient `grad = ∇f(x)` of an array-to-scalar function.
"""
function gradient(
    backend::AbstractADType, f::F, x::AbstractArray, extras=prepare_gradient(backend, f, x)
) where {F}
    return gradient_aux(backend, f, x, extras, mode(backend))
end

function gradient_aux(backend, f::F, x, extras, ::AbstractMode) where {F}
    return last(value_and_gradient(backend, f, x, extras))
end

function gradient_aux(backend, f::F, x, extras, ::ReverseMode) where {F}
    return pullback(backend, f, x, one(eltype(x)), extras)
end
