"""
    value_and_gradient!(grad, backend, f, x, [extras]) -> (y, grad)

Compute the primal value `y = f(x)` and the gradient `grad = ∇f(x)` of an array-to-scalar function, overwriting `grad` if possible.
"""
function value_and_gradient!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray, extras=nothing
)
    return value_and_gradient!(grad, backend, f, x, extras, autodiff_mode(backend))
end

function value_and_gradient!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray, extras, ::ForwardMode
)
    y = f(x)
    for j in eachindex(IndexCartesian(), x)
        dx_j = basisarray(backend, x, j)
        grad[j] = pushforward!(grad[j], backend, f, x, dx_j, extras)
    end
    return y, grad
end

function value_and_gradient!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray, extras, ::ReverseMode
)
    y = f(x)
    return y, pullback!(grad, backend, f, x, one(y), extras)
end

"""
    value_and_gradient(backend, f, x, [extras]) -> (y, grad)

Compute the primal value `y = f(x)` and the gradient `grad = ∇f(x)` of an array-to-scalar function.
"""
function value_and_gradient(backend::AbstractADType, f, x::AbstractArray, args...)
    grad = similar(x)
    return value_and_gradient!(grad, backend, f, x, args...)
end

"""
    gradient!(grad, backend, f, x, [extras]) -> grad

Compute the gradient `grad = ∇f(x)` of an array-to-scalar function, overwriting `grad` if possible.
"""
function gradient!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray, args...
)
    return last(value_and_gradient!(grad, backend, f, x, args...))
end

"""
    gradient(backend, f, x, [extras]) -> grad

Compute the gradient `grad = ∇f(x)` of an array-to-scalar function.
"""
function gradient(backend::AbstractADType, f, x::AbstractArray, args...)
    return last(value_and_gradient(backend, f, x, args...))
end
