"""
    value_and_gradient!(grad, backend, f, x, [extras]) -> (y, grad)

Compute the primal value `y = f(x)` and the gradient `grad = ∇f(x)` of an array-to-scalar function, overwriting `grad`.
"""
function value_and_gradient!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray, extras=nothing
)
    return value_and_gradient_aux!(grad, backend, f, x, extras, mode(backend))
end

function value_and_gradient_aux!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray, extras, ::ForwardMode
)
    y = f(x)
    for j in eachindex(IndexCartesian(), grad)
        dx_j = basisarray(backend, grad, j)
        grad[j] = pushforward(backend, f, x, dx_j, extras)
    end
    return y, grad
end

function value_and_gradient_aux!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray, extras, ::ReverseMode
)
    return value_and_pullback!(grad, backend, f, x, one(eltype(x)), extras)
end

"""
    value_and_gradient(backend, f, x, [extras]) -> (y, grad)

Compute the primal value `y = f(x)` and the gradient `grad = ∇f(x)` of an array-to-scalar function.
"""
function value_and_gradient(backend::AbstractADType, f, x::AbstractArray, extras=nothing)
    return value_and_gradient_aux(backend, f, x, extras, mode(backend))
end

function value_and_gradient_aux(
    backend::AbstractADType, f, x::AbstractArray, extras, ::ForwardMode
)
    grad = similar(x)
    return value_and_gradient!(grad, backend, f, x, extras)
end

function value_and_gradient_aux(
    backend::AbstractADType, f, x::AbstractArray, extras, ::ReverseMode
)
    return value_and_pullback(backend, f, x, one(eltype(x)), extras)
end

"""
    gradient!(grad, backend, f, x, [extras]) -> grad

Compute the gradient `grad = ∇f(x)` of an array-to-scalar function, overwriting `grad`.
"""
function gradient!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray, extras=nothing
)
    return gradient_aux!(grad, backend, f, x, extras, mode(backend))
end

function gradient_aux!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray, extras, ::ForwardMode
)
    return last(value_and_gradient!(grad, backend, f, x, extras))
end

function gradient_aux!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray, extras, ::ReverseMode
)
    return pullback!(grad, backend, f, x, one(eltype(x)), extras)
end

"""
    gradient(backend, f, x, [extras]) -> grad

Compute the gradient `grad = ∇f(x)` of an array-to-scalar function.
"""
function gradient(backend::AbstractADType, f, x::AbstractArray, extras=nothing)
    return gradient_aux(backend, f, x, extras, mode(backend))
end

function gradient_aux(backend::AbstractADType, f, x::AbstractArray, extras, ::ForwardMode)
    return last(value_and_gradient(backend, f, x, extras))
end

function gradient_aux(backend::AbstractADType, f, x::AbstractArray, extras, ::ReverseMode)
    return pullback(backend, f, x, one(eltype(x)), extras)
end
