"""
    value_and_gradient!(grad, backend, f, x) -> (y, grad)

Compute the primal value `y = f(x)` and the gradient `grad = ∇f(x)` of an array-to-scalar function, overwriting `grad` if possible.
"""
function value_and_gradient!(
    grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray
)
    return value_and_gradient!(Val{:fallback}(), grad, backend, f, x)
end

function value_and_gradient!(
    implem::Val{:fallback},
    grad::AbstractArray,
    backend::AbstractADType,
    f,
    x::AbstractArray,
)
    return value_and_gradient!(implem, autodiff_mode(backend), grad, backend, f, x)
end

function value_and_gradient!(
    ::Val{:fallback},
    ::Val{:forward},
    grad::AbstractArray,
    backend::AbstractADType,
    f,
    x::AbstractArray,
)
    y = f(x)
    for j in eachindex(IndexCartesian(), x)
        dx_j = basisarray(backend, x, j)
        grad[j] = pushforward!(grad[j], backend, f, x, dx_j)
    end
    return y, grad
end

function value_and_gradient!(
    ::Val{:fallback},
    ::Val{:reverse},
    grad::AbstractArray,
    backend::AbstractADType,
    f,
    x::AbstractArray,
)
    y = f(x)
    return y, pullback!(grad, backend, f, x, one(y))
end

"""
    value_and_gradient(backend, f, x) -> (y, grad)

Compute the primal value `y = f(x)` and the gradient `grad = ∇f(x)` of an array-to-scalar function.
"""
function value_and_gradient(backend::AbstractADType, f, x::AbstractArray)
    return value_and_gradient(Val{:fallback}(), backend, f, x)
end

function value_and_gradient(
    implem::Val{:fallback}, backend::AbstractADType, f, x::AbstractArray
)
    grad = similar(x)
    return value_and_gradient!(implem, grad, backend, f, x)
end

"""
    gradient!(grad, backend, f, x) -> grad

Compute the gradient `grad = ∇f(x)` of an array-to-scalar function, overwriting `grad` if possible.
"""
function gradient!(grad::AbstractArray, backend::AbstractADType, f, x::AbstractArray)
    return gradient!(Val{:fallback}(), grad, backend, f, x)
end

function gradient!(
    implem::Val{:fallback},
    grad::AbstractArray,
    backend::AbstractADType,
    f,
    x::AbstractArray,
)
    return last(value_and_gradient!(implem, grad, backend, f, x))
end

"""
    gradient(backend, f, x) -> grad

Compute the gradient `grad = ∇f(x)` of an array-to-scalar function.
"""
function gradient(backend::AbstractADType, f, x::AbstractArray)
    return gradient(Val{:fallback}(), backend, f, x)
end

function gradient(implem::Val{:fallback}, backend::AbstractADType, f, x::AbstractArray)
    return last(value_and_gradient(implem, backend, f, x))
end
