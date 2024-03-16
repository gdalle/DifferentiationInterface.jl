
const HESS_NOTES = """
## Notes

Regardless of the shape of `x`, if `x` has length `n`, then `hess` is expected to be a `n × n` matrix.
This function acts as if the input had been flattened with `vec`.
"""

function check_hess(hess::AbstractMatrix, x::AbstractArray)
    n = length(x)
    size(hess) != (n, n) && throw(
        DimensionMismatch("Size of Hessian buffer doesn't match expected size ($n, $n)")
    )
    return nothing
end

"""
    value_gradient_and_hessian!(grad, hess, backend, f, x, [extras]) -> (y, grad, hess)

Compute the primal value `y = f(x)`, the gradient `grad = ∇f(x)` and the Hessian `hess = ∇²f(x)` of an array-to-scalar function, overwriting `grad` and `hess`.

$HESS_NOTES
"""
function value_gradient_and_hessian!(
    grad::AbstractArray,
    hess::AbstractMatrix,
    backend::SecondOrder,
    f,
    x::AbstractArray,
    extras=nothing,
)
    return value_gradient_and_hessian_aux!(
        grad, hess, backend, f, x, extras, mode(inner(backend)), mode(outer(backend))
    )
end

function value_gradient_and_hessian_aux!(
    grad::AbstractArray,
    hess::AbstractMatrix,
    backend::SecondOrder,
    f,
    x::AbstractArray,
    extras,
    ::AbstractMode,
    ::ForwardMode,
)
    y = f(x)
    check_hess(hess, x)
    for (k, j) in enumerate(eachindex(IndexCartesian(), x))
        dx_j = basisarray(backend, x, j)
        hess_col_j = reshape(view(hess, :, k), size(x))
        gradient_and_hessian_vector_product!(grad, hess_col_j, backend, f, x, dx_j, extras)
    end
    return y, grad, hess
end

function value_gradient_and_hessian_aux!(
    grad::AbstractArray,
    hess::AbstractMatrix,
    backend::SecondOrder,
    f,
    x::AbstractArray,
    extras,
    ::AbstractMode,
    ::ReverseMode,
)
    y, _ = value_and_gradient!(grad, inner(backend), f, x, extras)
    check_hess(hess, x)
    for (k, j) in enumerate(eachindex(IndexCartesian(), x))
        dx_j = basisarray(backend, x, j)
        hess_col_j = reshape(view(hess, :, k), size(x))
        hessian_vector_product!(hess_col_j, backend, f, x, dx_j, extras)
    end
    return y, grad, hess
end

"""
    value_gradient_and_hessian(backend, f, x, [extras]) -> (y, grad, hess)

Compute the primal value `y = f(x)`, the gradient `grad = ∇f(x)` and the Hessian `hess = ∇²f(x)` of an array-to-scalar function, overwriting `grad` and `hess`.

$HESS_NOTES
"""
function value_gradient_and_hessian(
    backend::AbstractADType, f, x::AbstractArray, extras=nothing
)
    grad = similar(x)
    hess = similar(x, length(x), length(x))
    return value_gradient_and_hessian!(grad, hess, backend, f, x, extras)
end

"""
    hessian!(hess, backend, f, x, [extras]) -> hess

Compute the Hessian `hess = ∇²f(x)` of an array-to-scalar function, overwriting `hess`.

$HESS_NOTES
"""
function hessian!(
    hess::AbstractMatrix, backend::AbstractADType, f, x::AbstractArray, extras=nothing
)
    grad = similar(x)
    return last(value_gradient_and_hessian!(grad, hess, backend, f, x, extras))
end

"""
    hessian(backend, f, x, [extras]) -> hess

Compute the Hessian `hess = ∇²f(x)` of an array-to-scalar function.

$HESS_NOTES
"""
function hessian(backend::AbstractADType, f, x::AbstractArray, extras=nothing)
    return last(value_gradient_and_hessian(backend, f, x, extras))
end
