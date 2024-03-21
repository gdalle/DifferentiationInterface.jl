
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
    backend::AbstractADType,
    f::F,
    x::AbstractArray,
    extras=prepare_hessian(backend, f, x),
) where {F}
    return value_gradient_and_hessian!(
        grad, hess, SecondOrder(backend, backend), f, x, extras
    )
end

function value_gradient_and_hessian!(
    grad::AbstractArray,
    hess::AbstractMatrix,
    backend::SecondOrder,
    f::F,
    x::AbstractArray,
    extras=prepare_hessian(backend, f, x),
) where {F}
    return value_gradient_and_hessian_aux!(
        grad, hess, backend, f, x, extras, supports_mutation(outer(backend))
    )
end

function value_gradient_and_hessian_aux!(
    grad, hess, backend, f::F, x, extras, ::MutationSupported
) where {F}
    # TODO: suboptimal for reverse-over-forward (n^2 calls instead of n)
    y = f(x)
    function grad_closure!(storage, z)
        gradient!(storage, inner(backend), f, z, extras)
        return nothing
    end
    grad, hess = value_and_jacobian!(grad, hess, outer(backend), grad_closure!, x, extras)
    return y, grad, hess
end

function value_gradient_and_hessian_aux!(
    grad, hess, backend, f::F, x, extras, ::MutationNotSupported
) where {F}
    # TODO: suboptimal for reverse-over-forward (n^2 calls instead of n)
    y = f(x)
    grad_closure(z) = gradient(inner(backend), f, z, extras)
    new_grad, hess = value_and_jacobian!(hess, outer(backend), grad_closure, x, extras)
    grad .= new_grad
    return y, grad, hess
end

"""
    value_gradient_and_hessian(backend, f, x, [extras]) -> (y, grad, hess)

Compute the primal value `y = f(x)`, the gradient `grad = ∇f(x)` and the Hessian `hess = ∇²f(x)` of an array-to-scalar function, overwriting `grad` and `hess`.

$HESS_NOTES
"""
function value_gradient_and_hessian(
    backend::AbstractADType, f::F, x::AbstractArray, extras=prepare_hessian(backend, f, x)
) where {F}
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
    hess::AbstractMatrix,
    backend::AbstractADType,
    f::F,
    x::AbstractArray,
    extras=prepare_hessian(backend, f, x),
) where {F}
    grad = similar(x)
    return last(value_gradient_and_hessian!(grad, hess, backend, f, x, extras))
end

"""
    hessian(backend, f, x, [extras]) -> hess

Compute the Hessian `hess = ∇²f(x)` of an array-to-scalar function.

$HESS_NOTES
"""
function hessian(
    backend::AbstractADType, f::F, x::AbstractArray, extras=prepare_hessian(backend, f, x)
) where {F}
    return last(value_gradient_and_hessian(backend, f, x, extras))
end
