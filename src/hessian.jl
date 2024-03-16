#=
Sources:
- https://d2jud02ci9yv69.cloudfront.net/2024-05-07-bench-hvp-81/blog/bench-hvp/
- https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
=#

"""
    SecondOrder

Combination of two backends for second-order differentiation of array-to-scalar functions.

# Fields

$(TYPEDFIELDS)
"""
struct SecondOrder{AD1<:AbstractADType,AD2<:AbstractADType} <: AbstractADType
    "backend for the inner differentiation, must be reverse mode"
    first::AD1
    "backend for the outer differentiation, must be forward mode"
    second::AD2

    function SecondOrder(first::AbstractADType, second::AbstractADType)
        if !(autodiff_mode(first) isa ReverseMode)
            throw(
                ArgumentError(
                    "Second order is only supported with forward-over-reverse, and $first is not reverse mode.",
                ),
            )
        elseif !(autodiff_mode(second) isa ForwardMode)
            throw(
                ArgumentError(
                    "Second order is only supported with forward-over-reverse, and $second is not forward mode.",
                ),
            )
        end
        return new{typeof(first),typeof(second)}(first, second)
    end
end

function Base.show(io::IO, backend::SecondOrder)
    return print(io, "SecondOrder($(backend.first), $(backend.second))")
end

function autodiff_mode(backend::SecondOrder)
    return (autodiff_mode(backend.first), autodiff_mode(backend.second))
end

"""
    gradient_and_hessian_vector_product!(grad, hvp, backend, f, x, v, [extras]) -> (grad, hvp)

Compute the gradient `grad = ∇f(x)` and the Hessian-vector product `hvp = ∇²f(x) * v` of an array-to-scalar function, overwriting `grad` and `hvp`.
"""
function gradient_and_hessian_vector_product! end

function gradient_and_hessian_vector_product!(
    grad::AbstractArray,
    hvp::AbstractArray,
    backend::SecondOrder,
    f,
    x::AbstractArray,
    v::AbstractArray,
    extras,
)
    function grad_aux!(grad, x)
        gradient!(grad, backend.first, f, x, extras)
        return nothing
    end
    return value_and_pushforward!(grad, hvp, backend.second, grad_aux!, x, v, extras)
end

"""
    gradient_and_hessian_vector_product(f, x, v, [extras]) -> (grad, hvp)

Compute the gradient `grad = ∇f(x)` and the Hessian-vector product `hvp = ∇²f(x) * v` of an array-to-scalar function.
"""
function gradient_and_hessian_vector_product end

function gradient_and_hessian_vector_product(
    backend::SecondOrder, f, x::AbstractArray, v::AbstractArray, extras
)
    grad_aux(x) = gradient(backend.first, f, x, extras)
    return value_and_pushforward(backend.second, grad_aux, x, v, extras)
end

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
    value_and_gradient_and_hessian!(grad, hess, backend, f, x, [extras]) -> (y, grad, hess)

Compute the primal value `y = f(x)`, the gradient `grad = ∇f(x)` and the Hessian `hess = ∇²f(x)` of an array-to-scalar function, overwriting `grad` and `hess`.

$HESS_NOTES
"""
function value_and_gradient_and_hessian! end

function value_and_gradient_and_hessian!(
    grad::AbstractArray,
    hess::AbstractMatrix,
    backend::AbstractADType,
    f,
    x::AbstractArray,
    extras,
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

"""
    value_and_gradient_and_hessian(backend, f, x, [extras]) -> (y, grad, hess)

Compute the primal value `y = f(x)`, the gradient `grad = ∇f(x)` and the Hessian `hess = ∇²f(x)` of an array-to-scalar function, overwriting `grad` and `hess`.

$HESS_NOTES
"""
function value_and_gradient_and_hessian end

function value_and_gradient_and_hessian(
    backend::AbstractADType, f, x::AbstractArray, extras
)
    grad = similar(x)
    hess = similar(x, length(x), length(x))
    return value_and_gradient_and_hessian!(grad, hess, backend, f, x, extras)
end

"""
    hessian!(hess, backend, f, x, [extras]) -> hess

Compute the Hessian `hess = ∇²f(x)` of an array-to-scalar function, overwriting `hess`.

$HESS_NOTES
"""
function hessian! end

function hessian!(
    hess::AbstractMatrix, backend::AbstractADType, f, x::AbstractArray, extras
)
    grad = similar(x)
    return last(value_and_gradient_and_hessian!(grad, hess, backend, f, x, extras))
end

"""
    hessian(backend, f, x, [extras]) -> hess

Compute the Hessian `hess = ∇²f(x)` of an array-to-scalar function.

$HESS_NOTES
"""
function hessian end

function hessian(backend::AbstractADType, f, x::AbstractArray, extras)
    return last(value_and_gradient_and_hessian(backend, f, x, extras))
end
