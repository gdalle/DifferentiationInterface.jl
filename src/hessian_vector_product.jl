#=
Sources:
- https://d2jud02ci9yv69.cloudfront.net/2024-05-07-bench-hvp-81/blog/bench-hvp/
- https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

Start by reading the allocating versions
=#

## Forward-over-something backends give gradient in addition to HVP

"""
    gradient_and_hessian_vector_product(backend, f, x, v, [extras]) -> (grad, hvp)

Compute the gradient `grad = ∇f(x)` and the Hessian-vector product `hvp = ∇²f(x) * v` of an array-to-scalar function.

!!! warning
    Only works with a forward outer mode.
"""
function gradient_and_hessian_vector_product(
    backend::AbstractADType,
    f,
    x::AbstractArray,
    v::AbstractArray,
    extras=prepare_hessian_vector_product(backend, f, x),
)
    return gradient_and_hessian_vector_product(
        SecondOrder(backend, backend), f, x, v, extras
    )
end

function gradient_and_hessian_vector_product(
    backend::SecondOrder,
    f,
    x::AbstractArray,
    v::AbstractArray,
    extras=prepare_hessian_vector_product(backend, f, x),
)
    return gradient_and_hessian_vector_product_aux(
        backend, f, x, v, extras, mode(inner(backend)), mode(outer(backend))
    )
end

function gradient_and_hessian_vector_product_aux(
    backend, f, x, v, extras, ::AbstractMode, ::ForwardMode
)
    grad_aux(z) = gradient(inner(backend), f, z, extras)
    return value_and_pushforward(outer(backend), grad_aux, x, v, extras)
end

function gradient_and_hessian_vector_product_aux(
    backend, f, x, v, extras, ::AbstractMode, ::ReverseMode
)
    throw(ArgumentError("HVP must be computed without gradient for reverse-over-something"))
end

"""
    gradient_and_hessian_vector_product!(grad, backend, hvp, backend, f, x, v, [extras]) -> (grad, hvp)

Compute the gradient `grad = ∇f(x)` and the Hessian-vector product `hvp = ∇²f(x) * v` of an array-to-scalar function, overwriting `grad` and `hvp`.

!!! warning
    Only works with a forward outer mode.
"""
function gradient_and_hessian_vector_product!(
    grad::AbstractArray,
    hvp::AbstractArray,
    backend::AbstractADType,
    f,
    x::AbstractArray,
    v::AbstractArray,
    extras=prepare_hessian_vector_product(backend, f, x),
)
    return gradient_and_hessian_vector_product!(
        grad, hvp, SecondOrder(backend, backend), f, x, v, extras
    )
end

function gradient_and_hessian_vector_product!(
    grad::AbstractArray,
    hvp::AbstractArray,
    backend::SecondOrder,
    f,
    x::AbstractArray,
    v::AbstractArray,
    extras=prepare_hessian_vector_product(backend, f, x),
)
    return gradient_and_hessian_vector_product_aux!(
        grad, hvp, backend, f, x, v, extras, mode(inner(backend)), mode(outer(backend))
    )
end

function gradient_and_hessian_vector_product_aux!(
    grad, hvp, backend, f, x, v, extras, ::AbstractMode, ::ForwardMode
)
    function grad_aux!(storage, z)
        gradient!(storage, inner(backend), f, z, extras)
        return nothing
    end
    return value_and_pushforward!(grad, hvp, outer(backend), grad_aux!, x, v, extras)
end

function gradient_and_hessian_vector_product_aux!(
    grad, hvp, backend, f, x, v, extras, ::AbstractMode, ::ReverseMode
)
    throw(ArgumentError("HVP must be computed without gradient for reverse-over-something"))
end

## All backends can give the HVP

"""
    hessian_vector_product(backend, f, x, v, [extras]) -> hvp

Compute the Hessian-vector product `hvp = ∇²f(x) * v` of an array-to-scalar function.
"""
function hessian_vector_product(
    backend::AbstractADType,
    f,
    x::AbstractArray,
    v::AbstractArray,
    extras=prepare_hessian_vector_product(backend, f, x),
)
    return hessian_vector_product(SecondOrder(backend, backend), f, x, v, extras)
end

function hessian_vector_product(
    backend::SecondOrder,
    f,
    x::AbstractArray,
    v::AbstractArray,
    extras=prepare_hessian_vector_product(backend, f, x),
)
    return hessian_vector_product_aux(
        backend, f, x, v, extras, mode(inner(backend)), mode(outer(backend))
    )
end

function hessian_vector_product_aux(backend, f, x, v, extras, ::ReverseMode, ::ReverseMode)
    dotgrad_aux(z) = dot(gradient(inner(backend), f, z, extras), v)
    return gradient(outer(backend), dotgrad_aux, x, extras)
end

function hessian_vector_product_aux(backend, f, x, v, extras, ::ForwardMode, ::ReverseMode)
    jvp_aux(z) = pushforward(inner(backend), f, z, v, extras)
    return gradient(outer(backend), jvp_aux, x, extras)
end

function hessian_vector_product_aux(backend, f, x, v, extras, ::AbstractMode, ::ForwardMode)
    _, hvp = gradient_and_hessian_vector_product(backend, f, x, v, extras)
    return hvp
end

"""
    hessian_vector_product!(hvp, backend, f, x, v, [extras]) -> hvp

Compute the Hessian-vector product `hvp = ∇²f(x) * v` of an array-to-scalar function, overwriting `hvp`.
"""
function hessian_vector_product!(
    hvp::AbstractArray,
    backend::AbstractADType,
    f,
    x::AbstractArray,
    v::AbstractArray,
    extras=prepare_hessian_vector_product(backend, f, x),
)
    return hessian_vector_product!(hvp, SecondOrder(backend, backend), f, x, v, extras)
end

function hessian_vector_product!(
    hvp::AbstractArray,
    backend::SecondOrder,
    f,
    x::AbstractArray,
    v::AbstractArray,
    extras=prepare_hessian_vector_product(backend, f, x),
)
    return hessian_vector_product_aux!(
        hvp, backend, f, x, v, extras, mode(inner(backend)), mode(outer(backend))
    )
end

function hessian_vector_product_aux!(
    hvp, backend, f, x, v, extras, ::ReverseMode, ::ReverseMode
)
    dotgrad_aux(z) = dot(gradient(inner(backend), f, z, extras), v)  # allocates
    return gradient!(hvp, outer(backend), dotgrad_aux, x, extras)
end

function hessian_vector_product_aux!(
    hvp, backend, f, x, v, extras, ::ForwardMode, ::ReverseMode
)
    jvp_aux(z) = pushforward(inner(backend), f, z, v, extras)
    return gradient!(hvp, outer(backend), jvp_aux, x, extras)
end

function hessian_vector_product_aux!(
    hvp, backend, f, x, v, extras, ::AbstractMode, ::ForwardMode
)
    grad = similar(x)  # allocates
    _, hvp = gradient_and_hessian_vector_product!(grad, hvp, backend, f, x, v, extras)
    return hvp
end
