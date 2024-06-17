## Docstrings

"""
    prepare_hessian(f, backend, x) -> extras

Create an `extras` object that can be given to [`hessian`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hessian end

"""
    hessian(f, backend, x, [extras]) -> hess

Compute the Hessian matrix of the function `f` at point `x`.
"""
function hessian end

"""
    hessian!(f, hess, backend, x, [extras]) -> hess

Compute the Hessian matrix of the function `f` at point `x`, overwriting `hess`.
"""
function hessian! end

"""
    value_gradient_and_hessian(f, backend, x, [extras]) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`.
"""
function value_gradient_and_hessian end

"""
    value_gradient_and_hessian!(f, grad, hess, backend, x, [extras]) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`, overwriting `grad` and `hess`.
"""
function value_gradient_and_hessian! end

## Preparation

"""
    HessianExtras

Abstract type for additional information needed by [`hessian`](@ref) and its variants.
"""
abstract type HessianExtras <: Extras end

struct NoHessianExtras <: HessianExtras end

struct HVPGradientHessianExtras{N,E2<:HVPExtras,E1<:GradientExtras} <: HessianExtras
    hvp_batched_extras::E2
    gradient_extras::E1
end

function prepare_hessian(f::F, backend::AbstractADType, x) where {F}
    N = length(x)
    B = pick_batchsize(N)
    dx = basis(backend, x, first(CartesianIndices(x)))
    dx_batch = Batch(ntuple(Returns(dx), Val{B}()))
    hvp_batched_extras = prepare_hvp_batched(f, backend, x, dx_batch)
    gradient_extras = prepare_gradient(f, maybe_inner(backend), x)
    E2, E1 = typeof(hvp_batched_extras), typeof(gradient_extras)
    return HVPGradientHessianExtras{B,E2,E1}(hvp_batched_extras, gradient_extras)
end

## One argument

function value_gradient_and_hessian(f::F, backend::AbstractADType, x) where {F}
    return value_gradient_and_hessian(f, backend, x, prepare_hessian(f, backend, x))
end

function value_gradient_and_hessian!(f::F, grad, hess, backend::AbstractADType, x) where {F}
    return value_gradient_and_hessian!(
        f, grad, hess, backend, x, prepare_hessian(f, backend, x)
    )
end

function hessian(f::F, backend::AbstractADType, x) where {F}
    return hessian(f, backend, x, prepare_hessian(f, backend, x))
end

function hessian!(f::F, hess, backend::AbstractADType, x) where {F}
    return hessian!(f, hess, backend, x, prepare_hessian(f, backend, x))
end

function hessian(
    f::F, backend::AbstractADType, x, extras::HVPGradientHessianExtras{B}
) where {F,B}
    xinds = CartesianIndices(x)
    N = length(x)

    example_dx = basis(backend, x, first(xinds))
    example_dx_batch = Batch(ntuple(Returns(example_dx), Val{B}()))
    hvp_batched_extras_same = prepare_hvp_batched_same_point(
        f, backend, x, example_dx_batch, extras.hvp_batched_extras
    )

    hess = mapreduce(hcat, 1:div(N, B, RoundUp)) do k
        dx_batch_elements = ntuple(Val{B}()) do l
            basis(backend, x, xinds[1 + ((k - 1) * B + (l - 1)) % N])
        end
        dx_batch = Batch(dx_batch_elements)
        dg_batch = hvp_batched(f, backend, x, dx_batch, hvp_batched_extras_same)
        stack(vec, dg_batch.elements; dims=2)
    end

    return hess[:, 1:N]
end

function hessian!(
    f::F, hess, backend::AbstractADType, x, extras::HVPGradientHessianExtras{B}
) where {F,B}
    xinds = CartesianIndices(x)
    N = length(x)

    example_dx = basis(backend, x, first(xinds))
    example_dx_batch = Batch(ntuple(Returns(example_dx), Val{B}()))
    hvp_batched_extras_same = prepare_hvp_batched_same_point(
        f, backend, x, example_dx_batch, extras.hvp_batched_extras
    )

    for k in 1:div(N, B, RoundUp)
        dx_batch_elements = ntuple(Val{B}()) do l
            basis(backend, x, xinds[1 + ((k - 1) * B + (l - 1)) % N])
        end
        dx_batch = Batch(dx_batch_elements)
        dg_batch_elements = ntuple(Val{B}()) do l
            reshape(view(hess, :, 1 + ((k - 1) * B + (l - 1)) % N), size(x))
        end
        dg_batch = Batch(dg_batch_elements)
        hvp_batched!(f, dg_batch, backend, x, dx_batch, hvp_batched_extras_same)
    end

    return hess
end

function value_gradient_and_hessian(
    f::F, backend::AbstractADType, x, extras::HVPGradientHessianExtras
) where {F}
    y, grad = value_and_gradient(f, maybe_inner(backend), x, extras.gradient_extras)
    hess = hessian(f, backend, x, extras)
    return y, grad, hess
end

function value_gradient_and_hessian!(
    f::F, grad, hess, backend::AbstractADType, x, extras::HVPGradientHessianExtras
) where {F}
    y, _ = value_and_gradient!(f, grad, maybe_inner(backend), x, extras.gradient_extras)
    hessian!(f, hess, backend, x, extras)
    return y, grad, hess
end
