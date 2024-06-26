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

struct HVPGradientHessianExtras{B,D,E2<:HVPExtras,E1<:GradientExtras} <: HessianExtras
    seeds::D
    hvp_batched_extras::E2
    gradient_extras::E1
end

function prepare_hessian(f::F, backend::AbstractADType, x) where {F}
    N = length(x)
    B = pick_batchsize(maybe_outer(backend), N)
    seeds = [basis(backend, x, ind) for ind in CartesianIndices(x)]
    hvp_batched_extras = prepare_hvp_batched(
        f, backend, x, Batch(ntuple(Returns(seeds[1]), Val(B)))
    )
    gradient_extras = prepare_gradient(f, maybe_inner(backend), x)
    D = typeof(seeds)
    E2, E1 = typeof(hvp_batched_extras), typeof(gradient_extras)
    return HVPGradientHessianExtras{B,D,E2,E1}(seeds, hvp_batched_extras, gradient_extras)
end

## One argument

### Without extras

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

### With extras

function hessian(
    f::F, backend::AbstractADType, x, extras::HVPGradientHessianExtras{B}
) where {F,B}
    @compat (; seeds, hvp_batched_extras) = extras
    N = length(x)

    hvp_batched_extras_same = prepare_hvp_batched_same_point(
        f, backend, x, Batch(ntuple(Returns(seeds[1]), Val(B))), hvp_batched_extras
    )

    hess_blocks = map(1:div(N, B, RoundUp)) do a
        dx_batch_elements = ntuple(Val(B)) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % N]
        end
        dg_batch = hvp_batched(
            f, backend, x, Batch(dx_batch_elements), hvp_batched_extras_same
        )
        stack(vec, dg_batch.elements; dims=2)
    end

    hess = reduce(hcat, hess_blocks)
    if N < size(hess, 2)
        hess = hess[:, 1:N]
    end
    return hess
end

function hessian!(
    f::F, hess, backend::AbstractADType, x, extras::HVPGradientHessianExtras{B}
) where {F,B}
    xinds = CartesianIndices(x)
    N = length(x)

    dx_batch_elements = ntuple(Returns(basis(backend, x, xinds[1])), Val(B))
    hvp_batched_extras_same = prepare_hvp_batched_same_point(
        f, backend, x, Batch(dx_batch_elements), extras.hvp_batched_extras
    )

    for a in 1:div(N, B, RoundUp)
        dx_batch_elements = ntuple(Val(B)) do b
            basis(backend, x, xinds[1 + ((a - 1) * B + (b - 1)) % N])
        end
        dg_batch_elements = ntuple(Val(B)) do b
            reshape(view(hess, :, 1 + ((a - 1) * B + (b - 1)) % N), size(x))
        end
        hvp_batched!(
            f,
            Batch(dg_batch_elements),
            backend,
            x,
            Batch(dx_batch_elements),
            hvp_batched_extras_same,
        )
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
