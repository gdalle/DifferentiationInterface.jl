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

$(document_preparation("hessian"))
"""
function hessian end

"""
    hessian!(f, hess, backend, x, [extras]) -> hess

Compute the Hessian matrix of the function `f` at point `x`, overwriting `hess`.

$(document_preparation("hessian"))
"""
function hessian! end

"""
    value_gradient_and_hessian(f, backend, x, [extras]) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`.

$(document_preparation("hessian"))
"""
function value_gradient_and_hessian end

"""
    value_gradient_and_hessian!(f, grad, hess, backend, x, [extras]) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`, overwriting `grad` and `hess`.

$(document_preparation("hessian"))
"""
function value_gradient_and_hessian! end

## Preparation

"""
    HessianExtras

Abstract type for additional information needed by [`hessian`](@ref) and its variants.
"""
abstract type HessianExtras <: Extras end

struct NoHessianExtras <: HessianExtras end

struct HVPGradientHessianExtras{B,D,R,E2<:HVPExtras,E1<:GradientExtras} <: HessianExtras
    batched_seeds::Vector{Batch{B,D}}
    batched_results::Vector{Batch{B,R}}
    hvp_batched_extras::E2
    gradient_extras::E1
    N::Int
end

function prepare_hessian(f::F, backend::AbstractADType, x) where {F}
    N = length(x)
    B = pick_batchsize(maybe_outer(backend), N)
    seeds = [basis(backend, x, ind) for ind in CartesianIndices(x)]
    batched_seeds =
        Batch.([
            ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for
            a in 1:div(N, B, RoundUp)
        ])
    batched_results = Batch.([ntuple(b -> similar(x), Val(B)) for _ in batched_seeds])
    hvp_batched_extras = prepare_hvp_batched(f, backend, x, batched_seeds[1])
    gradient_extras = prepare_gradient(f, maybe_inner(backend), x)
    D = eltype(batched_seeds[1])
    R = eltype(batched_results[1])
    E2, E1 = typeof(hvp_batched_extras), typeof(gradient_extras)
    return HVPGradientHessianExtras{B,D,R,E2,E1}(
        batched_seeds, batched_results, hvp_batched_extras, gradient_extras, N
    )
end

## One argument

function hessian(
    f::F, backend::AbstractADType, x, extras::HVPGradientHessianExtras{B}
) where {F,B}
    @compat (; batched_seeds, hvp_batched_extras, N) = extras

    hvp_batched_extras_same = prepare_hvp_batched_same_point(
        f, backend, x, batched_seeds[1], hvp_batched_extras
    )

    hess_blocks = map(eachindex(batched_seeds)) do a
        dg_batch = hvp_batched(f, backend, x, batched_seeds[a], hvp_batched_extras_same)
        stack(vec, dg_batch.d; dims=2)
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
    @compat (; batched_seeds, batched_results, hvp_batched_extras, N) = extras

    hvp_batched_extras_same = prepare_hvp_batched_same_point(
        f, backend, x, batched_seeds[1], hvp_batched_extras
    )

    for a in eachindex(batched_seeds, batched_results)
        hvp_batched!(
            f, batched_results[a], backend, x, batched_seeds[a], hvp_batched_extras_same
        )

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(hess, :, 1 + ((a - 1) * B + (b - 1)) % N),
                vec(batched_results[a].d[b]),
            )
        end
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
