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
    value_gradient_and_hessian(f, [extras,] backend, x) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`.

$(document_preparation("hessian"))
"""
function value_gradient_and_hessian end

"""
    value_gradient_and_hessian!(f, grad, hess, [extras,] backend, x) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`, overwriting `grad` and `hess`.

$(document_preparation("hessian"))
"""
function value_gradient_and_hessian! end

## Preparation

struct HVPGradientHessianExtras{B,D,R,E2<:HVPExtras,E1<:GradientExtras} <: HessianExtras
    batched_seeds::Vector{Tangents{B,D}}
    batched_results::Vector{Tangents{B,R}}
    hvp_extras::E2
    gradient_extras::E1
    N::Int
end

function prepare_hessian(f::F, backend::AbstractADType, x) where {F}
    N = length(x)
    B = pick_batchsize(maybe_outer(backend), N)
    seeds = [basis(backend, x, ind) for ind in eachindex(x)]
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B))...) for
        a in 1:div(N, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(x), Val(B))...) for _ in batched_seeds]
    hvp_extras = prepare_hvp(f, backend, x, batched_seeds[1])
    gradient_extras = prepare_gradient(f, maybe_inner(backend), x)
    D = eltype(batched_seeds[1])
    R = eltype(batched_results[1])
    E2, E1 = typeof(hvp_extras), typeof(gradient_extras)
    return HVPGradientHessianExtras{B,D,R,E2,E1}(
        batched_seeds, batched_results, hvp_extras, gradient_extras, N
    )
end

## One argument

function hessian(
    f::F, extras::HVPGradientHessianExtras{B}, backend::AbstractADType, x
) where {F,B}
    @compat (; batched_seeds, hvp_extras, N) = extras

    hvp_extras_same = prepare_hvp_same_point(f, hvp_extras, backend, x, batched_seeds[1])

    hess_blocks = map(eachindex(batched_seeds)) do a
        dg_batch = hvp(f, hvp_extras_same, backend, x, batched_seeds[a])
        stack(vec, dg_batch.d; dims=2)
    end

    hess = reduce(hcat, hess_blocks)
    if N < size(hess, 2)
        hess = hess[:, 1:N]
    end
    return hess
end

function hessian!(
    f::F, hess, extras::HVPGradientHessianExtras{B}, backend::AbstractADType, x
) where {F,B}
    @compat (; batched_seeds, batched_results, hvp_extras, N) = extras

    hvp_extras_same = prepare_hvp_same_point(f, hvp_extras, backend, x, batched_seeds[1])

    for a in eachindex(batched_seeds, batched_results)
        hvp!(f, batched_results[a], hvp_extras_same, backend, x, batched_seeds[a])

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(hess, :, 1 + ((a - 1) * B + (b - 1)) % N), vec(batched_results[a].d[b])
            )
        end
    end

    return hess
end

function value_gradient_and_hessian(
    f::F, extras::HVPGradientHessianExtras, backend::AbstractADType, x
) where {F}
    y, grad = value_and_gradient(f, extras.gradient_extras, maybe_inner(backend), x)
    hess = hessian(f, extras, backend, x)
    return y, grad, hess
end

function value_gradient_and_hessian!(
    f::F, grad, hess, extras::HVPGradientHessianExtras, backend::AbstractADType, x
) where {F}
    y, _ = value_and_gradient!(f, grad, extras.gradient_extras, maybe_inner(backend), x)
    hessian!(f, hess, extras, backend, x)
    return y, grad, hess
end
