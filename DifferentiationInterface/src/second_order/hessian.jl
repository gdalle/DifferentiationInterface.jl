## Docstrings

"""
    prepare_hessian(f, backend, x, [contexts...]) -> prep

Create a `prep` object that can be given to [`hessian`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hessian end

"""
    hessian(f, [prep,] backend, x, [contexts...]) -> hess

Compute the Hessian matrix of the function `f` at point `x`.

$(document_preparation("hessian"))
"""
function hessian end

"""
    hessian!(f, hess, [prep,] backend, x, [contexts...]) -> hess

Compute the Hessian matrix of the function `f` at point `x`, overwriting `hess`.

$(document_preparation("hessian"))
"""
function hessian! end

"""
    value_gradient_and_hessian(f, [prep,] backend, x, [contexts...]) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`.

$(document_preparation("hessian"))
"""
function value_gradient_and_hessian end

"""
    value_gradient_and_hessian!(f, grad, hess, [prep,] backend, x, [contexts...]) -> (y, grad, hess)

Compute the value, gradient vector and Hessian matrix of the function `f` at point `x`, overwriting `grad` and `hess`.

$(document_preparation("hessian"))
"""
function value_gradient_and_hessian! end

## Preparation

struct HVPGradientHessianPrep{B,D,R,E2<:HVPPrep,E1<:GradientPrep} <: HessianPrep
    batched_seeds::Vector{Tangents{B,D}}
    batched_results::Vector{Tangents{B,R}}
    hvp_prep::E2
    gradient_prep::E1
    N::Int
end

function prepare_hessian(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    N = length(x)
    B = pick_batchsize(outer(backend), N)
    seeds = [basis(backend, x, ind) for ind in eachindex(x)]
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B))...) for
        a in 1:div(N, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(x), Val(B))...) for _ in batched_seeds]
    hvp_prep = prepare_hvp(f, backend, x, batched_seeds[1], contexts...)
    gradient_prep = prepare_gradient(f, inner(backend), x, contexts...)
    D = eltype(batched_seeds[1])
    R = eltype(batched_results[1])
    E2, E1 = typeof(hvp_prep), typeof(gradient_prep)
    return HVPGradientHessianPrep{B,D,R,E2,E1}(
        batched_seeds, batched_results, hvp_prep, gradient_prep, N
    )
end

## One argument

function hessian(
    f::F,
    prep::HVPGradientHessianPrep{B},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,B,C}
    @compat (; batched_seeds, hvp_prep, N) = prep

    hvp_prep_same = prepare_hvp_same_point(
        f, hvp_prep, backend, x, batched_seeds[1], contexts...
    )

    hess_blocks = map(eachindex(batched_seeds)) do a
        dg_batch = hvp(f, hvp_prep_same, backend, x, batched_seeds[a], contexts...)
        stack(vec, dg_batch.d; dims=2)
    end

    hess = reduce(hcat, hess_blocks)
    if N < size(hess, 2)
        hess = hess[:, 1:N]
    end
    return hess
end

function hessian!(
    f::F,
    hess,
    prep::HVPGradientHessianPrep{B},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,B,C}
    @compat (; batched_seeds, batched_results, hvp_prep, N) = prep

    hvp_prep_same = prepare_hvp_same_point(
        f, hvp_prep, backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        hvp!(
            f, batched_results[a], hvp_prep_same, backend, x, batched_seeds[a], contexts...
        )

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(hess, :, 1 + ((a - 1) * B + (b - 1)) % N), vec(batched_results[a].d[b])
            )
        end
    end

    return hess
end

function value_gradient_and_hessian(
    f::F,
    prep::HVPGradientHessianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, grad = value_and_gradient(f, prep.gradient_prep, inner(backend), x, contexts...)
    hess = hessian(f, prep, backend, x, contexts...)
    return y, grad, hess
end

function value_gradient_and_hessian!(
    f::F,
    grad,
    hess,
    prep::HVPGradientHessianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, _ = value_and_gradient!(f, grad, prep.gradient_prep, inner(backend), x, contexts...)
    hessian!(f, hess, prep, backend, x, contexts...)
    return y, grad, hess
end
