## Docstrings

"""
    prepare_hessian(f, backend, x, [contexts...]) -> prep

Create a `prep` object that can be given to [`hessian`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hessian end

"""
    prepare!_hessian(f, backend, x, [contexts...]) -> new_prep

Same behavior as [`prepare_hessian`](@ref) but can modify an existing `prep` object to avoid some allocations.

There is no guarantee that `prep` will be mutated, or that performance will be improved compared to preparation from scratch.

!!! danger
    For efficiency, this function needs to rely on backend package internals, therefore it not protected by semantic versioning.
"""
function prepare!_hessian end

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

struct HVPGradientHessianPrep{B,TD<:NTuple{B},TR<:NTuple{B},E2<:HVPPrep,E1<:GradientPrep} <:
       HessianPrep
    batched_seeds::Vector{TD}
    batched_results::Vector{TR}
    hvp_prep::E2
    gradient_prep::E1
    N::Int
end

function prepare_hessian(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    valB = pick_batchsize(backend, length(x))
    return _prepare_hessian_aux(valB, f, backend, x, contexts...)
end

function _prepare_hessian_aux(
    ::Val{B}, f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {B,F,C}
    N = length(x)
    seeds = [basis(backend, x, ind) for ind in eachindex(x)]
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for
        a in 1:div(N, B, RoundUp)
    ]
    batched_results = [ntuple(b -> similar(x), Val(B)) for _ in batched_seeds]
    hvp_prep = prepare_hvp(f, backend, x, batched_seeds[1], contexts...)
    gradient_prep = prepare_gradient(f, inner(backend), x, contexts...)
    TD = eltype(batched_seeds)
    TR = eltype(batched_results)
    E2, E1 = typeof(hvp_prep), typeof(gradient_prep)
    return HVPGradientHessianPrep{B,TD,TR,E2,E1}(
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
    (; batched_seeds, hvp_prep, N) = prep

    hvp_prep_same = prepare_hvp_same_point(
        f, hvp_prep, backend, x, batched_seeds[1], contexts...
    )

    hess_blocks = map(eachindex(batched_seeds)) do a
        dg_batch = hvp(f, hvp_prep_same, backend, x, batched_seeds[a], contexts...)
        block = stack_vec_col(dg_batch)
        if N % B != 0 && a == lastindex(batched_seeds)
            block = block[:, 1:(N - (a - 1) * B)]
        end
        block
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
    (; batched_seeds, batched_results, hvp_prep, N) = prep

    hvp_prep_same = prepare_hvp_same_point(
        f, hvp_prep, backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        hvp!(
            f, batched_results[a], hvp_prep_same, backend, x, batched_seeds[a], contexts...
        )

        for b in eachindex(batched_results[a])
            copyto!(
                view(hess, :, 1 + ((a - 1) * B + (b - 1)) % N), vec(batched_results[a][b])
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
