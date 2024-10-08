struct SparseHessianPrep{
    B,
    C<:AbstractColoringResult{:symmetric,:column},
    M<:AbstractMatrix{<:Real},
    TD<:NTuple{B},
    TR<:NTuple{B},
    E2<:HVPPrep,
    E1<:GradientPrep,
} <: HessianPrep
    coloring_result::C
    compressed_matrix::M
    batched_seeds::Vector{TD}
    batched_results::Vector{TR}
    hvp_prep::E2
    gradient_prep::E1
end

function SparseHessianPrep{B}(;
    coloring_result::C,
    compressed_matrix::M,
    batched_seeds::Vector{TD},
    batched_results::Vector{TR},
    hvp_prep::E2,
    gradient_prep::E1,
) where {B,C,M,TD,TR,E2,E1}
    return SparseHessianPrep{B,C,M,TD,TR,E2,E1}(
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        hvp_prep,
        gradient_prep,
    )
end

SMC.sparsity_pattern(prep::SparseHessianPrep) = sparsity_pattern(prep.coloring_result)
SMC.column_colors(prep::SparseHessianPrep) = column_colors(prep.coloring_result)
SMC.column_groups(prep::SparseHessianPrep) = column_groups(prep.coloring_result)

## Hessian, one argument

function DI.prepare_hessian(
    f::F, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    valB = pick_batchsize(dense_ad(backend), length(x))
    return _prepare_sparse_hessian_aux(valB, f, backend, x, contexts...)
end

function _prepare_sparse_hessian_aux(
    ::Val{B}, f::F, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {B,F,C}
    dense_backend = dense_ad(backend)
    sparsity = hessian_sparsity(
        with_contexts(f, contexts...), x, sparsity_detector(backend)
    )
    problem = ColoringProblem{:symmetric,:column}()
    coloring_result = coloring(
        sparsity, problem, coloring_algorithm(backend); decompression_eltype=eltype(x)
    )
    groups = column_groups(coloring_result)
    Ng = length(groups)
    seeds = [multibasis(backend, x, eachindex(x)[group]) for group in groups]
    compressed_matrix = stack(_ -> vec(similar(x)), groups; dims=2)
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % Ng], Val(B)) for
        a in 1:div(Ng, B, RoundUp)
    ]
    batched_results = [ntuple(b -> similar(x), Val(B)) for _ in batched_seeds]
    hvp_prep = prepare_hvp(f, dense_backend, x, batched_seeds[1], contexts...)
    gradient_prep = prepare_gradient(f, inner(dense_backend), x, contexts...)
    return SparseHessianPrep{B}(;
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        hvp_prep,
        gradient_prep,
    )
end

function DI.hessian!(
    f::F,
    hess,
    prep::SparseHessianPrep{B},
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {F,B,C}
    (; coloring_result, compressed_matrix, batched_seeds, batched_results, hvp_prep) = prep
    dense_backend = dense_ad(backend)
    Ng = length(column_groups(coloring_result))

    hvp_prep_same = prepare_hvp_same_point(
        f, hvp_prep, dense_backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        hvp!(
            f,
            batched_results[a],
            hvp_prep_same,
            dense_backend,
            x,
            batched_seeds[a],
            contexts...,
        )

        for b in eachindex(batched_results[a])
            copyto!(
                view(compressed_matrix, :, 1 + ((a - 1) * B + (b - 1)) % Ng),
                vec(batched_results[a][b]),
            )
        end
    end

    decompress!(hess, compressed_matrix, coloring_result)
    return hess
end

function DI.hessian(
    f::F, prep::SparseHessianPrep{B}, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,B,C}
    hess = similar(sparsity_pattern(prep), eltype(x))
    return DI.hessian!(f, hess, prep, backend, x, contexts...)
end

function DI.value_gradient_and_hessian!(
    f::F,
    grad,
    hess,
    prep::SparseHessianPrep,
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    y, _ = value_and_gradient!(
        f, grad, prep.gradient_prep, inner(dense_ad(backend)), x, contexts...
    )
    hessian!(f, hess, prep, backend, x, contexts...)
    return y, grad, hess
end

function DI.value_gradient_and_hessian(
    f::F, prep::SparseHessianPrep, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    y, grad = value_and_gradient(
        f, prep.gradient_prep, inner(dense_ad(backend)), x, contexts...
    )
    hess = hessian(f, prep, backend, x, contexts...)
    return y, grad, hess
end
