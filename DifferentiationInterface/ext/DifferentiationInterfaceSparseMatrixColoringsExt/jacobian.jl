## Preparation

struct PushforwardSparseJacobianPrep{
    BS<:BatchSizeSettings,
    C<:AbstractColoringResult{:nonsymmetric,:column},
    M<:AbstractMatrix{<:Real},
    S<:AbstractVector{<:NTuple},
    R<:AbstractVector{<:NTuple},
    E<:PushforwardPrep,
} <: SparseJacobianPrep
    batch_size_settings::BS
    coloring_result::C
    compressed_matrix::M
    batched_seeds::S
    batched_results::R
    pushforward_prep::E
end

struct PullbackSparseJacobianPrep{
    BS<:BatchSizeSettings,
    C<:AbstractColoringResult{:nonsymmetric,:row},
    M<:AbstractMatrix{<:Real},
    S<:AbstractVector{<:NTuple},
    R<:AbstractVector{<:NTuple},
    E<:PullbackPrep,
} <: SparseJacobianPrep
    batch_size_settings::BS
    coloring_result::C
    compressed_matrix::M
    batched_seeds::S
    batched_results::R
    pullback_prep::E
end

function DI.prepare_jacobian(
    f::F, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    dense_backend = dense_ad(backend)
    y = f(x, map(unwrap, contexts)...)
    perf = pushforward_performance(dense_backend)
    return _prepare_sparse_jacobian_aux(perf, y, (f,), backend, x, contexts...)
end

function DI.prepare_jacobian(
    f!::F, y, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    dense_backend = dense_ad(backend)
    perf = pushforward_performance(dense_backend)
    return _prepare_sparse_jacobian_aux(perf, y, (f!, y), backend, x, contexts...)
end

function _prepare_sparse_jacobian_aux(
    perf::PushforwardPerformance,
    y,
    f_or_f!y::FY,
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {FY,C}
    dense_backend = dense_ad(backend)
    sparsity = jacobian_sparsity(
        fy_with_contexts(f_or_f!y..., contexts...)..., x, sparsity_detector(backend)
    )
    if perf isa PushforwardFast
        problem = ColoringProblem{:nonsymmetric,:column}()
    else
        problem = ColoringProblem{:nonsymmetric,:row}()
    end
    coloring_result = coloring(
        sparsity,
        problem,
        coloring_algorithm(backend);
        decompression_eltype=promote_type(eltype(x), eltype(y)),
    )
    if perf isa PushforwardFast
        N = length(column_groups(coloring_result))
    else
        N = length(row_groups(coloring_result))
    end
    batch_size_settings = pick_batchsize(dense_backend, N)
    return _prepare_sparse_jacobian_aux_aux(
        batch_size_settings, coloring_result, y, f_or_f!y, backend, x, contexts...
    )
end

function _prepare_sparse_jacobian_aux_aux(
    batch_size_settings::BatchSizeSettings{B},
    coloring_result::AbstractColoringResult{:nonsymmetric,:column},
    y,
    f_or_f!y::FY,
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {B,FY,C}
    (; N, A) = batch_size_settings
    dense_backend = dense_ad(backend)
    groups = column_groups(coloring_result)
    seeds = [multibasis(backend, x, eachindex(x)[group]) for group in groups]
    compressed_matrix = stack(_ -> vec(similar(y)), groups; dims=2)
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for a in 1:A
    ]
    batched_results = [ntuple(b -> similar(y), Val(B)) for _ in batched_seeds]
    pushforward_prep = prepare_pushforward(
        f_or_f!y..., dense_backend, x, batched_seeds[1], contexts...
    )
    return PushforwardSparseJacobianPrep(
        batch_size_settings,
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pushforward_prep,
    )
end

function _prepare_sparse_jacobian_aux_aux(
    batch_size_settings::BatchSizeSettings{B},
    coloring_result::AbstractColoringResult{:nonsymmetric,:row},
    y,
    f_or_f!y::FY,
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {B,FY,C}
    (; N, A) = batch_size_settings
    dense_backend = dense_ad(backend)
    groups = row_groups(coloring_result)
    seeds = [multibasis(backend, y, eachindex(y)[group]) for group in groups]
    compressed_matrix = stack(_ -> vec(similar(x)), groups; dims=1)
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for a in 1:A
    ]
    batched_results = [ntuple(b -> similar(x), Val(B)) for _ in batched_seeds]
    pullback_prep = prepare_pullback(
        f_or_f!y..., dense_backend, x, batched_seeds[1], contexts...
    )
    return PullbackSparseJacobianPrep(
        batch_size_settings,
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pullback_prep,
    )
end

## One argument

function DI.jacobian!(
    f::F, jac, prep::SparseJacobianPrep, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    return _sparse_jacobian_aux!((f,), jac, prep, backend, x, contexts...)
end

function DI.jacobian(
    f::F, prep::SparseJacobianPrep, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    jac = similar(sparsity_pattern(prep), eltype(x))
    return DI.jacobian!(f, jac, prep, backend, x, contexts...)
end

function DI.value_and_jacobian(
    f::F, prep::SparseJacobianPrep, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    return f(x, map(unwrap, contexts)...), jacobian(f, prep, backend, x, contexts...)
end

function DI.value_and_jacobian!(
    f::F, jac, prep::SparseJacobianPrep, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    return f(x, map(unwrap, contexts)...), jacobian!(f, jac, prep, backend, x, contexts...)
end

## Two arguments

function DI.jacobian!(
    f!::F,
    y,
    jac,
    prep::SparseJacobianPrep,
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return _sparse_jacobian_aux!((f!, y), jac, prep, backend, x, contexts...)
end

function DI.jacobian(
    f!::F, y, prep::SparseJacobianPrep, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    jac = similar(sparsity_pattern(prep), promote_type(eltype(x), eltype(y)))
    return DI.jacobian!(f!, y, jac, prep, backend, x, contexts...)
end

function DI.value_and_jacobian(
    f!::F, y, prep::SparseJacobianPrep, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    jac = jacobian(f!, y, prep, backend, x, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    return y, jac
end

function DI.value_and_jacobian!(
    f!::F,
    y,
    jac,
    prep::SparseJacobianPrep,
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    jacobian!(f!, y, jac, prep, backend, x, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    return y, jac
end

## Common auxiliaries

function _sparse_jacobian_aux!(
    f_or_f!y::FY,
    jac,
    prep::PushforwardSparseJacobianPrep{<:BatchSizeSettings{B}},
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    (;
        batch_size_settings,
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pushforward_prep,
    ) = prep
    (; N) = batch_size_settings
    dense_backend = dense_ad(backend)

    pushforward_prep_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_prep, dense_backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        pushforward!(
            f_or_f!y...,
            batched_results[a],
            pushforward_prep_same,
            dense_backend,
            x,
            batched_seeds[a],
            contexts...,
        )

        for b in eachindex(batched_results[a])
            copyto!(
                view(compressed_matrix, :, 1 + ((a - 1) * B + (b - 1)) % N),
                vec(batched_results[a][b]),
            )
        end
    end

    decompress!(jac, compressed_matrix, coloring_result)
    return jac
end

function _sparse_jacobian_aux!(
    f_or_f!y::FY,
    jac,
    prep::PullbackSparseJacobianPrep{<:BatchSizeSettings{B}},
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    (;
        batch_size_settings,
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pullback_prep,
    ) = prep
    (; N) = batch_size_settings
    dense_backend = dense_ad(backend)

    pullback_prep_same = prepare_pullback_same_point(
        f_or_f!y..., pullback_prep, dense_backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        pullback!(
            f_or_f!y...,
            batched_results[a],
            pullback_prep_same,
            dense_backend,
            x,
            batched_seeds[a],
            contexts...,
        )

        for b in eachindex(batched_results[a])
            copyto!(
                view(compressed_matrix, 1 + ((a - 1) * B + (b - 1)) % N, :),
                vec(batched_results[a][b]),
            )
        end
    end

    decompress!(jac, compressed_matrix, coloring_result)
    return jac
end
