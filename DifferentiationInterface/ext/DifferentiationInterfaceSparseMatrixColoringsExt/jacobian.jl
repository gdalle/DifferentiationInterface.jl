## Preparation

abstract type SparseJacobianPrep <: JacobianPrep end

struct PushforwardSparseJacobianPrep{
    B,
    C<:AbstractColoringResult{:nonsymmetric,:column},
    M<:AbstractMatrix{<:Real},
    D,
    R,
    E<:PushforwardPrep,
} <: SparseJacobianPrep
    coloring_result::C
    compressed_matrix::M
    batched_seeds::Vector{Tangents{B,D}}
    batched_results::Vector{Tangents{B,R}}
    pushforward_prep::E
end

struct PullbackSparseJacobianPrep{
    B,
    C<:AbstractColoringResult{:nonsymmetric,:row},
    M<:AbstractMatrix{<:Real},
    D,
    R,
    E<:PullbackPrep,
} <: SparseJacobianPrep
    coloring_result::C
    compressed_matrix::M
    batched_seeds::Vector{Tangents{B,D}}
    batched_results::Vector{Tangents{B,R}}
    pullback_prep::E
end

function PushforwardSparseJacobianPrep{B}(;
    coloring_result::C,
    compressed_matrix::M,
    batched_seeds::Vector{Tangents{B,D}},
    batched_results::Vector{Tangents{B,R}},
    pushforward_prep::E,
) where {B,C,M,D,R,E}
    return PushforwardSparseJacobianPrep{B,C,M,D,R,E}(
        coloring_result, compressed_matrix, batched_seeds, batched_results, pushforward_prep
    )
end

function PullbackSparseJacobianPrep{B}(;
    coloring_result::C,
    compressed_matrix::M,
    batched_seeds::Vector{Tangents{B,D}},
    batched_results::Vector{Tangents{B,R}},
    pullback_prep::E,
) where {B,C,M,D,R,E}
    return PullbackSparseJacobianPrep{B,C,M,D,R,E}(
        coloring_result, compressed_matrix, batched_seeds, batched_results, pullback_prep
    )
end

function DI.prepare_jacobian(
    f::F, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    y = f(x, map(unwrap, contexts)...)
    return _prepare_sparse_jacobian_aux(
        pushforward_performance(backend), y, (f,), backend, x, contexts...
    )
end

function DI.prepare_jacobian(
    f!::F, y, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_sparse_jacobian_aux(
        pushforward_performance(backend), y, (f!, y), backend, x, contexts...
    )
end

function _prepare_sparse_jacobian_aux(
    ::PushforwardFast, y, f_or_f!y::FY, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {FY,C}
    dense_backend = dense_ad(backend)

    sparsity = jacobian_sparsity(
        with_contexts(f_or_f!y..., contexts...)..., x, sparsity_detector(backend)
    )
    problem = ColoringProblem{:nonsymmetric,:column}()
    coloring_result = coloring(
        sparsity,
        problem,
        coloring_algorithm(backend);
        decompression_eltype=promote_type(eltype(x), eltype(y)),
    )
    groups = column_groups(coloring_result)
    Ng = length(groups)
    B = pick_batchsize(dense_backend, Ng)
    seeds = [multibasis(backend, x, eachindex(x)[group]) for group in groups]
    compressed_matrix = stack(_ -> vec(similar(y)), groups; dims=2)
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % Ng], Val(B))...) for
        a in 1:div(Ng, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(y), Val(B))...) for _ in batched_seeds]
    pushforward_prep = prepare_pushforward(
        f_or_f!y..., dense_backend, x, batched_seeds[1], contexts...
    )
    return PushforwardSparseJacobianPrep{B}(;
        coloring_result, compressed_matrix, batched_seeds, batched_results, pushforward_prep
    )
end

function _prepare_sparse_jacobian_aux(
    ::PushforwardSlow, y, f_or_f!y::FY, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {FY,C}
    dense_backend = dense_ad(backend)
    sparsity = jacobian_sparsity(
        with_contexts(f_or_f!y..., contexts...)..., x, sparsity_detector(backend)
    )
    problem = ColoringProblem{:nonsymmetric,:row}()
    coloring_result = coloring(
        sparsity,
        problem,
        coloring_algorithm(backend);
        decompression_eltype=promote_type(eltype(x), eltype(y)),
    )
    groups = row_groups(coloring_result)
    Ng = length(groups)
    B = pick_batchsize(dense_backend, Ng)
    seeds = [multibasis(backend, y, eachindex(y)[group]) for group in groups]
    compressed_matrix = stack(_ -> vec(similar(x)), groups; dims=1)
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % Ng], Val(B))...) for
        a in 1:div(Ng, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(x), Val(B))...) for _ in batched_seeds]
    pullback_prep = prepare_pullback(
        f_or_f!y..., dense_backend, x, batched_seeds[1], contexts...
    )
    return PullbackSparseJacobianPrep{B}(;
        coloring_result, compressed_matrix, batched_seeds, batched_results, pullback_prep
    )
end

## One argument

function DI.jacobian(
    f::F, prep::SparseJacobianPrep, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    return _sparse_jacobian_aux((f,), prep, backend, x, contexts...)
end

function DI.jacobian!(
    f::F, jac, prep::SparseJacobianPrep, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    return _sparse_jacobian_aux!((f,), jac, prep, backend, x, contexts...)
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

function DI.jacobian(
    f!::F, y, prep::SparseJacobianPrep, backend::AutoSparse, x, contexts::Vararg{Context,C}
) where {F,C}
    return _sparse_jacobian_aux((f!, y), prep, backend, x, contexts...)
end

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

function _sparse_jacobian_aux(
    f_or_f!y::FY,
    prep::PushforwardSparseJacobianPrep{B},
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    @compat (; coloring_result, batched_seeds, pushforward_prep) = prep
    dense_backend = dense_ad(backend)
    Ng = length(column_groups(coloring_result))

    pushforward_prep_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_prep, dense_backend, x, batched_seeds[1], contexts...
    )

    compressed_blocks = map(eachindex(batched_seeds)) do a
        dy_batch = pushforward(
            f_or_f!y...,
            pushforward_prep_same,
            dense_backend,
            x,
            batched_seeds[a],
            contexts...,
        )
        stack(vec, dy_batch.d; dims=2)
    end

    compressed_matrix = reduce(hcat, compressed_blocks)
    if Ng < size(compressed_matrix, 2)
        compressed_matrix = compressed_matrix[:, 1:Ng]
    end
    return decompress(compressed_matrix, coloring_result)
end

function _sparse_jacobian_aux(
    f_or_f!y::FY,
    prep::PullbackSparseJacobianPrep{B},
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    @compat (; coloring_result, batched_seeds, pullback_prep) = prep
    dense_backend = dense_ad(backend)
    Ng = length(row_groups(coloring_result))

    pullback_prep_same = prepare_pullback_same_point(
        f_or_f!y..., pullback_prep, dense_backend, x, batched_seeds[1], contexts...
    )

    compressed_blocks = map(eachindex(batched_seeds)) do a
        dx_batch = pullback(
            f_or_f!y...,
            pullback_prep_same,
            dense_backend,
            x,
            batched_seeds[a],
            contexts...,
        )
        stack(vec, dx_batch.d; dims=1)
    end

    compressed_matrix = reduce(vcat, compressed_blocks)
    if Ng < size(compressed_matrix, 1)
        compressed_matrix = compressed_matrix[1:Ng, :]
    end
    return decompress(compressed_matrix, coloring_result)
end

function _sparse_jacobian_aux!(
    f_or_f!y::FY,
    jac,
    prep::PushforwardSparseJacobianPrep{B},
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    @compat (;
        coloring_result, compressed_matrix, batched_seeds, batched_results, pushforward_prep
    ) = prep
    dense_backend = dense_ad(backend)
    Ng = length(column_groups(coloring_result))

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

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(compressed_matrix, :, 1 + ((a - 1) * B + (b - 1)) % Ng),
                vec(batched_results[a].d[b]),
            )
        end
    end

    decompress!(jac, compressed_matrix, coloring_result)
    return jac
end

function _sparse_jacobian_aux!(
    f_or_f!y::FY,
    jac,
    prep::PullbackSparseJacobianPrep{B},
    backend::AutoSparse,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    @compat (;
        coloring_result, compressed_matrix, batched_seeds, batched_results, pullback_prep
    ) = prep
    dense_backend = dense_ad(backend)
    Ng = length(row_groups(coloring_result))

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

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(compressed_matrix, 1 + ((a - 1) * B + (b - 1)) % Ng, :),
                vec(batched_results[a].d[b]),
            )
        end
    end

    decompress!(jac, compressed_matrix, coloring_result)
    return jac
end
