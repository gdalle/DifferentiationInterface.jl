## Preparation

abstract type SparseJacobianExtras <: JacobianExtras end

struct PushforwardSparseJacobianExtras{
    B,
    C<:AbstractColoringResult{:nonsymmetric,:column},
    M<:AbstractMatrix{<:Real},
    D,
    R,
    E<:PushforwardExtras,
} <: SparseJacobianExtras
    coloring_result::C
    compressed_matrix::M
    batched_seeds::Vector{Tangents{B,D}}
    batched_results::Vector{Tangents{B,R}}
    pushforward_extras::E
end

struct PullbackSparseJacobianExtras{
    B,
    C<:AbstractColoringResult{:nonsymmetric,:row},
    M<:AbstractMatrix{<:Real},
    D,
    R,
    E<:PullbackExtras,
} <: SparseJacobianExtras
    coloring_result::C
    compressed_matrix::M
    batched_seeds::Vector{Tangents{B,D}}
    batched_results::Vector{Tangents{B,R}}
    pullback_extras::E
end

function PushforwardSparseJacobianExtras{B}(;
    coloring_result::C,
    compressed_matrix::M,
    batched_seeds::Vector{Tangents{B,D}},
    batched_results::Vector{Tangents{B,R}},
    pushforward_extras::E,
) where {B,C,M,D,R,E}
    return PushforwardSparseJacobianExtras{B,C,M,D,R,E}(
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pushforward_extras,
    )
end

function PullbackSparseJacobianExtras{B}(;
    coloring_result::C,
    compressed_matrix::M,
    batched_seeds::Vector{Tangents{B,D}},
    batched_results::Vector{Tangents{B,R}},
    pullback_extras::E,
) where {B,C,M,D,R,E}
    return PullbackSparseJacobianExtras{B,C,M,D,R,E}(
        coloring_result, compressed_matrix, batched_seeds, batched_results, pullback_extras
    )
end

function DI.prepare_jacobian(f::F, backend::AutoSparse, x) where {F}
    y = f(x)
    return _prepare_sparse_jacobian_aux(
        (f,), backend, x, y, pushforward_performance(backend)
    )
end

function DI.prepare_jacobian(f!::F, y, backend::AutoSparse, x) where {F}
    return _prepare_sparse_jacobian_aux(
        (f!, y), backend, x, y, pushforward_performance(backend)
    )
end

function _prepare_sparse_jacobian_aux(
    f_or_f!y::FY, backend::AutoSparse, x, y, ::PushforwardFast
) where {FY}
    dense_backend = dense_ad(backend)
    sparsity = jacobian_sparsity(f_or_f!y..., x, sparsity_detector(backend))
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
    pushforward_extras = prepare_pushforward(
        f_or_f!y..., dense_backend, x, batched_seeds[1]
    )
    return PushforwardSparseJacobianExtras{B}(;
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pushforward_extras,
    )
end

function _prepare_sparse_jacobian_aux(
    f_or_f!y::FY, backend::AutoSparse, x, y, ::PushforwardSlow
) where {FY}
    dense_backend = dense_ad(backend)
    sparsity = jacobian_sparsity(f_or_f!y..., x, sparsity_detector(backend))
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
    pullback_extras = prepare_pullback(f_or_f!y..., dense_backend, x, batched_seeds[1])
    return PullbackSparseJacobianExtras{B}(;
        coloring_result, compressed_matrix, batched_seeds, batched_results, pullback_extras
    )
end

## One argument

function DI.jacobian(f::F, extras::SparseJacobianExtras, backend::AutoSparse, x) where {F}
    return _sparse_jacobian_aux((f,), extras, backend, x)
end

function DI.jacobian!(
    f::F, jac, extras::SparseJacobianExtras, backend::AutoSparse, x
) where {F}
    return _sparse_jacobian_aux!((f,), jac, extras, backend, x)
end

function DI.value_and_jacobian(
    f::F, extras::SparseJacobianExtras, backend::AutoSparse, x
) where {F}
    return f(x), jacobian(f, extras, backend, x)
end

function DI.value_and_jacobian!(
    f::F, jac, extras::SparseJacobianExtras, backend::AutoSparse, x
) where {F}
    return f(x), jacobian!(f, jac, extras, backend, x)
end

## Two arguments

function DI.jacobian(
    f!::F, y, extras::SparseJacobianExtras, backend::AutoSparse, x
) where {F}
    return _sparse_jacobian_aux((f!, y), extras, backend, x)
end

function DI.jacobian!(
    f!::F, y, jac, extras::SparseJacobianExtras, backend::AutoSparse, x
) where {F}
    return _sparse_jacobian_aux!((f!, y), jac, extras, backend, x)
end

function DI.value_and_jacobian(
    f!::F, y, extras::SparseJacobianExtras, backend::AutoSparse, x
) where {F}
    jac = jacobian(f!, y, extras, backend, x)
    f!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!::F, y, jac, extras::SparseJacobianExtras, backend::AutoSparse, x
) where {F}
    jacobian!(f!, y, jac, extras, backend, x)
    f!(y, x)
    return y, jac
end

## Common auxiliaries

function _sparse_jacobian_aux(
    f_or_f!y::FY, extras::PushforwardSparseJacobianExtras{B}, backend::AutoSparse, x
) where {FY,B}
    @compat (; coloring_result, batched_seeds, pushforward_extras) = extras
    dense_backend = dense_ad(backend)
    Ng = length(column_groups(coloring_result))

    pushforward_extras_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_extras, dense_backend, x, batched_seeds[1]
    )

    compressed_blocks = map(eachindex(batched_seeds)) do a
        dy_batch = pushforward(
            f_or_f!y..., pushforward_extras_same, dense_backend, x, batched_seeds[a]
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
    f_or_f!y::FY, extras::PullbackSparseJacobianExtras{B}, backend::AutoSparse, x
) where {FY,B}
    @compat (; coloring_result, batched_seeds, pullback_extras) = extras
    dense_backend = dense_ad(backend)
    Ng = length(row_groups(coloring_result))

    pullback_extras_same = prepare_pullback_same_point(
        f_or_f!y..., pullback_extras, dense_backend, x, batched_seeds[1]
    )

    compressed_blocks = map(eachindex(batched_seeds)) do a
        dx_batch = pullback(
            f_or_f!y..., pullback_extras_same, dense_backend, x, batched_seeds[a]
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
    f_or_f!y::FY, jac, extras::PushforwardSparseJacobianExtras{B}, backend::AutoSparse, x
) where {FY,B}
    @compat (;
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pushforward_extras,
    ) = extras
    dense_backend = dense_ad(backend)
    Ng = length(column_groups(coloring_result))

    pushforward_extras_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_extras, dense_backend, x, batched_seeds[1]
    )

    for a in eachindex(batched_seeds, batched_results)
        pushforward!(
            f_or_f!y...,
            batched_results[a],
            pushforward_extras_same,
            dense_backend,
            x,
            batched_seeds[a],
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
    f_or_f!y::FY, jac, extras::PullbackSparseJacobianExtras{B}, backend::AutoSparse, x
) where {FY,B}
    @compat (;
        coloring_result, compressed_matrix, batched_seeds, batched_results, pullback_extras
    ) = extras
    dense_backend = dense_ad(backend)
    Ng = length(row_groups(coloring_result))

    pullback_extras_same = prepare_pullback_same_point(
        f_or_f!y..., pullback_extras, dense_backend, x, batched_seeds[1]
    )

    for a in eachindex(batched_seeds, batched_results)
        pullback!(
            f_or_f!y...,
            batched_results[a],
            pullback_extras_same,
            dense_backend,
            x,
            batched_seeds[a],
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
