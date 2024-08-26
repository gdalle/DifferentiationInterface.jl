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
    batched_seeds::Vector{Batch{B,D}}
    batched_results::Vector{Batch{B,R}}
    pushforward_batched_extras::E
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
    batched_seeds::Vector{Batch{B,D}}
    batched_results::Vector{Batch{B,R}}
    pullback_batched_extras::E
end

function PushforwardSparseJacobianExtras{B}(;
    coloring_result::C,
    compressed_matrix::M,
    batched_seeds::Vector{Batch{B,D}},
    batched_results::Vector{Batch{B,R}},
    pushforward_batched_extras::E,
) where {B,C,M,D,R,E}
    return PushforwardSparseJacobianExtras{B,C,M,D,R,E}(
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pushforward_batched_extras,
    )
end

function PullbackSparseJacobianExtras{B}(;
    coloring_result::C,
    compressed_matrix::M,
    batched_seeds::Vector{Batch{B,D}},
    batched_results::Vector{Batch{B,R}},
    pullback_batched_extras::E,
) where {B,C,M,D,R,E}
    return PullbackSparseJacobianExtras{B,C,M,D,R,E}(
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pullback_batched_extras,
    )
end

function prepare_jacobian(f::F, backend::AutoSparse, x) where {F}
    y = f(x)
    return prepare_sparse_jacobian_aux(
        (f,), backend, x, y, pushforward_performance(backend)
    )
end

function prepare_jacobian(f!::F, y, backend::AutoSparse, x) where {F}
    return prepare_sparse_jacobian_aux(
        (f!, y), backend, x, y, pushforward_performance(backend)
    )
end

function prepare_sparse_jacobian_aux(
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
    seeds = [multibasis(backend, x, CartesianIndices(x)[group]) for group in groups]
    compressed_matrix = stack(_ -> vec(similar(y)), groups; dims=2)
    batched_seeds =
        Batch.([
            ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % Ng], Val(B)) for
            a in 1:div(Ng, B, RoundUp)
        ])
    batched_results = Batch.([ntuple(b -> similar(y), Val(B)) for _ in batched_seeds])
    pushforward_batched_extras = prepare_pushforward_batched(
        f_or_f!y..., dense_backend, x, batched_seeds[1]
    )
    return PushforwardSparseJacobianExtras{B}(;
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pushforward_batched_extras,
    )
end

function prepare_sparse_jacobian_aux(
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
    seeds = [multibasis(backend, y, CartesianIndices(y)[group]) for group in groups]
    compressed_matrix = stack(_ -> vec(similar(x)), groups; dims=1)
    batched_seeds =
        Batch.([
            ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % Ng], Val(B)) for
            a in 1:div(Ng, B, RoundUp)
        ])
    batched_results = Batch.([ntuple(b -> similar(x), Val(B)) for _ in batched_seeds])
    pullback_batched_extras = prepare_pullback_batched(
        f_or_f!y..., dense_backend, x, batched_seeds[1]
    )
    return PullbackSparseJacobianExtras{B}(;
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pullback_batched_extras,
    )
end

## One argument

function jacobian(f::F, backend::AutoSparse, x, extras::SparseJacobianExtras) where {F}
    return sparse_jacobian_aux((f,), backend, x, extras)
end

function jacobian!(
    f::F, jac, backend::AutoSparse, x, extras::SparseJacobianExtras
) where {F}
    return sparse_jacobian_aux!((f,), jac, backend, x, extras)
end

function value_and_jacobian(
    f::F, backend::AutoSparse, x, extras::SparseJacobianExtras
) where {F}
    return f(x), jacobian(f, backend, x, extras)
end

function value_and_jacobian!(
    f::F, jac, backend::AutoSparse, x, extras::SparseJacobianExtras
) where {F}
    return f(x), jacobian!(f, jac, backend, x, extras)
end

## Two arguments

function jacobian(f!::F, y, backend::AutoSparse, x, extras::SparseJacobianExtras) where {F}
    return sparse_jacobian_aux((f!, y), backend, x, extras)
end

function jacobian!(
    f!::F, y, jac, backend::AutoSparse, x, extras::SparseJacobianExtras
) where {F}
    return sparse_jacobian_aux!((f!, y), jac, backend, x, extras)
end

function value_and_jacobian(
    f!::F, y, backend::AutoSparse, x, extras::SparseJacobianExtras
) where {F}
    jac = jacobian(f!, y, backend, x, extras)
    f!(y, x)
    return y, jac
end

function value_and_jacobian!(
    f!::F, y, jac, backend::AutoSparse, x, extras::SparseJacobianExtras
) where {F}
    jacobian!(f!, y, jac, backend, x, extras)
    f!(y, x)
    return y, jac
end

## Common auxiliaries

function sparse_jacobian_aux(
    f_or_f!y::FY, backend::AutoSparse, x, extras::PushforwardSparseJacobianExtras{B}
) where {FY,B}
    @compat (; coloring_result, batched_seeds, pushforward_batched_extras) = extras
    dense_backend = dense_ad(backend)
    Ng = length(column_groups(coloring_result))

    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f_or_f!y..., dense_backend, x, batched_seeds[1], pushforward_batched_extras
    )

    compressed_blocks = map(eachindex(batched_seeds)) do a
        dy_batch = pushforward_batched(
            f_or_f!y...,
            dense_backend,
            x,
            batched_seeds[a],
            pushforward_batched_extras_same,
        )
        stack(vec, dy_batch.elements; dims=2)
    end

    compressed_matrix = reduce(hcat, compressed_blocks)
    if Ng < size(compressed_matrix, 2)
        compressed_matrix = compressed_matrix[:, 1:Ng]
    end
    return decompress(compressed_matrix, coloring_result)
end

function sparse_jacobian_aux(
    f_or_f!y::FY, backend::AutoSparse, x, extras::PullbackSparseJacobianExtras{B}
) where {FY,B}
    @compat (; coloring_result, batched_seeds, pullback_batched_extras) = extras
    dense_backend = dense_ad(backend)
    Ng = length(row_groups(coloring_result))

    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f_or_f!y..., dense_backend, x, batched_seeds[1], pullback_batched_extras
    )

    compressed_blocks = map(eachindex(batched_seeds)) do a
        dx_batch = pullback_batched(
            f_or_f!y...,
            dense_backend,
            x,
            batched_seeds[a],
            pullback_batched_extras_same,
        )
        stack(vec, dx_batch.elements; dims=1)
    end

    compressed_matrix = reduce(vcat, compressed_blocks)
    if Ng < size(compressed_matrix, 1)
        compressed_matrix = compressed_matrix[1:Ng, :]
    end
    return decompress(compressed_matrix, coloring_result)
end

function sparse_jacobian_aux!(
    f_or_f!y::FY, jac, backend::AutoSparse, x, extras::PushforwardSparseJacobianExtras{B}
) where {FY,B}
    @compat (;
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pushforward_batched_extras,
    ) = extras
    dense_backend = dense_ad(backend)
    Ng = length(column_groups(coloring_result))

    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f_or_f!y..., dense_backend, x, batched_seeds[1], pushforward_batched_extras
    )

    for a in eachindex(batched_seeds, batched_results)
        pushforward_batched!(
            f_or_f!y...,
            batched_results[a],
            dense_backend,
            x,
            batched_seeds[a],
            pushforward_batched_extras_same,
        )

        for b in eachindex(batched_results[a].elements)
            copyto!(
                view(compressed_matrix, :, 1 + ((a - 1) * B + (b - 1)) % Ng),
                vec(batched_results[a].elements[b]),
            )
        end
    end

    decompress!(jac, compressed_matrix, coloring_result)
    return jac
end

function sparse_jacobian_aux!(
    f_or_f!y::FY, jac, backend::AutoSparse, x, extras::PullbackSparseJacobianExtras{B}
) where {FY,B}
    @compat (;
        coloring_result,
        compressed_matrix,
        batched_seeds,
        batched_results,
        pullback_batched_extras,
    ) = extras
    dense_backend = dense_ad(backend)
    Ng = length(row_groups(coloring_result))

    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f_or_f!y..., dense_backend, x, batched_seeds[1], pullback_batched_extras
    )

    for a in eachindex(batched_seeds, batched_results)
        pullback_batched!(
            f_or_f!y...,
            batched_results[a],
            dense_backend,
            x,
            batched_seeds[a],
            pullback_batched_extras_same,
        )

        for b in eachindex(batched_results[a].elements)
            copyto!(
                view(compressed_matrix, 1 + ((a - 1) * B + (b - 1)) % Ng, :),
                vec(batched_results[a].elements[b]),
            )
        end
    end

    decompress!(jac, compressed_matrix, coloring_result)
    return jac
end
