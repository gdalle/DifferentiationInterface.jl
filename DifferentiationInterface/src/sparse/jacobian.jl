## Preparation

abstract type SparseJacobianExtras <: JacobianExtras end

struct PushforwardSparseJacobianExtras{
    B,S<:AbstractMatrix{Bool},C<:AbstractMatrix{<:Real},D,R,E<:PushforwardExtras
} <: SparseJacobianExtras
    sparsity::S
    colors::Vector{Int}
    groups::Vector{Vector{Int}}
    compressed::C
    batched_seeds::Vector{Batch{B,D}}
    batched_results::Vector{Batch{B,R}}
    pushforward_batched_extras::E
end

struct PullbackSparseJacobianExtras{
    B,S<:AbstractMatrix{Bool},C<:AbstractMatrix{<:Real},D,R,E<:PullbackExtras
} <: SparseJacobianExtras
    sparsity::S
    colors::Vector{Int}
    groups::Vector{Vector{Int}}
    compressed::C
    batched_seeds::Vector{Batch{B,D}}
    batched_results::Vector{Batch{B,R}}
    pullback_batched_extras::E
end

function PushforwardSparseJacobianExtras{B}(;
    sparsity::S,
    colors,
    groups,
    compressed::C,
    batched_seeds::Vector{Batch{B,D}},
    batched_results::Vector{Batch{B,R}},
    pushforward_batched_extras::E,
) where {B,S,C,D,R,E}
    @assert size(sparsity, 1) == size(compressed, 1)
    @assert size(sparsity, 2) == length(colors)
    return PushforwardSparseJacobianExtras{B,S,C,D,R,E}(
        sparsity,
        colors,
        groups,
        compressed,
        batched_seeds,
        batched_results,
        pushforward_batched_extras,
    )
end

function PullbackSparseJacobianExtras{B}(;
    sparsity::S,
    colors,
    groups,
    compressed::C,
    batched_seeds::Vector{Batch{B,D}},
    batched_results::Vector{Batch{B,R}},
    pullback_batched_extras::E,
) where {B,S,C,D,R,E}
    @assert size(sparsity, 2) == size(compressed, 2)
    @assert size(sparsity, 1) == length(colors)
    return PullbackSparseJacobianExtras{B,S,C,D,R,E}(
        sparsity,
        colors,
        groups,
        compressed,
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
    f_or_f!y::FY, backend, x, y, ::PushforwardFast
) where {FY}
    dense_backend = dense_ad(backend)
    initial_sparsity = jacobian_sparsity(f_or_f!y..., x, sparsity_detector(backend))
    sparsity = col_major(initial_sparsity)
    colors = column_coloring(sparsity, coloring_algorithm(backend))
    groups = color_groups(colors)
    Ng = length(groups)
    B = pick_batchsize(dense_backend, Ng)
    seeds = map(group -> make_seed(x, group), groups)
    compressed = stack(_ -> vec(similar(y)), groups; dims=2)
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
        sparsity,
        colors,
        groups,
        compressed,
        batched_seeds,
        batched_results,
        pushforward_batched_extras,
    )
end

function prepare_sparse_jacobian_aux(
    f_or_f!y::FY, backend, x, y, ::PushforwardSlow
) where {FY}
    dense_backend = dense_ad(backend)
    initial_sparsity = jacobian_sparsity(f_or_f!y..., x, sparsity_detector(backend))
    sparsity = row_major(initial_sparsity)
    colors = row_coloring(sparsity, coloring_algorithm(backend))
    groups = color_groups(colors)
    Ng = length(groups)
    B = pick_batchsize(dense_backend, Ng)
    seeds = map(group -> make_seed(y, group), groups)
    compressed = stack(_ -> vec(similar(x)), groups; dims=1)
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
        sparsity,
        colors,
        groups,
        compressed,
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
    @compat (;
        sparsity, compressed, colors, groups, batched_seeds, pushforward_batched_extras
    ) = extras
    dense_backend = dense_ad(backend)
    Ng = length(groups)

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

    compressed = reduce(hcat, compressed_blocks)
    if Ng < size(compressed, 2)
        compressed = compressed[:, 1:Ng]
    end
    return decompress_columns(sparsity, compressed, colors)
end

function sparse_jacobian_aux(
    f_or_f!y::FY, backend::AutoSparse, x, extras::PullbackSparseJacobianExtras{B}
) where {FY,B}
    @compat (;
        sparsity, compressed, colors, groups, batched_seeds, pullback_batched_extras
    ) = extras
    dense_backend = dense_ad(backend)
    Ng = length(groups)

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

    compressed = reduce(vcat, compressed_blocks)
    if Ng < size(compressed, 1)
        compressed = compressed[1:Ng, :]
    end
    return decompress_rows(sparsity, compressed, colors)
end

function sparse_jacobian_aux!(
    f_or_f!y::FY, jac, backend::AutoSparse, x, extras::PushforwardSparseJacobianExtras{B}
) where {FY,B}
    @compat (;
        sparsity,
        compressed,
        colors,
        groups,
        batched_seeds,
        batched_results,
        pushforward_batched_extras,
    ) = extras
    dense_backend = dense_ad(backend)
    Ng = length(groups)

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
    end

    for a in eachindex(batched_results)
        for b in eachindex(batched_results[a].elements)
            copyto!(
                view(compressed, :, 1 + ((a - 1) * B + (b - 1)) % Ng),
                vec(batched_results[a].elements[b]),
            )
        end
    end

    decompress_columns!(jac, sparsity, compressed, colors)
    return jac
end

function sparse_jacobian_aux!(
    f_or_f!y::FY, jac, backend::AutoSparse, x, extras::PullbackSparseJacobianExtras{B}
) where {FY,B}
    @compat (;
        sparsity,
        compressed,
        colors,
        groups,
        batched_seeds,
        batched_results,
        pullback_batched_extras,
    ) = extras
    dense_backend = dense_ad(backend)
    Ng = length(groups)

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
    end

    for a in eachindex(batched_results)
        for b in eachindex(batched_results[a].elements)
            copyto!(
                view(compressed, 1 + ((a - 1) * B + (b - 1)) % Ng, :),
                vec(batched_results[a].elements[b]),
            )
        end
    end

    decompress_rows!(jac, sparsity, compressed, colors)
    return jac
end
