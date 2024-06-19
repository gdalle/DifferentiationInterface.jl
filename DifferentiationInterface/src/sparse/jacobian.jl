## Preparation

abstract type SparseJacobianExtras <: JacobianExtras end

struct PushforwardSparseJacobianExtras{
    B,
    S<:AbstractMatrix{Bool},
    C<:AbstractMatrix{<:Real},
    K<:AbstractVector{<:Integer},
    D<:AbstractVector,
    P<:AbstractVector,
    E<:PushforwardExtras,
} <: SparseJacobianExtras
    sparsity::S
    compressed::C
    colors::K
    seeds::D
    products::P
    pushforward_batched_extras::E
end

struct PullbackSparseJacobianExtras{
    B,
    S<:AbstractMatrix{Bool},
    C<:AbstractMatrix{<:Real},
    K<:AbstractVector{<:Integer},
    D<:AbstractVector,
    P<:AbstractVector,
    E<:PullbackExtras,
} <: SparseJacobianExtras
    sparsity::S
    compressed::C
    colors::K
    seeds::D
    products::P
    pullback_batched_extras::E
end

function PushforwardSparseJacobianExtras{B}(;
    sparsity::S,
    compressed::C,
    colors::K,
    seeds::D,
    products::P,
    pushforward_batched_extras::E,
) where {B,S,C,K,D,P,E}
    @assert length(seeds) == length(products) == size(compressed, 2)
    @assert size(sparsity, 1) == size(compressed, 1)
    @assert size(sparsity, 2) == length(colors)
    return PushforwardSparseJacobianExtras{B,S,C,K,D,P,E}(
        sparsity, compressed, colors, seeds, products, pushforward_batched_extras
    )
end

function PullbackSparseJacobianExtras{B}(;
    sparsity::S, compressed::C, colors::K, seeds::D, products::P, pullback_batched_extras::E
) where {B,S,C,K,D,P,E}
    @assert length(seeds) == length(products) == size(compressed, 1)
    @assert size(sparsity, 2) == size(compressed, 2)
    @assert size(sparsity, 1) == length(colors)
    return PullbackSparseJacobianExtras{B,S,C,K,D,P,E}(
        sparsity, compressed, colors, seeds, products, pullback_batched_extras
    )
end

function prepare_jacobian(f::F, backend::AutoSparse, x) where {F}
    return prepare_sparse_jacobian_aux(f, backend, x, pushforward_performance(backend))
end

function prepare_jacobian(f!::F, y, backend::AutoSparse, x) where {F}
    return prepare_sparse_jacobian_aux(f!, y, backend, x, pushforward_performance(backend))
end

function prepare_sparse_jacobian_aux(f::F, backend, x, ::PushforwardFast) where {F}
    y = f(x)
    dense_backend = dense_ad(backend)
    initial_sparsity = jacobian_sparsity(f, x, sparsity_detector(backend))
    sparsity = col_major(initial_sparsity)
    colors = column_coloring(sparsity, coloring_algorithm(backend))
    groups = color_groups(colors)
    seeds = map(group -> make_seed(x, group), groups)
    G = length(seeds)
    B = pick_batchsize(backend, G)
    dx_batch = Batch(ntuple(Returns(seeds[1]), Val{B}()))
    pushforward_batched_extras = prepare_pushforward_batched(f, dense_backend, x, dx_batch)
    products = map(_ -> similar(y), seeds)
    compressed = stack(vec, products; dims=2)
    return PushforwardSparseJacobianExtras{B}(;
        sparsity, compressed, colors, seeds, products, pushforward_batched_extras
    )
end

function prepare_sparse_jacobian_aux(f::F, backend, x, ::PushforwardSlow) where {F}
    y = f(x)
    dense_backend = dense_ad(backend)
    initial_sparsity = jacobian_sparsity(f, x, sparsity_detector(backend))
    sparsity = row_major(initial_sparsity)
    colors = row_coloring(sparsity, coloring_algorithm(backend))
    groups = color_groups(colors)
    seeds = map(group -> make_seed(y, group), groups)
    G = length(seeds)
    B = pick_batchsize(backend, G)
    dx_batch = Batch(ntuple(Returns(seeds[1]), Val{B}()))
    pullback_batched_extras = prepare_pullback_batched(f, dense_backend, x, dx_batch)
    products = map(_ -> similar(x), seeds)
    compressed = stack(vec, products; dims=1)
    return PullbackSparseJacobianExtras{B}(;
        sparsity, compressed, colors, seeds, products, pullback_batched_extras
    )
end

function prepare_sparse_jacobian_aux(f!::F, y, backend, x, ::PushforwardFast) where {F}
    dense_backend = dense_ad(backend)
    initial_sparsity = jacobian_sparsity(f!, y, x, sparsity_detector(backend))
    sparsity = col_major(initial_sparsity)
    colors = column_coloring(sparsity, coloring_algorithm(backend))
    groups = color_groups(colors)
    seeds = map(group -> make_seed(x, group), groups)
    G = length(seeds)
    B = pick_batchsize(backend, G)
    dx_batch = Batch(ntuple(Returns(seeds[1]), Val{B}()))
    pushforward_batched_extras = prepare_pushforward_batched(
        f!, y, dense_backend, x, dx_batch
    )
    products = map(_ -> similar(y), seeds)
    compressed = stack(vec, products; dims=2)
    return PushforwardSparseJacobianExtras{B}(;
        sparsity, compressed, colors, seeds, products, pushforward_batched_extras
    )
end

function prepare_sparse_jacobian_aux(f!::F, y, backend, x, ::PushforwardSlow) where {F}
    dense_backend = dense_ad(backend)
    initial_sparsity = jacobian_sparsity(f!, y, x, sparsity_detector(backend))
    sparsity = row_major(initial_sparsity)
    colors = row_coloring(sparsity, coloring_algorithm(backend))
    groups = color_groups(colors)
    seeds = map(group -> make_seed(y, group), groups)
    G = length(seeds)
    B = pick_batchsize(backend, G)
    dx_batch = Batch(ntuple(Returns(seeds[1]), Val{B}()))
    pullback_batched_extras = prepare_pullback_batched(f!, y, dense_backend, x, dx_batch)
    products = map(_ -> similar(x), seeds)
    compressed = stack(vec, products; dims=1)
    return PullbackSparseJacobianExtras{B}(;
        sparsity, compressed, colors, seeds, products, pullback_batched_extras
    )
end

## One argument

function jacobian!(
    f::F, jac, backend::AutoSparse, x, extras::PushforwardSparseJacobianExtras{B}
) where {F,B}
    @compat (; sparsity, compressed, colors, seeds, products, pushforward_batched_extras) =
        extras
    dense_backend = dense_ad(backend)
    G = length(seeds)

    example_dx_batch_elements = ntuple(Returns(seeds[1]), Val{B}())
    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f, dense_backend, x, Batch(example_dx_batch_elements), pushforward_batched_extras
    )

    for a in 1:div(G, B, RoundUp)
        dx_batch_elements = ntuple(Val{B}()) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % G]
        end
        dy_batch_elements = ntuple(Val{B}()) do l
            products[1 + ((a - 1) * B + (b - 1)) % G]
        end
        pushforward_batched!(
            f,
            Batch(dy_batch_elements),
            backend,
            x,
            Batch(dx_batch_elements),
            pushforward_batched_extras_same,
        )
    end

    for k in eachindex(products)
        copyto!(view(compressed, :, k), vec(products[k]))
    end

    decompress_columns!(jac, sparsity, compressed, colors)
    return jac
end

function jacobian!(
    f::F, jac, backend::AutoSparse, x, extras::PullbackSparseJacobianExtras{B}
) where {F,B}
    @compat (; sparsity, compressed, colors, seeds, products, pullback_batched_extras) =
        extras
    dense_backend = dense_ad(backend)
    G = length(seeds)

    example_dy_batch_elements = ntuple(Returns(seeds[1]), Val{B}())
    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f, dense_backend, x, Batch(example_dy_batch_elements), pullback_batched_extras
    )

    for a in 1:div(G, B, RoundUp)
        dy_batch_elements = ntuple(Val{B}()) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % G]
        end
        dx_batch_elements = ntuple(Val{B}()) do l
            products[1 + ((a - 1) * B + (b - 1)) % G]
        end
        pullback_batched!(
            f,
            Batch(dx_batch_elements),
            backend,
            x,
            Batch(dy_batch_elements),
            pullback_batched_extras_same,
        )
    end

    for k in eachindex(products)
        copyto!(view(compressed, k, :), vec(products[k]))
    end

    decompress_rows!(jac, sparsity, compressed, colors)
    return jac
end

function jacobian(
    f::F, backend::AutoSparse, x, extras::PushforwardSparseJacobianExtras{B}
) where {F,B}
    @compat (; sparsity, compressed, colors, seeds, products, pushforward_batched_extras) =
        extras
    dense_backend = dense_ad(backend)
    G = length(seeds)

    example_dx_batch_elements = ntuple(Returns(seeds[1]), Val{B}())
    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f, dense_backend, x, Batch(example_dx_batch_elements), pushforward_batched_extras
    )

    compressed = mapreduce(hcat, 1:div(G, B, RoundUp)) do a
        dx_batch_elements = ntuple(Val{B}()) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % G]
        end
        dy_batch = pushforward_batched(
            f, backend, x, Batch(dx_batch_elements), pushforward_batched_extras_same
        )
        stack(vec, dy_batch.elements; dims=2)
    end

    if G < size(compressed, 2)
        compressed = compressed[:, 1:G]
    end

    return decompress_columns(sparsity, compressed, colors)
end

function jacobian(
    f::F, backend::AutoSparse, x, extras::PullbackSparseJacobianExtras{B}
) where {F,B}
    @compat (; sparsity, compressed, colors, seeds, products, pullback_batched_extras) =
        extras
    dense_backend = dense_ad(backend)
    G = length(seeds)

    example_dy_batch_elements = ntuple(Returns(seeds[1]), Val{B}())
    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f, dense_backend, x, Batch(example_dy_batch_elements), pullback_batched_extras
    )

    compressed = mapreduce(hcat, 1:div(G, B, RoundUp)) do a
        dy_batch_elements = ntuple(Val{B}()) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % G]
        end
        dx_batch = pullback_batched(
            f, backend, x, Batch(dy_batch_elements), pullback_batched_extras_same
        )
        stack(vec, dx_batch.elements; dims=1)
    end

    if G < size(compressed, 1)
        compressed = compressed[1:G, :]
    end

    return decompress_rows(sparsity, compressed, colors)
end

function value_and_jacobian!(
    f::F, jac, backend::AutoSparse, x, extras::SparseJacobianExtras
) where {F}
    return f(x), jacobian!(f, jac, backend, x, extras)
end

function value_and_jacobian(
    f::F, backend::AutoSparse, x, extras::SparseJacobianExtras
) where {F}
    return f(x), jacobian(f, backend, x, extras)
end

## Two arguments

function jacobian!(
    f!::F, y, jac, backend::AutoSparse, x, extras::PushforwardSparseJacobianExtras{B}
) where {F,B}
    @compat (; sparsity, compressed, colors, seeds, products, pushforward_batched_extras) =
        extras
    dense_backend = dense_ad(backend)
    G = length(seeds)

    example_dx_batch_elements = ntuple(Returns(seeds[1]), Val{B}())
    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f!,
        y,
        dense_backend,
        x,
        Batch(example_dx_batch_elements),
        pushforward_batched_extras,
    )

    for a in 1:div(G, B, RoundUp)
        dx_batch_elements = ntuple(Val{B}()) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % G]
        end
        dy_batch_elements = ntuple(Val{B}()) do l
            products[1 + ((a - 1) * B + (b - 1)) % G]
        end
        pushforward_batched!(
            f!,
            y,
            Batch(dy_batch_elements),
            backend,
            x,
            Batch(dx_batch_elements),
            pushforward_batched_extras_same,
        )
    end

    for k in eachindex(products)
        copyto!(view(compressed, :, k), vec(products[k]))
    end
    decompress_columns!(jac, sparsity, compressed, colors)
    return jac
end

function jacobian!(
    f!::F, y, jac, backend::AutoSparse, x, extras::PullbackSparseJacobianExtras{B}
) where {F,B}
    @compat (; sparsity, compressed, colors, seeds, products, pullback_batched_extras) =
        extras
    G = length(seeds)

    example_dy_batch_elements = ntuple(Returns(seeds[1]), Val{B}())
    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f!, y, dense_backend, x, Batch(example_dy_batch_elements), pullback_batched_extras
    )

    for a in 1:div(G, B, RoundUp)
        dy_batch_elements = ntuple(Val{B}()) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % G]
        end
        dx_batch_elements = ntuple(Val{B}()) do l
            products[1 + ((a - 1) * B + (b - 1)) % G]
        end
        pullback_batched!(
            f!,
            y,
            Batch(dx_batch_elements),
            backend,
            x,
            Batch(dy_batch_elements),
            pullback_batched_extras_same,
        )
    end

    for k in eachindex(products)
        copyto!(view(compressed, k, :), vec(products[k]))
    end
    decompress_rows!(jac, sparsity, compressed, colors)
    return jac
end

function jacobian(
    f!::F, y, backend::AutoSparse, x, extras::PushforwardSparseJacobianExtras{B}
) where {F,B}
    @compat (; sparsity, compressed, colors, seeds, products, pushforward_batched_extras) =
        extras
    dense_backend = dense_ad(backend)
    G = length(seeds)

    example_dx_batch_elements = ntuple(Returns(seeds[1]), Val{B}())
    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f!,
        y,
        dense_backend,
        x,
        Batch(example_dx_batch_elements),
        pushforward_batched_extras,
    )

    compressed = mapreduce(hcat, 1:div(G, B, RoundUp)) do a
        dx_batch_elements = ntuple(Val{B}()) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % G]
        end
        dy_batch = pushforward_batched(
            f!, y, backend, x, Batch(dx_batch_elements), pushforward_batched_extras_same
        )
        stack(vec, dy_batch.elements; dims=2)
    end

    if G < size(compressed, 2)
        compressed = compressed[:, 1:G]
    end

    return decompress_columns(sparsity, compressed, colors)
end

function jacobian(
    f!::F, y, backend::AutoSparse, x, extras::PullbackSparseJacobianExtras{B}
) where {F,B}
    @compat (; sparsity, compressed, colors, seeds, products, pullback_batched_extras) =
        extras
    dense_backend = dense_ad(backend)
    G = length(seeds)

    example_dy_batch_elements = ntuple(Returns(seeds[1]), Val{B}())
    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f!, y, dense_backend, x, Batch(example_dy_batch_elements), pullback_batched_extras
    )

    compressed = mapreduce(hcat, 1:div(G, B, RoundUp)) do a
        dy_batch_elements = ntuple(Val{B}()) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % G]
        end
        dx_batch = pullback_batched(
            f!, y, backend, x, Batch(dy_batch_elements), pullback_batched_extras_same
        )
        stack(vec, dx_batch.elements; dims=1)
    end

    if G < size(compressed, 1)
        compressed = compressed[1:G, :]
    end
    return decompress_rows(sparsity, compressed, colors)
end

function value_and_jacobian!(
    f!::F, y, jac, backend::AutoSparse, x, extras::SparseJacobianExtras
) where {F}
    jacobian!(f!, y, jac, backend, x, extras)
    f!(y, x)
    return y, jac
end

function value_and_jacobian(
    f!::F, y, backend::AutoSparse, x, extras::SparseJacobianExtras
) where {F}
    jac = jacobian(f!, y, backend, x, extras)
    f!(y, x)
    return y, jac
end
