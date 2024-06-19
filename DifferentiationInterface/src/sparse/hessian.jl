struct SparseHessianExtras{
    B,
    S<:AbstractMatrix{Bool},
    C<:AbstractMatrix{<:Real},
    K<:AbstractVector{<:Integer},
    D<:AbstractVector,
    P<:AbstractVector,
    E2<:HVPExtras,
    E1<:GradientExtras,
} <: HessianExtras
    sparsity::S
    compressed::C
    colors::K
    seeds::D
    products::P
    hvp_batched_extras::E2
    gradient_extras::E1
end

function SparseHessianExtras{B}(;
    sparsity::S,
    compressed::C,
    colors::K,
    seeds::D,
    products::P,
    hvp_batched_extras::E2,
    gradient_extras::E1,
) where {B,S,C,K,D,P,E2,E1}
    @assert length(seeds) == length(products) == size(compressed, 2)
    @assert size(sparsity, 1) == size(sparsity, 2) == size(compressed, 1) == length(colors)
    return SparseHessianExtras{B,S,C,K,D,P,E2,E1}(
        sparsity, compressed, colors, seeds, products, hvp_batched_extras, gradient_extras
    )
end

## Hessian, one argument

function prepare_hessian(f::F, backend::AutoSparse, x) where {F}
    dense_backend = dense_ad(backend)
    initial_sparsity = hessian_sparsity(f, x, sparsity_detector(backend))
    sparsity = col_major(initial_sparsity)
    colors = symmetric_coloring(sparsity, coloring_algorithm(backend))
    groups = color_groups(colors)
    seeds = map(group -> make_seed(x, group), groups)
    G = length(seeds)
    B = pick_batchsize(backend, G)
    dx_batch = Batch(ntuple(Returns(seeds[1]), Val(B)))
    hvp_batched_extras = prepare_hvp_batched(f, dense_backend, x, dx_batch)
    products = map(_ -> similar(x), seeds)
    compressed = stack(vec, products; dims=2)
    gradient_extras = prepare_gradient(f, maybe_inner(dense_backend), x)
    return SparseHessianExtras{B}(;
        sparsity, compressed, colors, seeds, products, hvp_batched_extras, gradient_extras
    )
end

function hessian(f::F, backend::AutoSparse, x, extras::SparseHessianExtras{B}) where {F,B}
    @compat (; sparsity, compressed, colors, seeds, products, hvp_batched_extras) = extras
    G = length(seeds)
    dense_backend = dense_ad(backend)

    hvp_batched_extras_same = prepare_hvp_batched_same_point(
        f, dense_backend, x, Batch(ntuple(Returns(seeds[1]), Val(B))), hvp_batched_extras
    )

    compressed = mapreduce(hcat, 1:div(G, B, RoundUp)) do a
        dx_batch_elements = ntuple(Val(B)) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % G]
        end
        dg_batch = hvp_batched(
            f, dense_backend, x, Batch(dx_batch_elements), hvp_batched_extras_same
        )
        stack(vec, dg_batch.elements; dims=2)
    end

    if G < size(compressed, 2)
        compressed = compressed[:, 1:G]
    end
    return decompress_symmetric(sparsity, compressed, colors)
end

function hessian!(
    f::F, hess, backend::AutoSparse, x, extras::SparseHessianExtras{B}
) where {F,B}
    @compat (; sparsity, compressed, colors, seeds, products, hvp_batched_extras) = extras
    dense_backend = dense_ad(backend)
    G = length(seeds)

    hvp_batched_extras_same = prepare_hvp_batched_same_point(
        f, dense_backend, x, Batch(ntuple(Returns(seeds[1]), Val(B))), hvp_batched_extras
    )

    for a in 1:div(G, B, RoundUp)
        dx_batch_elements = ntuple(Val(B)) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % G]
        end
        dg_batch_elements = ntuple(Val(B)) do l
            products[1 + ((a - 1) * B + (b - 1)) % G]
        end
        hvp_batched!(
            f,
            Batch(dg_batch_elements),
            dense_backend,
            x,
            Batch(dx_batch_elements),
            hvp_batched_extras_same,
        )
    end

    for k in eachindex(products)
        copyto!(view(compressed, :, k), vec(products[k]))
    end

    decompress_symmetric!(hess, sparsity, compressed, colors)
    return hess
end

function value_gradient_and_hessian!(
    f::F, grad, hess, backend::AutoSparse, x, extras::SparseHessianExtras
) where {F}
    y, _ = value_and_gradient!(
        f, grad, maybe_inner(dense_ad(backend)), x, extras.gradient_extras
    )
    hessian!(f, hess, backend, x, extras)
    return y, grad, hess
end

function value_gradient_and_hessian(
    f::F, backend::AutoSparse, x, extras::SparseHessianExtras
) where {F}
    y, grad = value_and_gradient(
        f, maybe_inner(dense_ad(backend)), x, extras.gradient_extras
    )
    hess = hessian(f, backend, x, extras)
    return y, grad, hess
end
