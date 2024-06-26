struct SparseHessianExtras{
    B,S<:AbstractMatrix{Bool},C<:AbstractMatrix{<:Real},D,E2<:HVPExtras,E1<:GradientExtras
} <: HessianExtras
    sparsity::S
    colors::Vector{Int}
    groups::Vector{Vector{Int}}
    compressed::C
    batched_seeds::Vector{Batch{B,D}}
    hvp_batched_extras::E2
    gradient_extras::E1
end

function SparseHessianExtras{B}(;
    sparsity::S,
    colors,
    groups,
    compressed::C,
    batched_seeds::Vector{Batch{B,D}},
    hvp_batched_extras::E2,
    gradient_extras::E1,
) where {B,S,C,D,E2,E1}
    @assert size(sparsity, 1) == size(sparsity, 2) == size(compressed, 1) == length(colors)
    return SparseHessianExtras{B,S,C,D,E2,E1}(
        sparsity,
        colors,
        groups,
        compressed,
        batched_seeds,
        hvp_batched_extras,
        gradient_extras,
    )
end

## Hessian, one argument

function prepare_hessian(f::F, backend::AutoSparse, x) where {F}
    dense_backend = dense_ad(backend)
    initial_sparsity = hessian_sparsity(f, x, sparsity_detector(backend))
    sparsity = col_major(initial_sparsity)
    colors = symmetric_coloring(sparsity, coloring_algorithm(backend))
    groups = color_groups(colors)
    Ng = length(groups)
    B = pick_batchsize(maybe_outer(dense_backend), Ng)
    seeds = map(group -> make_seed(x, group), groups)
    compressed = stack(_ -> vec(similar(x)), groups; dims=2)
    batched_seeds =
        Batch.([
            ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % Ng], Val(B)) for
            a in 1:div(Ng, B, RoundUp)
        ])
    hvp_batched_extras = prepare_hvp_batched(f, dense_backend, x, batched_seeds[1])
    gradient_extras = prepare_gradient(f, maybe_inner(dense_backend), x)
    return SparseHessianExtras{B}(;
        sparsity,
        colors,
        groups,
        compressed,
        batched_seeds,
        hvp_batched_extras,
        gradient_extras,
    )
end

function hessian(f::F, backend::AutoSparse, x, extras::SparseHessianExtras{B}) where {F,B}
    @compat (; sparsity, compressed, colors, groups, batched_seeds, hvp_batched_extras) =
        extras
    dense_backend = dense_ad(backend)
    Ng = length(groups)

    hvp_batched_extras_same = prepare_hvp_batched_same_point(
        f, dense_backend, x, batched_seeds[1], hvp_batched_extras
    )

    compressed_blocks = map(eachindex(batched_seeds)) do a
        dg_batch = hvp_batched(f, dense_backend, x, batched_seeds[a], hvp_batched_extras_same)
        stack(vec, dg_batch.elements; dims=2)
    end

    compressed = reduce(hcat, compressed_blocks)
    if Ng < size(compressed, 2)
        compressed = compressed[:, 1:Ng]
    end
    return decompress_symmetric(sparsity, compressed, colors)
end

function hessian!(
    f::F, hess, backend::AutoSparse, x, extras::SparseHessianExtras{B}
) where {F,B}
    @compat (; sparsity, compressed, colors, groups, batched_seeds, hvp_batched_extras) =
        extras
    dense_backend = dense_ad(backend)
    Ng = length(groups)

    hvp_batched_extras_same = prepare_hvp_batched_same_point(
        f, dense_backend, x, batched_seeds[1], hvp_batched_extras
    )

    for a in 1:div(Ng, B, RoundUp)
        dg_batch_elements = ntuple(Val(B)) do b
            reshape(view(compressed, :, 1 + ((a - 1) * B + (b - 1)) % Ng), size(x))
        end
        hvp_batched!(
            f,
            Batch(dg_batch_elements),
            dense_backend,
            x,
            batched_seeds[a],
            hvp_batched_extras_same,
        )
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
