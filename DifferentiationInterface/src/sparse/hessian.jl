Base.@kwdef struct SparseHessianExtras{
    C<:CompressedMatrix{:col},S<:AbstractVector,P<:AbstractVector,E<:Extras
} <: HessianExtras
    compressed::C
    seeds::S
    products::P
    hvp_extras::E
end

## Hessian, one argument

function prepare_hessian(f::F, backend::AutoSparse, x) where {F}
    initial_sparsity = hessian_sparsity(f, x, sparsity_detector(backend))
    sparsity = col_major(initial_sparsity)
    colors = column_coloring(sparsity, coloring_algorithm(backend))  # no star coloring
    groups = color_groups(colors)
    seeds = map(groups) do group
        seed = zero(x)
        seed[group] .= one(eltype(x))
        seed
    end
    hvp_extras = prepare_hvp(f, backend, x, first(seeds))
    products = map(seeds) do seed
        similar(x)
    end
    aggregates = stack(vec, products; dims=2)
    compressed = CompressedMatrix{:col}(sparsity, colors, groups, aggregates)
    return SparseHessianExtras(; compressed, seeds, products, hvp_extras)
end

function hessian!(f::F, hess, backend::AutoSparse, x, extras::SparseHessianExtras) where {F}
    @compat (; compressed, seeds, products, hvp_extras) = extras
    hvp_extras_same = prepare_hvp_same_point(f, backend, x, seeds[1], hvp_extras)
    for k in eachindex(seeds, products)
        hvp!(f, products[k], backend, x, seeds[k], hvp_extras_same)
        copyto!(view(compressed.aggregates, :, k), vec(products[k]))
    end
    decompress!(hess, compressed)
    return hess
end

function hessian(f::F, backend::AutoSparse, x, extras::SparseHessianExtras) where {F}
    hess = similar(extras.compressed.sparsity, eltype(x))
    return hessian!(f, hess, backend, x, extras)
end
