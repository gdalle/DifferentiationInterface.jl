Base.@kwdef struct SparseHessianExtras{
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
    hvp_extras::E2
    gradient_extras::E1
end

## Hessian, one argument

function prepare_hessian(f::F, backend::AutoSparse, x) where {F}
    dense_backend = dense_ad(backend)
    initial_sparsity = hessian_sparsity(f, x, sparsity_detector(backend))
    sparsity = col_major(initial_sparsity)
    colors = symmetric_coloring(sparsity, coloring_algorithm(backend))
    groups = color_groups(colors)
    seeds = map(groups) do group
        seed = zero(x)
        seed[group] .= one(eltype(x))
        seed
    end
    hvp_extras = prepare_hvp(f, dense_backend, x, first(seeds))
    products = map(seeds) do _
        similar(x)
    end
    compressed = stack(vec, products; dims=2)
    gradient_extras = prepare_gradient(f, maybe_inner(dense_backend), x)
    return SparseHessianExtras(;
        sparsity, compressed, colors, seeds, products, hvp_extras, gradient_extras
    )
end

function hessian!(f::F, hess, backend::AutoSparse, x, extras::SparseHessianExtras) where {F}
    @compat (; sparsity, compressed, colors, seeds, products, hvp_extras) = extras
    dense_backend = dense_ad(backend)
    hvp_extras_same = prepare_hvp_same_point(f, dense_backend, x, seeds[1], hvp_extras)
    for k in eachindex(seeds, products)
        hvp!(f, products[k], dense_backend, x, seeds[k], hvp_extras_same)
        copyto!(view(compressed, :, k), vec(products[k]))
    end
    decompress_symmetric!(hess, sparsity, compressed, colors)
    return hess
end

function hessian(f::F, backend::AutoSparse, x, extras::SparseHessianExtras) where {F}
    @compat (; sparsity, compressed, colors, seeds, products, hvp_extras) = extras
    dense_backend = dense_ad(backend)
    hvp_extras_same = prepare_hvp_same_point(f, dense_backend, x, seeds[1], hvp_extras)
    compressed = stack(eachindex(seeds, products); dims=2) do k
        vec(hvp(f, dense_backend, x, seeds[k], hvp_extras_same))
    end
    return decompress_symmetric(sparsity, compressed, colors)
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
    hess = hessian(f, hess, backend, x, extras)
    return y, grad, hess
end
