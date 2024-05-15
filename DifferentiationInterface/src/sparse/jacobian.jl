struct SparseJacobianExtras{
    args,dir,C<:CompressedMatrix{dir},S<:AbstractVector,P<:AbstractVector,E<:Extras
} <: JacobianExtras
    compressed::C
    seeds::S
    products::P
    jp_extras::E
end

function SparseJacobianExtras{args}(;
    compressed::CompressedMatrix{dir}, seeds::S, products::P, jp_extras::E
) where {args,dir,S,P,E}
    if dir == :col
        @assert jp_extras isa PushforwardExtras
    elseif dir == :row
        @assert jp_extras isa PullbackExtras
    end
    C = typeof(compressed)
    return SparseJacobianExtras{args,dir,C,S,P,E}(compressed, seeds, products, jp_extras)
end

## Jacobian, one argument

function prepare_jacobian(f::F, backend::AutoSparse, x) where {F}
    y = f(x)
    initial_sparsity = jacobian_sparsity(f, x, sparsity_detector(backend))
    if Bool(pushforward_performance(backend))
        sparsity = col_major(initial_sparsity)
        colors = column_coloring(sparsity, coloring_algorithm(backend))
        groups = color_groups(colors)
        seeds = map(groups) do group
            seed = zero(x)
            seed[group] .= one(eltype(x))
            seed
        end
        jp_extras = prepare_pushforward(f, backend, x, first(seeds))
        products = map(seeds) do seed
            similar(y)
        end
        aggregates = stack(vec, products; dims=2)
        compressed = CompressedMatrix{:col}(sparsity, colors, groups, aggregates)
    else
        sparsity = row_major(initial_sparsity)
        colors = row_coloring(sparsity, coloring_algorithm(backend))
        groups = color_groups(colors)
        seeds = map(groups) do group
            seed = zero(y)
            seed[group] .= one(eltype(y))
            seed
        end
        jp_extras = prepare_pullback(f, backend, x, first(seeds))
        products = map(seeds) do seed
            similar(x)
        end
        aggregates = stack(vec, products; dims=1)
        compressed = CompressedMatrix{:row}(sparsity, colors, groups, aggregates)
    end
    return SparseJacobianExtras{1}(; compressed, seeds, products, jp_extras)
end

function jacobian!(
    f::F, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{1,:col}
) where {F}
    @compat (; compressed, seeds, products, jp_extras) = extras
    pushforward_extras_same = prepare_pushforward_same_point(
        f, backend, x, seeds[1], jp_extras
    )
    for k in eachindex(seeds, products)
        pushforward!(f, products[k], backend, x, seeds[k], pushforward_extras_same)
        copyto!(view(compressed.aggregates, :, k), vec(products[k]))
    end
    decompress!(jac, compressed)
    return jac
end

function jacobian!(
    f::F, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{1,:row}
) where {F}
    @compat (; compressed, seeds, products, jp_extras) = extras
    pullback_extras_same = prepare_pullback_same_point(f, backend, x, seeds[1], jp_extras)
    for k in eachindex(seeds, products)
        pullback!(f, products[k], backend, x, seeds[k], pullback_extras_same)
        copyto!(view(compressed.aggregates, k, :), vec(products[k]))
    end
    decompress!(jac, compressed)
    return jac
end

function jacobian(f::F, backend::AutoSparse, x, extras::SparseJacobianExtras{1}) where {F}
    jac = major_respecting_similar(extras.compressed.sparsity, eltype(x))
    return jacobian!(f, jac, backend, x, extras)
end

function value_and_jacobian!(
    f::F, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{1}
) where {F}
    return f(x), jacobian!(f, jac, backend, x, extras)
end

function value_and_jacobian(
    f::F, backend::AutoSparse, x, extras::SparseJacobianExtras{1}
) where {F}
    return f(x), jacobian(f, backend, x, extras)
end

## Jacobian, two arguments

function prepare_jacobian(f!::F, y, backend::AutoSparse, x) where {F}
    initial_sparsity = jacobian_sparsity(f!, y, x, sparsity_detector(backend))
    if Bool(pushforward_performance(backend))
        sparsity = col_major(initial_sparsity)
        colors = column_coloring(sparsity, coloring_algorithm(backend))
        groups = color_groups(colors)
        seeds = map(groups) do group
            seed = zero(x)
            seed[group] .= one(eltype(x))
            seed
        end
        jp_extras = prepare_pushforward(f!, y, backend, x, first(seeds))
        products = map(seeds) do seed
            similar(y)
        end
        aggregates = stack(vec, products; dims=2)
        compressed = CompressedMatrix{:col}(sparsity, colors, groups, aggregates)
    else
        sparsity = row_major(initial_sparsity)
        colors = row_coloring(sparsity, coloring_algorithm(backend))
        groups = color_groups(colors)
        seeds = map(groups) do group
            seed = zero(y)
            seed[group] .= one(eltype(y))
            seed
        end
        jp_extras = prepare_pullback(f!, y, backend, x, first(seeds))
        products = map(seeds) do seed
            similar(x)
        end
        aggregates = stack(vec, products; dims=1)
        compressed = CompressedMatrix{:row}(sparsity, colors, groups, aggregates)
    end
    return SparseJacobianExtras{2}(; compressed, seeds, products, jp_extras)
end

function jacobian!(
    f!::F, y, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{2,:col}
) where {F}
    @compat (; compressed, seeds, products, jp_extras) = extras
    pushforward_extras_same = prepare_pushforward_same_point(
        f!, y, backend, x, seeds[1], jp_extras
    )
    for k in eachindex(seeds, products)
        pushforward!(f!, y, products[k], backend, x, seeds[k], pushforward_extras_same)
        copyto!(view(compressed.aggregates, :, k), vec(products[k]))
    end
    decompress!(jac, compressed)
    return jac
end

function jacobian!(
    f!::F, y, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{2,:row}
) where {F}
    @compat (; compressed, seeds, products, jp_extras) = extras
    pullback_extras_same = prepare_pullback_same_point(
        f!, y, backend, x, seeds[1], jp_extras
    )
    for k in eachindex(seeds, products)
        pullback!(f!, y, products[k], backend, x, seeds[k], pullback_extras_same)
        copyto!(view(compressed.aggregates, k, :), vec(products[k]))
    end
    decompress!(jac, compressed)
    return jac
end

function jacobian(
    f!::F, y, backend::AutoSparse, x, extras::SparseJacobianExtras{2}
) where {F}
    jac = major_respecting_similar(extras.compressed.sparsity, eltype(x))
    return jacobian!(f!, y, jac, backend, x, extras)
end

function value_and_jacobian!(
    f!::F, y, jac, backend::AutoSparse, x, extras::SparseJacobianExtras{2}
) where {F}
    jacobian!(f!, y, jac, backend, x, extras)
    f!(y, x)
    return y, jac
end

function value_and_jacobian(
    f!::F, y, backend::AutoSparse, x, extras::SparseJacobianExtras{2}
) where {F}
    jac = jacobian(f!, y, backend, x, extras)
    f!(y, x)
    return y, jac
end
