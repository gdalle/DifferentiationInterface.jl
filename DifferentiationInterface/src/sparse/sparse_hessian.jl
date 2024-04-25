## Hessian, one argument

function prepare_hessian(f, backend::AutoSparse, x)
    sparsity = hessian_sparsity(f, x, sparsity_detector(backend))
    symmetric_colors = symmetric_coloring(sparsity, coloring_algorithm(backend))
    symmetric_color_groups = get_groups(symmetric_colors)
    compressed_v = similar(x)
    compressed_col = similar(x)
    hvp_extras = prepare_hvp(f, backend, x, compressed_v)
    return (;
        sparsity,
        symmetric_colors,
        symmetric_color_groups,
        compressed_v,
        compressed_col,
        hvp_extras,
    )
end

function hessian!(f, hess, backend::AutoSparse, x, extras::NamedTuple)
    (; sparsity, symmetric_color_groups, compressed_v, compressed_col, hvp_extras) = extras
    for group in symmetric_color_groups
        compressed_v .= zero(eltype(compressed_v))
        for j in group
            compressed_v[j] = one(eltype(compressed_v))
        end
        hvp!(f, compressed_col, backend, x, compressed_v, hvp_extras)
        @views for j in group
            for i in axes(hess, 1)
                if (!iszero(sparsity[i, j]) && count(!iszero, sparsity[i, group]) == 1)
                    hess[i, j] = compressed_col[i]
                    hess[j, i] = compressed_col[i]
                end
            end
        end
    end
    return hess
end

function hessian(f, backend::AutoSparse, x, extras::NamedTuple)
    hess = similar(extras.sparsity, eltype(x))
    return hessian!(f, hess, backend, x, extras)
end
