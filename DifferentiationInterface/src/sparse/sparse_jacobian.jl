## Jacobian, one argument (based on `value_and_jacobian!`)

struct SparseOneArgJacobianExtras{dir,S,C<:CompressedMatrix{dir},D,E<:Extras} <:
       JacobianExtras
    sparsity::S
    colors::Vector{Int}
    groups::Vector{Vector{Int}}
    J_compressed::C
    seed:D
    extras::E
end

function SparseOneArgJacobianExtras{dir}(;
    sparsity::S, colors, groups, J_compressed::C, seed::D, extras::E
) where {dir,S,C,E}
    return SparseOneArgJacobianExtras{dir,S,C,D,E}(
        sparsity, colors, groups, J_compressed, seed, extras
    )
end

function prepare_jacobian(f, backend::AutoSparse, x)
    return sparse_prepare_jacobian_aux(f, backend, x, pushforward_performance(backend))
end

function sparse_prepare_jacobian_aux(f, backend, x, ::PushforwardFast)
    y = f(x)
    sparsity = jacobian_sparsity(f, x, sparsity_detector(backend))
    colors = column_coloring(sparsity, coloring_algorithm(backend))
    groups = get_groups(colors)
    seed = similar(x)
    extras = prepare_pushforward(f, backend, x, seed)
    J_compressed = stack(groups) do g
        pushforward(f, backend, x, seed)
    end
    return (;
        sparsity,
        column_colors,
        column_color_groups,
        compressed_dx,
        compressed_col,
        pushforward_extras,
    )
end

function sparse_prepare_jacobian_aux(f, backend, x, ::PushforwardSlow)
    y = f(x)
    sparsity = jacobian_sparsity(f, x, sparsity_detector(backend))
    row_colors = row_coloring(sparsity, coloring_algorithm(backend))
    row_color_groups = get_groups(row_colors)
    compressed_dy = similar(y)
    compressed_row = similar(x)
    pullback_extras = prepare_pullback(f, backend, x, compressed_dy)
    return (;
        sparsity,
        row_colors,
        row_color_groups,
        compressed_dy,
        compressed_row,
        pullback_extras,
    )
end

function value_and_jacobian!(f, jac, backend::AutoSparse, x, extras::NamedTuple)
    return sparse_value_and_jacobian_aux!(
        f, jac, backend, x, extras, pushforward_performance(backend)
    )
end

function sparse_value_and_jacobian_aux!(f, jac, backend, x, extras, ::PushforwardFast)
    (; sparsity, column_color_groups, compressed_dx, compressed_col, pushforward_extras) =
        extras
    y = f(x)
    for group in column_color_groups
        compressed_dx .= zero(eltype(compressed_dx))
        for j in group
            compressed_dx[j] = one(eltype(compressed_dx))
        end
        pushforward!(f, compressed_col, backend, x, compressed_dx, pushforward_extras)
        @views for j in group
            nonzero_rows_j = (!iszero).(sparsity[:, j])
            copyto!(jac[nonzero_rows_j, j], compressed_col[nonzero_rows_j])
        end
    end
    return y, jac
end

function sparse_value_and_jacobian_aux!(f, jac, backend, x, extras, ::PushforwardSlow)
    (; sparsity, row_color_groups, compressed_dy, compressed_row, pullback_extras) = extras
    y = f(x)
    for group in row_color_groups
        compressed_dy .= zero(eltype(compressed_dy))
        for i in group
            compressed_dy[i] = one(eltype(compressed_dy))
        end
        pullback!(f, compressed_row, backend, x, compressed_dy, pullback_extras)
        @views for i in group
            nonzero_columns_i = (!iszero).(sparsity[i, :])
            copyto!(jac[i, nonzero_columns_i], compressed_row[nonzero_columns_i])
        end
    end
    return y, jac
end

function value_and_jacobian(f, backend::AutoSparse, x, extras::NamedTuple)
    jac = similar(extras.sparsity, eltype(x))
    return value_and_jacobian!(f, jac, backend, x, extras)
end

function jacobian!(f, jac, backend::AutoSparse, x, extras::NamedTuple)
    return value_and_jacobian!(f, jac, backend, x, extras)[2]
end

function jacobian(f, backend::AutoSparse, x, extras::NamedTuple)
    return value_and_jacobian(f, backend, x, extras)[2]
end

## Jacobian, two arguments (based on `jacobian!`)

function prepare_jacobian(f!, y, backend::AutoSparse, x)
    return sparse_prepare_jacobian_aux(f!, y, backend, x, pushforward_performance(backend))
end

function sparse_prepare_jacobian_aux(f!, y, backend, x, ::PushforwardFast)
    sparsity = jacobian_sparsity(f!, y, x, sparsity_detector(backend))
    column_colors = column_coloring(sparsity, coloring_algorithm(backend))
    column_color_groups = get_groups(column_colors)
    compressed_dx = similar(x)
    compressed_col = similar(y)
    pushforward_extras = prepare_pushforward(f!, y, backend, x, compressed_dx)
    return (;
        sparsity,
        column_colors,
        column_color_groups,
        compressed_dx,
        compressed_col,
        pushforward_extras,
    )
end

function sparse_prepare_jacobian_aux(f!, y, backend, x, ::PushforwardSlow)
    sparsity = jacobian_sparsity(f!, y, x, sparsity_detector(backend))
    row_colors = row_coloring(sparsity, coloring_algorithm(backend))
    row_color_groups = get_groups(row_colors)
    compressed_dy = similar(y)
    compressed_row = similar(x)
    pullback_extras = prepare_pullback(f!, y, backend, x, compressed_dy)
    return (;
        sparsity,
        row_colors,
        row_color_groups,
        compressed_dy,
        compressed_row,
        pullback_extras,
    )
end

function jacobian!(f!, y, jac, backend::AutoSparse, x, extras::NamedTuple)
    return sparse_jacobian_aux!(
        f!, y, jac, backend, x, extras, pushforward_performance(backend)
    )
end

function sparse_jacobian_aux!(f!, y, jac, backend, x, extras, ::PushforwardFast)
    (; sparsity, column_color_groups, compressed_dx, compressed_col, pushforward_extras) =
        extras
    for group in column_color_groups
        compressed_dx .= zero(eltype(compressed_dx))
        for j in group
            compressed_dx[j] = one(eltype(compressed_dx))
        end
        pushforward!(f!, y, compressed_col, backend, x, compressed_dx, pushforward_extras)
        @views for j in group
            nonzero_rows_j = (!iszero).(sparsity[:, j])
            copyto!(jac[nonzero_rows_j, j], compressed_col[nonzero_rows_j])
        end
    end
    return jac
end

function sparse_jacobian_aux!(f!, y, jac, backend, x, extras, ::PushforwardSlow)
    (; sparsity, row_color_groups, compressed_dy, compressed_row, pullback_extras) = extras
    for group in row_color_groups
        compressed_dy .= zero(eltype(compressed_dy))
        for i in group
            compressed_dy[i] = one(eltype(compressed_dy))
        end
        pullback!(f!, y, compressed_row, backend, x, compressed_dy, pullback_extras)
        @views for i in group
            nonzero_columns_i = (!iszero).(sparsity[i, :])
            copyto!(jac[i, nonzero_columns_i], compressed_row[nonzero_columns_i])
        end
    end
    return jac
end

function value_and_jacobian!(f!, y, jac, backend::AutoSparse, x, extras::NamedTuple)
    jacobian!(f!, y, jac, backend, x, extras)
    f!(y, x)
    return y, jac
end

function jacobian(f!, y, backend::AutoSparse, x, extras::NamedTuple)
    jac = similar(extras.sparsity, eltype(y))
    return jacobian!(f!, y, jac, backend, x, extras)
end

function value_and_jacobian(f!, y, backend::AutoSparse, x, extras::NamedTuple)
    jac = similar(extras.sparsity, eltype(y))
    return value_and_jacobian!(f!, y, jac, backend, x, extras)
end
