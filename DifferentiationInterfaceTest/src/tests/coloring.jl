function check_structurally_orthogonal_columns(
    A::AbstractMatrix, colors::AbstractVector{<:Integer}
)
    for c in unique(colors)
        js = filter(j -> colors[j] == c, axes(A, 2))
        Ajs = @view A[:, js]
        nonzeros_per_row = count(!iszero, Ajs; dims=2)
        if maximum(nonzeros_per_row) > 1
            @warn "Color $c has columns $js sharing nonzeros"
            return false
        end
    end
    return true
end

function check_structurally_orthogonal_rows(
    A::AbstractMatrix, colors::AbstractVector{<:Integer}
)
    for c in unique(colors)
        is = filter(i -> colors[i] == c, axes(A, 1))
        Ais = @view A[is, :]
        nonzeros_per_column = count(!iszero, Ais; dims=1)
        if maximum(nonzeros_per_column) > 1
            @warn "Color $c has rows $is sharing nonzeros"
            return false
        end
    end
    return true
end

function check_symmetrically_structurally_orthogonal(
    A::AbstractMatrix, colors::AbstractVector{<:Integer}
)
    for i in axes(A, 2), j in axes(A, 2)
        if !iszero(A[i, j])
            group_i = filter(i2 -> (i2 != i) && (colors[i2] == colors[i]), axes(A, 2))
            group_j = filter(j2 -> (j2 != j) && (colors[j2] == colors[j]), axes(A, 2))
            A_group_i_column_j = @view A[group_i, j]
            A_group_j_column_i = @view A[group_j, i]
            nonzeros_group_i_column_j = count(!iszero, A_group_i_column_j)
            nonzeros_group_j_column_i = count(!iszero, A_group_j_column_i)
            if nonzeros_group_i_column_j > 0 && nonzeros_group_j_column_i > 0
                @warn """
                For coefficient $((i, j)), both of the following have confounding zeros:
                - color $(colors[j]) with group $group_j
                - color $(colors[i]) with group $group_i
                """
                return false
            end
        end
    end
    return true
end
