"""
    CompressedMatrix{dir}

Compressed representation `B` of a sparse matrix `A ∈ ℝ^{m×n}` obtained by summing some of its columns (if `dir == :col`) or rows (if `dir == :row`), grouped by color.

# Fields

- `sparsity::AbstractMatrix{Bool}`: boolean sparsity pattern of the matrix `A`
- `colors::Vector{Int}`: vector such that
  - if `dir == `:col`, then `colors[j] ∈ 1:c` is the color of column `j`
  - if `dir == `:row`, then `colors[i] ∈ 1:c` is the color of row `i`
- `groups::Vector{Vector{Int}}`: vector of length `c` such that
  - if `dir == :col`, then `groups[k]` is the vector of column indices assigned to the same color `k ∈ 1:c`
  - if `dir == :row`, then `groups[k]` is the vector of row indices assigned to the same color `k ∈ 1:c`
- `aggregates::AbstractMatrix`: matrix `B` such that
  - if `dir == :col`, then `size(B) = (m, c)` and `B[:, c] = sum(A[:, k] for k in groups[c])`
  - if `dir == :row`, then `size(B) = (c, n)` and `B[c, :] = sum(A[k, :] for k in groups[c])`
"""
mutable struct CompressedMatrix{dir,S<:AbstractMatrix{Bool},M<:AbstractMatrix}
    sparsity::S
    colors::Vector{Int}
    groups::Vector{Vector{Int}}
    aggregates::M
end

"""
    CompressedMatrix{dir}(sparsity, groups, aggregates)

Constructor for [`CompressedMatrix`](@ref).
"""
function CompressedMatrix{dir}(sparsity, colors, groups, aggregates) where {dir}
    @assert dir in (:col, :row)
    return CompressedMatrix{dir,typeof(sparsity),typeof(aggregates)}(
        sparsity, colors, groups, aggregates
    )
end

function decompress!(A::AbstractMatrix, compressed::CompressedMatrix{:col})
    (; sparsity, colors, aggregates) = compressed
    A .= zero(eltype(A))
    @views for j in axes(A, 2)
        k = colors[j]
        nz_rows_j = (!iszero).(sparsity[:, j])
        copyto!(A[nz_rows_j, j], aggregates[nz_rows_j, k])
    end
    return A
end

function decompress!(A::AbstractMatrix, compressed::CompressedMatrix{:row})
    (; sparsity, colors, aggregates) = compressed
    A .= zero(eltype(A))
    @views for i in axes(A, 1)
        k = colors[i]
        nz_cols_i = (!iszero).(sparsity[i, :])
        copyto!(A[i, nz_cols_i], aggregates[k, nz_cols_i])
    end
    return A
end
