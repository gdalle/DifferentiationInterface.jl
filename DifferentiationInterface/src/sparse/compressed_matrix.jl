"""
    CompressedMatrix{dir}

Compressed representation `B` of a sparse matrix `A ∈ ℝ^{m×n}` obtained by summing some of its columns (if `dir == :col`) or rows (if `dir == :row`), grouped by color.

# Fields

- `sparsity::AbstractMatrix{Bool}`: boolean sparsity pattern of the matrix `A`
- `groups::Vector{Vector{Int}}`: vector of length `c` such that
  - if `dir == :col`, then `groups[k]` is the vector of column indices assigned to the same color `k ∈ 1:c`
  - if `dir == :row`, then `groups[k]` is the vector of row indices assigned to the same color `k ∈ 1:c`
- `aggregates::AbstractMatrix`: matrix `B` such that
  - if `dir == :col`, then `size(B) = (m, c)` and `B[:, c] = sum(A[:, k] for k in groups[c])`
  - if `dir == :row`, then `size(B) = (c, n)` and `B[c, :] = sum(A[k, :] for k in groups[c])`
"""
struct CompressedMatrix{dir,S<:AbstractMatrix{Bool},M<:AbstractMatrix}
    sparsity::S
    groups::Vector{Vector{Int}}
    aggregates::M
end

"""
    CompressedMatrix{dir}(sparsity, groups, aggregates)

Constructor for [`CompressedMatrix`](@ref).
"""
function CompressedMatrix{dir}(sparsity, groups, aggregates) where {dir}
    return CompressedMatrix{dir,typeof(sparsity),typeof(aggregates)}(
        sparsity, groups, aggregates
    )
end
