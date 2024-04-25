"""
    Compressed{dir}

Compressed representation `B` of a matrix `A ∈ ℝ^{m×n}`, obtained by summing some of its columns (if `dir == :col`) or rows (if `dir == :row`).

# Fields

- `groups::Vector{Vector{Int}}`: vector of length `c` such that
  - if `dir == :col`, then `groups[k]` contains the set of columns with the same color `k ∈ 1:c`
  - if `dir == :row`, then `groups[k]` contains the set of rows with the same color `k ∈ 1:c`
- `compressed::AbstractMatrix`: matrix `B` such that
  - if `dir == :col`, then `size(B) = (m, c)` and the column `B[:, c] = sum(A[:, k] for k in groups[c])`
  - if `dir == :rows`, then `size(B) = (c, n)` and the row `B[c, :] = sum(A[k, :] for k in groups[c])`
"""
struct Compressed{dir,M<:AbstractMatrix}
    groups::Vector{Vector{Int}}
    compressed::M
end
