"""
    CompressedMatrix{dir}

Compressed representation `B` of a `(m, n)` sparse matrix `A` obtained by summing some of its columns (if `dir == :col`) or rows (if `dir == :row`) if they have the same color.

# Fields

| field        | type                     | size                 | meaning                    | if `dir` is `:col`                | if `dir` is `:row`                |
| :----------- | :----------------------- | :------------------- | :------------------------- | :-------------------------------- | :-------------------------------- |
| `sparsity`   | `AbstractMatrix{Bool}`   | `(m, n)`             | sparsity pattern        | column-major                      | row-major                         |
| `colors`     | `Vector{Int}`            | `n` or `m`           | color assignments in `1:c` | `colors[j]` of col `j`            | `colors[i]` of row `i`            |
| `groups`     | `Vector{Vector{Int}}`    | `c`                  | groups with same color     | `groups[k] = {j : colors[j] = k}` | `groups[k] = {i : colors[i] = k}` |
| `aggregates` | `AbstractMatrix{<:Real}` | `(m, c)` or `(c, n)` | color-summed values `B`    | `B[:, c] = sum(A[:, groups[k]])`  | `B[c, :] = sum(A[groups[k], :])`  |
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

## Column decompression

function decompress!(A::AbstractMatrix, compressed::CompressedMatrix{:col})
    (; sparsity, colors, aggregates) = compressed
    A .= zero(eltype(A))
    @views for j in axes(A, 2)
        k = colors[j]
        rows_j = (!iszero).(sparsity[:, j])
        copyto!(A[rows_j, j], aggregates[rows_j, k])
    end
    return A
end

function decompress!(
    A::SparseMatrixCSC, compressed::CompressedMatrix{:col,<:SparseMatrixCSC}
)
    # A and compressed.sparsity have the same pattern
    (; colors, aggregates) = compressed
    Anz, Arv = nonzeros(A), rowvals(A)
    Anz .= zero(eltype(A))
    @views for j in axes(A, 2)
        k = colors[j]
        nzrange_j = nzrange(A, j)
        rows_j = Arv[nzrange_j]
        copyto!(Anz[nzrange_j], aggregates[rows_j, k])
    end
    return A
end

## Row decompression

function decompress!(A::AbstractMatrix, compressed::CompressedMatrix{:row})
    (; sparsity, colors, aggregates) = compressed
    A .= zero(eltype(A))
    @views for i in axes(A, 1)
        k = colors[i]
        cols_i = (!iszero).(sparsity[i, :])
        copyto!(A[i, cols_i], aggregates[k, cols_i])
    end
    return A
end

function decompress!(
    A::Transpose{<:Any,<:SparseMatrixCSC},
    compressed::CompressedMatrix{:row,<:Transpose{<:Any,<:SparseMatrixCSC}},
)
    # A and compressed.sparsity have the same pattern
    (; colors, aggregates) = compressed
    PA = parent(A)
    PAnz, PArv = nonzeros(PA), rowvals(PA)
    PAnz .= zero(eltype(A))
    @views for i in axes(A, 1)
        k = colors[i]
        nzrange_i = nzrange(PA, i)
        cols_i = PArv[nzrange_i]
        copyto!(PAnz[nzrange_i], aggregates[k, cols_i])
    end
    return A
end
