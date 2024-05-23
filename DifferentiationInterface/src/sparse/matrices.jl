## Conversion between row- and col-major

"""
    col_major(A::AbstractMatrix)

Construct a column-major representation of the matrix `A`.
"""
col_major(A::M) where {M<:AbstractMatrix} = A
col_major(A::Transpose{<:Any,M}) where {M<:AbstractMatrix} = M(A)

"""
    row_major(A::AbstractMatrix)

Construct a row-major representation of the matrix `A`.
"""
row_major(A::M) where {M<:AbstractMatrix} = transpose(M(transpose(A)))
row_major(A::Transpose{<:Any,M}) where {M<:AbstractMatrix} = A

## Similar

major_respecting_similar(A::AbstractMatrix, ::Type{T}) where {T} = similar(A, T)

function major_respecting_similar(A::Transpose, ::Type{T}) where {T}
    return transpose(similar(parent(A), T))
end
