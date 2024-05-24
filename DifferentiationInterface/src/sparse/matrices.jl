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
