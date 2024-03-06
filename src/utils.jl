"""
    basisarray(a::AbstractArray, i)
    basisarray(backend, a::AbstractArray, i)

Construct the `i`-th stardard basis array in the vector space of `a` with element type `eltype(a)`.

## Note

If an AD backend benefits from a more specialized unit vector implementation,
this function can be extended on the backend type.

function basisarray(::AbstractBackend, a::AbstractVector{T}, i::Integer) where {T}
    return OneElement(one(T), i, length(v))
end
"""
basisarray(::AbstractBackend, a::AbstractArray, i) = basisarray(a, i)

function basisarray(a::AbstractArray{T,N}, i::CartesianIndex{N}) where {T,N}
    return OneElement(one(T), Tuple(i), axes(a))
end

mysimilar(x::Number) = zero(x)
mysimilar(x::AbstractArray) = similar(x)
