"""
    unitvector(backend, v::AbstractVector, i)
    unitvector(backend, v::Real, i)

Construct `i`-th stardard basis vector in the vector space of `v` with element type `eltype(v)`.
If `v` is a real number, one is returned.

## Note
If an AD backend benefits from a more specialized unit vector implementation,
this function can be extended on the backend type.
"""
function unitvector(::AbstractBackend, v::AbstractVector{T}, i) where {T}
    return OneElement(one(T), i, length(v))
end
unitvector(::AbstractBackend, v::T, i) where {T<:Real} = one(v)
