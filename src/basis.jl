"""
    basisarray(backend, v::AbstractArray, i)

Construct the `i`-th stardard basis array in the vector space of `v` with element type `eltype(v)`.

## Note

If an AD backend benefits from a more specialized unit vector implementation,
this function can be extended on the backend type.
"""
function basisarray(::AbstractBackend, v::AbstractVector{T}, i::Integer) where {T}
    return OneElement(one(T), i, length(v))
end
