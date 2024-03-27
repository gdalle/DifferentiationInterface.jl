mymul!!(x::Number, a) = x * a
mymul!!(x::AbstractArray, a) = x .*= a

myupdate!!(_old::Number, new::Number) = new
myupdate!!(old::AbstractArray, new) = old .= new

"""
    basis(backend, a::AbstractArray, i::CartesianIndex)

Construct the `i`-th stardard basis array in the vector space of `a` with element type `eltype(a)`.

## Note

If an AD backend benefits from a more specialized basis array implementation,
this function can be extended on the backend type.
"""
basis(::AbstractADType, a::AbstractArray, i) = basis(a, i)

function basis(a::AbstractArray{T,N}, i::CartesianIndex{N}) where {T,N}
    return OneElement(one(T), Tuple(i), axes(a))
end
