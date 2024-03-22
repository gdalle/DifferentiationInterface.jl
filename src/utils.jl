myeltype(x) = eltype(x)

mysimilar(x::Number) = zero(x)
mysimilar(x::AbstractArray{T}) where {T} = similar(x, T, axes(x)) # strip structure (issue #35)
mysimilar(x) = similar(x)

myupdate!(_old::Number, new::Number) = new
myupdate!(old, new) = old .= new
myupdate!(old, new::Nothing) = old

myzero(x) = zero(x)

myzero!(x::Number) = zero(x)
myzero!(x) = x .= zero(myeltype(x))

myvec(x::Number) = [x]
myvec(x) = vec(x)

"""
    basisarray(backend, a::AbstractArray, i::CartesianIndex)

Construct the `i`-th stardard basis array in the vector space of `a` with element type `eltype(a)`.

## Note

If an AD backend benefits from a more specialized basis array implementation,
this function can be extended on the backend type.
"""
basisarray(::AbstractADType, a::AbstractArray, i) = basisarray(a, i)

function basisarray(a::AbstractArray{T,N}, i::CartesianIndex{N}) where {T,N}
    return OneElement(one(T), Tuple(i), axes(a))
end
