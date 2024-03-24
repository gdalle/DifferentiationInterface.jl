myupdate!!(_old::Number, new::Number) = new
myupdate!!(old::AbstractArray, new) = old .= new
myupdate!!(old, new::Nothing) = old

mysimilar(x::Number) = zero(x)
mysimilar(x::AbstractArray) = similar(x)

mysimilar_random(x::Number) = randn(typeof(x))
mysimilar_random(x::AbstractArray) = randn(eltype(x), size(x))

myvec(x::Number) = [x]
myvec(x::AbstractArray) = vec(x)

myzero(x::Number) = zero(x)
myzero(x::AbstractArray) = zero(x)

myzero!!(x::Number) = zero(x)
myzero!!(x::AbstractArray) = x .= zero(eltype(x))

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
