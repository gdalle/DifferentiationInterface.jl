"""
    basis(backend, a::AbstractArray, i::CartesianIndex)

Construct the `i`-th standard basis array in the vector space of `a` with element type `eltype(a)`.

## Note

If an AD backend benefits from a more specialized basis array implementation,
this function can be extended on the backend type.
"""
basis(::AbstractADType, a::AbstractArray, i) = basis(a, i)

"""
    multibasis(backend, a::AbstractArray, inds::AbstractVector{<:CartesianIndex})

Construct the sum of the `i`-th standard basis arrays in the vector space of `a` with element type `eltype(a)`, for all `i ∈ inds`.

## Note

If an AD backend benefits from a more specialized basis array implementation,
this function can be extended on the backend type.
"""
multibasis(::AbstractADType, a::AbstractArray, inds) = multibasis(a, inds)

function basis(a::AbstractArray{T,N}, i::CartesianIndex{N}) where {T,N}
    return zero(a) + OneElement(one(T), Tuple(i), axes(a))
end

function multibasis(
    a::AbstractArray{T,N}, inds::AbstractVector{<:CartesianIndex{N}}
) where {T,N}
    seed = zero(a)
    for i in inds
        seed += OneElement(one(T), Tuple(i), axes(a))
    end
    return seed
end
