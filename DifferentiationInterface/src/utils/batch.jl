"""
    pick_batchsize(backend::AbstractADType, dimension::Integer)

Pick a reasonable batch size for batched derivative evaluation with a given total `dimension`.
"""
function pick_batchsize(::AbstractADType, dimension::Integer)
    return min(dimension, 8)
end

"""
    Batch{B,T}

Efficient storage for `B` elements of type `T` (`NTuple` wrapper).

A `Batch` can be used as seed to trigger batched-mode `pushforward`, `pullback` and `hvp`.

# Fields

- `elements::NTuple{B,T}`
"""
struct Batch{B,T}
    elements::NTuple{B,T}
    Batch(elements::NTuple) = new{length(elements),eltype(elements)}(elements)
end

Base.:(==)(b1::Batch{B}, b2::Batch{B}) where {B} = b1.elements == b2.elements

function Base.isapprox(b1::Batch{B}, b2::Batch{B}; kwargs...) where {B}
    return all(isapprox.(b1.elements, b2.elements; kwargs...))
end
