"""
    pick_batchsize(backend::AbstractADType, dimension::Integer)

Pick a reasonable batch size for batched derivative evaluation with a given total `dimension`.

Returns `1` for backends which have not overloaded it.
"""
pick_batchsize(::AbstractADType, dimension::Integer) = 1

"""
    Tangents{B,T}

Storage for `B` (co)tangents of type `T` (`NTuple` wrapper).

`Tangents{B}` with `B > 1` can be used as seed to trigger Tangentsed-mode `pushforward`, `pullback` and `hvp`.

# Fields

- `d::NTuple{B,T}`
"""
struct Tangents{B,T}
    d::NTuple{B,T}
    Tangents(d::Vararg{T,B}...) = new{B,T}(d)
end

Base.eltype(::Tangents{B,T}) where {B,T} = T

Base.:(==)(b1::Tangents{B}, b2::Tangents{B}) where {B} = b1.d == b2.d

function Base.isapprox(b1::Tangents{B}, b2::Tangents{B}; kwargs...) where {B}
    return all(isapprox.(b1.d, b2.d; kwargs...))
end
