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
    Tangents(d::Vararg{T,B}) where {B,T} = new{B,T}(d)
end

Base.eltype(::Tangents{B,T}) where {B,T} = T

Base.only(t::Tangents) = only(t.d)
Base.first(t::Tangents) = first(t.d)

Base.:(==)(t1::Tangents{B}, t2::Tangents{B}) where {B} = t1.d == t2.d

function Base.isapprox(t1::Tangents{B}, t2::Tangents{B}; kwargs...) where {B}
    return all(isapprox.(t1.d, t2.d; kwargs...))
end

function Base.copyto!(t1::Tangents{B}, t2::Tangents{B}) where {B}
    copyto!(t1.d, t2.d)
    return t1
end
