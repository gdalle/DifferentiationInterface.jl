"""
    pick_batchsize(backend::AbstractADType, dimension::Integer)

Pick a reasonable batch size for batched derivative evaluation with a given total `dimension`.

Returns `1` for backends which have not overloaded it.
"""
pick_batchsize(::AbstractADType, dimension::Integer) = 1

"""
    Tangents{B}

Storage for a batch of `B` tangents (wrapper around an `NTuple`).

The operators [`pushforward`](@ref), [`pullback`](@ref) and [`hvp`](@ref) require a `Tangents` argument in addition to the input `x`.

The underlying `NTuple` of `t::Tangents` can be retrieved with `NTuple(t)`.
We also define a few utility functions, as shown below.

# Constructors

    Tangents(d1)
    Tangents(d1, d2, ..., dB)

# Example

```jldoctest
julia> using DifferentiationInterface

julia> t = Tangents(2.0)
Tangents{1, Float64}((2.0,))

julia> NTuple(t)
(2.0,)

julia> length(t)
1

julia> only(t)
2.0

julia> t = Tangents([2.0], [4.0], [6.0])
Tangents{3, Vector{Float64}}(([2.0], [4.0], [6.0]))

julia> NTuple(t)
([2.0], [4.0], [6.0])

julia> length(t)
3

julia> t[2]
1-element Vector{Float64}:
 4.0
```
"""
struct Tangents{B,T}
    d::NTuple{B,T}

    function Tangents(d::Vararg{T,B}) where {T,B}
        return new{B,T}(d)
    end

    function Tangents()
        throw(ArgumentError("You must provide at least one tangent."))
    end
end

Base.length(::Tangents{B,T}) where {B,T} = B
Base.eltype(::Tangents{B,T}) where {B,T} = T

Base.NTuple(t::Tangents) = t.d

Base.only(t::Tangents) = only(NTuple(t))
Base.getindex(t::Tangents, ind) = NTuple(t)[ind]
Base.firstindex(t::Tangents) = firstindex(NTuple(t))
Base.lastindex(t::Tangents) = lastindex(NTuple(t))

Base.iterate(t::Tangents) = iterate(NTuple(t))
Base.iterate(t::Tangents, state) = iterate(NTuple(t), state)

Base.map(f, t::Tangents) = Tangents(map(f, NTuple(t))...)

Base.:(==)(t1::Tangents{B}, t2::Tangents{B}) where {B} = NTuple(t1) == NTuple(t2)

function Base.isapprox(t1::Tangents{B}, t2::Tangents{B}; kwargs...) where {B}
    return all(isapprox.(NTuple(t1), NTuple(t2); kwargs...))
end

function Base.copyto!(t1::Tangents{B}, t2::Tangents{B}) where {B}
    foreach(copyto!, NTuple(t1), NTuple(t2))
    return t1
end
