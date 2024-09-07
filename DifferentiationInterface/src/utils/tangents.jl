"""
    pick_batchsize(backend::AbstractADType, dimension::Integer)

Pick a reasonable batch size for batched derivative evaluation with a given total `dimension`.

Returns `1` for backends which have not overloaded it.
"""
pick_batchsize(::AbstractADType, dimension::Integer) = 1

"""
    Tangents{B}

Storage for `B` (co)tangents (`NTuple` wrapper).

Must be used for the tangent argument to [`pushforward`](@ref), [`pullback`](@ref) and [`hvp`](@ref).

# Constructors

    Tangents(d)
    Tangents(d1, d2, ..., dB)
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

Base.only(t::Tangents) = only(t.d)
Base.first(t::Tangents) = first(t.d)
Base.getindex(t::Tangents, ind) = t.d[ind]
Base.firstindex(t::Tangents) = firstindex(t.d)
Base.lastindex(t::Tangents) = lastindex(t.d)

Base.iterate(t::Tangents) = iterate(t.d)
Base.iterate(t::Tangents, state) = iterate(t.d, state)

Base.map(f, t::Tangents) = Tangents(map(f, t.d)...)

Base.:(==)(t1::Tangents{B}, t2::Tangents{B}) where {B} = t1.d == t2.d

function Base.isapprox(t1::Tangents{B}, t2::Tangents{B}; kwargs...) where {B}
    return all(isapprox.(t1.d, t2.d; kwargs...))
end

function Base.copyto!(t1::Tangents{B}, t2::Tangents{B}) where {B}
    for b in eachindex(t1.d, t2.d)
        copyto!(t1.d[b], t2.d[b])
    end
    return t1
end
