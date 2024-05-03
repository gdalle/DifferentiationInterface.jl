"""
    SecondOrder

Combination of two backends for second-order differentiation.

# Constructor

    SecondOrder(outer, inner)

# Fields

$(TYPEDFIELDS)
"""
struct SecondOrder{ADO<:AbstractADType,ADI<:AbstractADType} <: AbstractADType
    "backend for the outer differentiation"
    outer::ADO
    "backend for the inner differentiation"
    inner::ADI
end

SecondOrder(backend::AbstractADType) = SecondOrder(backend, backend)

inner(backend::SecondOrder) = backend.inner
outer(backend::SecondOrder) = backend.outer

function Base.show(io::IO, backend::SecondOrder)
    return print(io, "SecondOrder($(outer(backend)) / $(inner(backend)))")
end

"""
    mode(backend::SecondOrder)

Return the _outer_ mode of the second-order backend.
"""
ADTypes.mode(backend::SecondOrder) = mode(outer(backend))

function twoarg_support(backend::SecondOrder)
    if Bool(twoarg_support(inner(backend))) && Bool(twoarg_support(outer(backend)))
        return TwoArgSupported()
    else
        return TwoArgNotSupported()
    end
end
