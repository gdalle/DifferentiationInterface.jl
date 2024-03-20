"""
    SecondOrder

Combination of two backends for second-order differentiation.

# Fields

$(TYPEDFIELDS)
"""
struct SecondOrder{AD1<:AbstractADType,AD2<:AbstractADType} <: AbstractADType
    "backend for the inner differentiation"
    inner::AD1
    "backend for the outer differentiation"
    outer::AD2
end

inner(backend::SecondOrder) = backend.inner
outer(backend::SecondOrder) = backend.outer
mode(backend::SecondOrder) = (mode(inner(backend)), mode(outer(backend)))

function Base.show(io::IO, backend::SecondOrder)
    return print(io, "SecondOrder($(inner(backend)), $(outer(backend)))")
end
