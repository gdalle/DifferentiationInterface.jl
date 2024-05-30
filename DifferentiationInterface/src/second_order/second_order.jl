"""
    SecondOrder

Combination of two backends for second-order differentiation.

!!! danger
    `SecondOrder` backends do not support first-order operators.

# Constructor

    SecondOrder(outer_backend, inner_backend)

# Fields

$(TYPEDFIELDS)
"""
struct SecondOrder{ADO<:AbstractADType,ADI<:AbstractADType} <: AbstractADType
    "backend for the outer differentiation"
    outer::ADO
    "backend for the inner differentiation"
    inner::ADI
end

function Base.show(io::IO, backend::SecondOrder)
    return print(io, "SecondOrder($(outer(backend)) / $(inner(backend)))")
end

"""
    inner(backend::SecondOrder)

Return the inner backend of a [`SecondOrder`](@ref) object, tasked with differentiation at the first order.
"""
inner(backend::SecondOrder) = backend.inner

"""
    outer(backend::SecondOrder)

Return the outer backend of a [`SecondOrder`](@ref) object, tasked with differentiation at the second order.
"""
outer(backend::SecondOrder) = backend.outer

"""
    mode(backend::SecondOrder)

Return the _outer_ mode of the second-order backend.
"""
ADTypes.mode(backend::SecondOrder) = mode(outer(backend))

"""
    nested(operator, backend)

Return a possibly modified `backend` that can call `operator` while nested inside another differentiation procedure.

At the moment, this is only useful for Enzyme, which needs `autodiff_deferred` to be compatible with higher-order differentiation.
"""
nested(operator::Function, backend::AbstractADType) = backend
