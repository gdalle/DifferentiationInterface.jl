module DifferentiationInterfaceEnzymeExt

using DifferentiationInterface
using DocStringExtensions
using Enzyme

## Forward-mode

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    _dy::Y, ::EnzymeForwardBackend, f, x::X, dx::X
) where {X,Y<:Real}
    return only(autodiff(Forward, f, DuplicatedNoNeed, Duplicated(x, dx)))
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::EnzymeForwardBackend, f, x::X, dx::X
) where {X,Y<:AbstractArray}
    dy .= only(autodiff(Forward, f, DuplicatedNoNeed, Duplicated(x, dx)))
    return dy
end

## Reverse-mode

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pullback!(
    _dx::X, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:Number,Y<:Union{Real,Nothing}}
    return only(first(autodiff(Reverse, f, Active, Active(x)))) * dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pullback!(
    dx::X, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Union{Real,Nothing}}
    dx .= zero(eltype(dx))
    autodiff(Reverse, f, Active, Duplicated(x, dx))
    dx .*= dy
    return dx
end

end # module
