module DifferentiationInterfaceEnzymeExt

using DifferentiationInterface
using DocStringExtensions
using Enzyme

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    _dy::Y, ::EnzymeForwardBackend, f, x::X, dx::X
) where {X,Y}
    return only(autodiff(Forward, f, DuplicatedNoNeed, Duplicated(x, dx)))
end

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

end
