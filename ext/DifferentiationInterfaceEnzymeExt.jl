module DifferentiationInterfaceEnzymeExt

using DifferentiationInterface
using DocStringExtensions
using Enzyme

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::EnzymeBackend, f, x::X, dx::X
) where {X,Y}
    return only(autodiff(Forward, f, DuplicatedNoNeed, Duplicated(x, dx)))
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pullback!(
    dx::X, ::EnzymeBackend, f, x::X, dy::Y
) where {X,Y<:Union{Real,Nothing}}
    dx .= zero(eltype(dx))
    autodiff(Reverse, f, Active, Duplicated(x, dx))
    dx .*= dy  # TODO: doesn't work with arbitrary dx
    return dx
end

end
