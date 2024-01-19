module DifferentiationInterfaceForwardDiffExt

using DifferentiationInterface
using DocStringExtensions
using ForwardDiff
using LinearAlgebra

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:Real,Y<:Real}
    new_dy = ForwardDiff.derivative(f, x) * dx
    return new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:Real,Y<:AbstractArray}
    ForwardDiff.derivative!(dy, f, x)
    dy .*= dx
    return dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:Real}
    g = ForwardDiff.gradient(f, x)  # TODO: replace with duals, n times too slow
    new_dy = dot(g, dx)
    return new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:AbstractArray}
    J = ForwardDiff.jacobian(f, x)  # TODO: replace with duals, n times too slow
    mul!(dy, J, dx)
    return dy
end

end
