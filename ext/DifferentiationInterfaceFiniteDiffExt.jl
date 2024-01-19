module DifferentiationInterfaceFiniteDiffExt

using DifferentiationInterface
using DocStringExtensions
using FiniteDiff
using LinearAlgebra

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:Number,Y<:Number}
    new_dy = FiniteDiff.finite_difference_derivative(f, x) * dx
    return new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:Number,Y<:AbstractArray}
    new_dy = FiniteDiff.finite_difference_derivative(f, x)
    dy .= new_dy .* dx
    return dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:Number}
    g = FiniteDiff.finite_difference_gradient(f, x)
    new_dy = dot(g, dx)
    return new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:AbstractArray}
    J = FiniteDiff.finite_difference_jacobian(f, x)
    mul!(dy, J, dx)
    return dy
end

end # module
