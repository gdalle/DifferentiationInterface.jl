module DifferentiationInterfaceFiniteDiffExt

using DifferentiationInterface
using DocStringExtensions
using FiniteDiff
using LinearAlgebra

const DEFAULT_FDTYPE = Val{:central}

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:Number,Y<:Number}
    y = f(x)
    der = FiniteDiff.finite_difference_derivative(
        f,
        x,
        Val{:central},  # fdtype
        eltype(dy),  # returntype
        y,  # fx
    )
    new_dy = der * dx
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:Number,Y<:AbstractArray}
    y = f(x)
    FiniteDiff.finite_difference_gradient!(
        dy,
        f,
        x,
        DEFAULT_FDTYPE,  # fdtype
        eltype(dy),  # returntype
        Val{false},  # inplace
        y,  # fx
    )
    dy .*= dx
    return y, dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:Number}
    y = f(x)
    g = FiniteDiff.finite_difference_gradient(
        f,
        x,
        DEFAULT_FDTYPE,  # fdtype
        eltype(dy),  # returntype
        Val{false},  # inplace
        y,  # fx
    )
    new_dy = dot(g, dx)
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:AbstractArray}
    y = f(x)
    J = FiniteDiff.finite_difference_jacobian(
        f,
        x,
        DEFAULT_FDTYPE,  # fdtype
        eltype(dy),  # returntype
    )
    mul!(dy, J, dx)
    return y, dy
end

end # module
