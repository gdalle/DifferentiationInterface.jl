module DifferentiationInterfaceFiniteDiffExt

using DifferentiationInterface
import DifferentiationInterface: value_and_pushforward!

using FiniteDiff:
    finite_difference_derivative,
    finite_difference_gradient,
    finite_difference_gradient!,
    finite_difference_jacobian
using LinearAlgebra: dot, mul!

const DEFAULT_FDTYPE = Val{:central}

function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:Number,Y<:Number}
    y = f(x)
    der = finite_difference_derivative(
        f,
        x,
        DEFAULT_FDTYPE,  # fdtype
        eltype(dy),  # returntype
        y,  # fx
    )
    new_dy = der * dx
    return y, new_dy
end

function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:Number,Y<:AbstractArray}
    y = f(x)
    finite_difference_gradient!(
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

function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:Number}
    y = f(x)
    g = finite_difference_gradient(
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

function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::FiniteDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:AbstractArray}
    y = f(x)
    J = finite_difference_jacobian(
        f,
        x,
        DEFAULT_FDTYPE,  # fdtype
        eltype(dy),  # returntype
    )
    mul!(dy, J, dx)
    return y, dy
end

end # module
