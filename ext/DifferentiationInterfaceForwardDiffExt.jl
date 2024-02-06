module DifferentiationInterfaceForwardDiffExt

using DifferentiationInterface
using DocStringExtensions
using ForwardDiff
using ForwardDiff: Dual, Tag, value, extract_derivative, extract_derivative!
using LinearAlgebra

function extract_value(::Type{T}, ydual) where {T}
    return value.(T, ydual)
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    _dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:Real,Y<:Real}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = extract_value(T, ydual)
    new_dy = extract_derivative(T, ydual)
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:Real,Y<:AbstractArray}
    T = typeof(Tag(f, X))
    xdual = Dual{T}(x, dx)
    ydual = f(xdual)
    y = extract_value(T, ydual)
    dy = extract_derivative!(T, dy, ydual)
    return y, dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    _dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:Real}
    res = DiffResults.GradientResult(x)
    ForwardDiff.gradient!(res, f, x)
    y = DiffResults.value(res)
    new_dy = dot(DiffResults.gradient(res), dx)
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.pushforward!(
    dy::Y, ::ForwardDiffBackend, f, x::X, dx::X
) where {X<:AbstractArray,Y<:AbstractArray}
    res = DiffResults.JacobianResult(x)
    ForwardDiff.jacobian!(res, f, x) # TODO: replace with duals, n times too slow
    y = DiffResults.value(res)
    J = DiffResults.jacobian(res)
    mul!(dy, J, dx)
    return y, dy
end

end # module
