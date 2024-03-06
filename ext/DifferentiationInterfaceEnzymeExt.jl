module DifferentiationInterfaceEnzymeExt

using DifferentiationInterface: EnzymeReverseBackend, EnzymeForwardBackend
import DifferentiationInterface as DI
using DocStringExtensions
using Enzyme:
    Active,
    Duplicated,
    Forward,
    Reverse,
    ReverseWithPrimal,
    autodiff,
    gradient,
    gradient!,
    jacobian

const EnzymeBackends = Union{EnzymeForwardBackend,EnzymeReverseBackend}

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DI.basisarray(::EnzymeBackends, a::AbstractArray{T}, i) where {T}
    b = zero(a)
    b[i] = one(T)
    return b
end

## Primitives

### Forward mode

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(
    _dy::Y, ::EnzymeForwardBackend, f, x::X, dx
) where {X,Y<:Real}
    y, new_dy = autodiff(Forward, f, Duplicated, Duplicated(x, dx))
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(
    dy::Y, ::EnzymeForwardBackend, f, x::X, dx
) where {X,Y<:AbstractArray}
    y, new_dy = autodiff(Forward, f, Duplicated, Duplicated(x, dx))
    dy .= new_dy
    return y, dy
end

### Reverse mode

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pullback!(
    _dx, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:Number,Y<:Union{Real,Nothing}}
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pullback!(
    dx::X, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Union{Real,Nothing}}
    dx .= zero(eltype(dx))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx))
    dx .*= dy
    return y, dx
end

## Special cases

### Forward mode

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_jacobian(::EnzymeForwardBackend, f, x::AbstractArray)
    y = f(x)
    jac = jacobian(Forward, f, x)
    return y, jac
end

### Reverse mode

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_gradient(::EnzymeReverseBackend, f, x::AbstractArray)
    y = f(x)
    grad = gradient(Reverse, f, x)
    return y, grad
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_gradient!(
    grad::AbstractArray, ::EnzymeReverseBackend, f, x::AbstractArray
)
    y = f(x)
    gradient!(Reverse, grad, f, x)
    return y, grad
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_jacobian(::EnzymeReverseBackend, f, x::AbstractArray)
    y = f(x)
    jac = jacobian(Reverse, f, x)
    return y, jac
end

end # module
