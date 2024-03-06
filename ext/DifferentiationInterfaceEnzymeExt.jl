module DifferentiationInterfaceEnzymeExt

using DifferentiationInterface
using DocStringExtensions
using Enzyme: Forward, ReverseWithPrimal, Active, Duplicated, autodiff

const EnzymeBackends = Union{EnzymeForwardBackend,EnzymeReverseBackend}

## Unit vector

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DifferentiationInterface.basisarray(
    ::EnzymeBackends, a::AbstractArray{T}, i
) where {T}
    b = zero(a)
    b[i] = one(T)
    return b
end

## Forward mode

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    _dy::Y, ::EnzymeForwardBackend, f, x::X, dx
) where {X,Y<:Real}
    y, new_dy = autodiff(Forward, f, Duplicated, Duplicated(x, dx))
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    dy::Y, ::EnzymeForwardBackend, f, x::X, dx
) where {X,Y<:AbstractArray}
    y, new_dy = autodiff(Forward, f, Duplicated, Duplicated(x, dx))
    dy .= new_dy
    return y, dy
end

## Reverse mode

function DifferentiationInterface.value_and_pullback!(
    _dx, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:Number,Y<:Union{Real,Nothing}}
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

function DifferentiationInterface.value_and_pullback!(
    dx::X, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Union{Real,Nothing}}
    dx .= zero(eltype(dx))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx))
    dx .*= dy
    return y, dx
end

# Enzyme's Duplicated assumes x and dx to be of the same type.
# When writing into pre-allocated arrays, e.g. Jacobians,
# dx often is a view or SubArray.
# This requires a specialized method that allocates a new dx.
"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pullback!(
    dx, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Union{Real,Nothing}}
    _dx = zero(x)
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, _dx))
    @. dx = _dx * dy
    return y, dx
end

end # module
