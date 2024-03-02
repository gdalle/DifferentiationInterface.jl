module DifferentiationInterfaceEnzymeExt

using DifferentiationInterface
using Enzyme: Forward, ReverseWithPrimal, Active, Duplicated, autodiff

const EnzymeBackends = Union{EnzymeForwardBackend,EnzymeReverseBackend}

## Unit vector

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DifferentiationInterface.unitvector(
    ::EnzymeBackends, v::AbstractVector{T}, i
) where {T}
    uv = zero(v)
    uv[i] = one(T)
    return uv
end

## Forward mode

function DifferentiationInterface.value_and_pushforward!(
    _dy::Y, ::EnzymeForwardBackend, f, x::X, dx
) where {X,Y<:Real}
    y, new_dy = autodiff(Forward, f, Duplicated, Duplicated(x, dx))
    return y, new_dy
end

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
    dx, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Union{Real,Nothing}}
    dx .= zero(eltype(dx))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx))
    dx .*= dy
    return y, dx
end

end # module
