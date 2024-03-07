
## Backend construction

"""
$(SIGNATURES)
"""
DI.EnzymeReverseBackend(; custom::Bool=true) = EnzymeReverseBackend{custom}()

## Primitives

function DI.value_and_pullback!(
    _dx, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:Number,Y<:Union{Real,Nothing}}
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback!(
    dx::X, ::EnzymeReverseBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Union{Real,Nothing}}
    dx .= zero(eltype(dx))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx))
    dx .*= dy
    return y, dx
end

## Utilities

function DI.value_and_gradient(::EnzymeReverseBackend{true}, f, x::AbstractArray)
    y = f(x)
    grad = gradient(Reverse, f, x)
    return y, grad
end

function DI.value_and_gradient!(
    grad::AbstractArray, ::EnzymeReverseBackend{true}, f, x::AbstractArray
)
    y = f(x)
    gradient!(Reverse, grad, f, x)
    return y, grad
end
