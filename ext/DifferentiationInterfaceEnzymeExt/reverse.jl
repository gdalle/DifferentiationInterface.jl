const AutoReverseEnzyme = AutoEnzyme{Val{:reverse}}
DI.autodiff_mode(::AutoReverseEnzyme) = Val{:reverse}()
DI.handles_output_type(::AutoReverseEnzyme, ::Type{<:AbstractArray}) = false

## Primitives

function DI.value_and_pullback!(
    _dx, ::AutoReverseEnzyme, f, x::X, dy::Y
) where {X<:Number,Y<:Union{Real,Nothing}}
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback!(
    dx::X, ::AutoReverseEnzyme, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:Union{Real,Nothing}}
    dx .= zero(eltype(dx))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx))
    dx .*= dy
    return y, dx
end

## Utilities

function DI.value_and_gradient(::AutoReverseEnzyme, f, x::AbstractArray)
    y = f(x)
    grad = gradient(Reverse, f, x)
    return y, grad
end

function DI.value_and_gradient!(
    grad::AbstractArray, ::AutoReverseEnzyme, f, x::AbstractArray
)
    y = f(x)
    gradient!(Reverse, grad, f, x)
    return y, grad
end
