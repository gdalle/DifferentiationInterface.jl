const AutoReverseEnzyme = AutoEnzyme{Val{:reverse}}
DI.autodiff_mode(::AutoReverseEnzyme) = DI.ReverseMode()

struct Mutating{F}
    f::F
end

function (m::Mutating{F})(y::AbstractArray, x) where {F}
    # see https://enzymead.github.io/Enzyme.jl/stable/pullbacks/
    y .= m.f(x)
    return nothing
end

## Primitives

function DI.value_and_pullback!(
    _dx, ::AutoReverseEnzyme, f, x::X, dy::Y, extras::Nothing=nothing
) where {X<:Number,Y<:Union{Real,Nothing}}
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback!(
    dx::X, ::AutoReverseEnzyme, f, x::X, dy::Y, extras::Nothing=nothing
) where {X<:AbstractArray,Y<:Union{Real,Nothing}}
    dx .= zero(eltype(dx))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx))
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback!(
    _dx, ::AutoReverseEnzyme, f, x::X, dy::Y, extras::Nothing=nothing
) where {X<:Number,Y<:AbstractArray}
    y = f(x)
    mf = Mutating(f)
    _, new_dx = only(autodiff(Reverse, mf, Const, Duplicated(y, copy(dy)), Active(x)))
    return y, new_dx
end

function DI.value_and_pullback!(
    dx, ::AutoReverseEnzyme, f, x::X, dy::Y, extras::Nothing=nothing
) where {X<:AbstractArray,Y<:AbstractArray}
    y = f(x)
    dx_righttype = similar(x)
    dx_righttype .= zero(eltype(dx_righttype))
    mf = Mutating(f)
    autodiff(Reverse, mf, Const, Duplicated(y, copy(dy)), Duplicated(x, dx_righttype))
    dx .= dx_righttype
    return y, dx
end

## Utilities

function DI.value_and_gradient(
    ::AutoReverseEnzyme, f, x::AbstractArray, extras::Nothing=nothing
)
    y = f(x)
    grad = gradient(Reverse, f, x)
    return y, grad
end

function DI.value_and_gradient!(
    grad::AbstractArray, ::AutoReverseEnzyme, f, x::AbstractArray, extras::Nothing=nothing
)
    y = f(x)
    gradient!(Reverse, grad, f, x)
    return y, grad
end
