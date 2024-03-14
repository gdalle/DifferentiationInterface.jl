const AutoReverseEnzyme = AutoEnzyme{Val{:reverse}}
DI.autodiff_mode(::AutoReverseEnzyme) = DI.ReverseMode()

# see https://enzymead.github.io/Enzyme.jl/stable/pullbacks/

struct MakeFunctionMutating{F}
    f::F
end

function (mf!::MakeFunctionMutating)(y::AbstractArray, x)
    y .= mf!.f(x)
    return nothing
end

## Primitives

function DI.value_and_pullback!(
    _dx::Number, ::AutoReverseEnzyme, f, x::Number, dy, extras::Nothing=nothing
)
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback!(
    dx::AbstractArray,
    ::AutoReverseEnzyme,
    f,
    x::AbstractArray,
    dy::Number,
    extras::Nothing=nothing,
)
    dx .= zero(eltype(dx))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx))
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback!(
    _dx::Number,
    ::AutoReverseEnzyme,
    f,
    x::Number,
    dy::AbstractArray,
    extras::Nothing=nothing,
)
    y = f(x)
    mf! = MakeFunctionMutating(f)
    _, new_dx = only(autodiff(Reverse, mf!, Const, Duplicated(y, copy(dy)), Active(x)))
    return y, new_dx
end

function DI.value_and_pullback!(
    dx::AbstractArray,
    ::AutoReverseEnzyme,
    f,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing=nothing,
)
    y = f(x)
    dx_like_x = zero(x)
    mf! = MakeFunctionMutating(f)
    autodiff(Reverse, mf!, Const, Duplicated(y, copy(dy)), Duplicated(x, dx_like_x))
    dx .= dx_like_x
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
