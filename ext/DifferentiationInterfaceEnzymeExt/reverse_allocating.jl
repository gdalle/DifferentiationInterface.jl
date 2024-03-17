# see https://enzymead.github.io/Enzyme.jl/stable/pullbacks/

struct MakeFunctionMutating{F}
    f::F
end

function (f!::MakeFunctionMutating)(y::AbstractArray, x)
    y .= f!.f(x)
    return nothing
end

## Primitives

function DI.value_and_pullback!(
    _dx::Number, ::AutoReverseEnzyme, f, x::Number, dy::Number, extras::Nothing
)
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback!(
    dx::AbstractArray, ::AutoReverseEnzyme, f, x::AbstractArray, dy::Number, extras::Nothing
)
    dx .= zero(eltype(dx))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx))
    dx .*= dy
    return y, dx
end

function DI.pullback!(
    _dx::Number, ::AutoReverseEnzyme, f, x::Number, dy::Number, extras::Nothing
)
    der = only(autodiff(Reverse, f, Active, Active(x)))
    new_dx = dy * only(der)
    return new_dx
end

function DI.pullback!(
    dx::AbstractArray, ::AutoReverseEnzyme, f, x::AbstractArray, dy::Number, extras::Nothing
)
    dx .= zero(eltype(dx))
    autodiff(Reverse, f, Active, Duplicated(x, dx))
    dx .*= dy
    return dx
end

function DI.value_and_pullback!(
    dx::Number, backend::AutoReverseEnzyme, f, x::Number, dy::AbstractArray, extras::Nothing
)
    y = f(x)
    f! = MakeFunctionMutating(f)
    return DI.value_and_pullback!(y, dx, backend, f!, x, dy, extras)
end

function DI.value_and_pullback!(
    dx::AbstractArray,
    backend::AutoReverseEnzyme,
    f,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing,
)
    y = f(x)
    f! = MakeFunctionMutating(f)
    return DI.value_and_pullback!(y, dx, backend, f!, x, dy, extras)
end

## Utilities

function DI.value_and_gradient!(
    grad::AbstractArray, ::AutoReverseEnzyme, f, x::AbstractArray, extras::Nothing
)
    y = f(x)
    grad_sametype = convert(typeof(x), grad)
    gradient!(Reverse, grad_sametype, f, x)
    grad .= grad_sametype
    return y, grad
end

function DI.value_and_gradient(::AutoReverseEnzyme, f, x::AbstractArray, extras::Nothing)
    y = f(x)
    grad = gradient(Reverse, f, x)
    return y, grad
end

function DI.gradient!(
    grad::AbstractArray, ::AutoReverseEnzyme, f, x::AbstractArray, extras::Nothing
)
    grad_sametype = convert(typeof(x), grad)
    gradient!(Reverse, grad_sametype, f, x)
    grad .= grad_sametype
    return grad
end

function DI.gradient(::AutoReverseEnzyme, f, x::AbstractArray, extras::Nothing)
    grad = gradient(Reverse, f, x)
    return grad
end
