# see https://enzymead.github.io/Enzyme.jl/stable/pullbacks/

struct MakeFunctionMutating{F}
    f::F
end

function (f!::MakeFunctionMutating)(y::AbstractArray, x)
    y .= f!.f(x)
    return nothing
end

## Primitives

function DI.value_and_pullback!(_dx::Number, ::AutoReverseEnzyme, f, x::Number, dy::Number)
    der, y = autodiff(ReverseWithPrimal, f, Active, Active(x))
    new_dx = dy * only(der)
    return y, new_dx
end

function DI.value_and_pullback!(
    dx::AbstractArray, ::AutoReverseEnzyme, f, x::AbstractArray, dy::Number
)
    dx_sametype = convert(typeof(x), dx)
    dx_sametype .= zero(eltype(dx_sametype))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx_sametype))
    dx .= dx_sametype
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback!(
    dx::Number, backend::AutoReverseEnzyme, f, x::Number, dy::AbstractArray
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
