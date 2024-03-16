## Primitives

function DI.value_and_pullback!(
    y::AbstractArray,
    _dx::Number,
    ::AutoReverseEnzyme,
    f!,
    x::Number,
    dy::AbstractArray,
    extras::Nothing=nothing,
)
    _, dx = only(autodiff(Reverse, f!, Const, Duplicated(y, copy(dy)), Active(x)))
    return y, dx
end

function DI.value_and_pullback!(
    y::AbstractArray,
    dx::AbstractArray,
    ::AutoReverseEnzyme,
    f!,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing=nothing,
)
    dx_sametype = convert(typeof(x), dx)
    dx_sametype .= zero(eltype(dx_sametype))
    dy_sametype = convert(typeof(y), copy(dy))  # TODO: dy is overwritten to zeros
    autodiff(Reverse, f!, Const, Duplicated(y, dy_sametype), Duplicated(x, dx_sametype))
    dx .= dx_sametype
    return y, dx
end
