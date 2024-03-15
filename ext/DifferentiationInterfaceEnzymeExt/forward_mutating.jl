## Primitives

function DI.value_and_pushforward!(
    y::AbstractArray,
    dy::AbstractArray,
    backend::AutoForwardEnzyme,
    f!,
    x,
    dx,
    extras::Nothing,
)
    dx_sametype = convert(typeof(x), dx)
    dy_sametype = convert(typeof(y), dy)
    autodiff(
        backend.mode, f!, Const, Duplicated(y, dy_sametype), Duplicated(x, dx_sametype)
    )
    dy .= dy_sametype
    return y, dy
end
