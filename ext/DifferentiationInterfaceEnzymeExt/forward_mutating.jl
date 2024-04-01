## Pushforward

function DI.value_and_pushforward!!(
    f!, y, dy, backend::AutoForwardEnzyme, x, dx, extras::Nothing
)
    dx_sametype = convert(typeof(x), dx)
    dy_sametype = zero_sametype!!(dy, y)
    autodiff(
        backend.mode, f!, Const, Duplicated(y, dy_sametype), Duplicated(x, dx_sametype)
    )
    return y, dy_sametype
end
