## Pushforward

DI.prepare_pushforward(f!, y, ::AutoForwardOrNothingEnzyme, x, dx) = NoPushforwardExtras()

function DI.value_and_pushforward(
    f!, y, backend::AutoForwardOrNothingEnzyme, x, dx, ::NoPushforwardExtras
)
    dx_sametype = convert(typeof(x), dx)
    dy_sametype = zero(y)
    autodiff(
        forward_mode(backend),
        f!,
        Const,
        Duplicated(y, dy_sametype),
        Duplicated(x, dx_sametype),
    )
    return y, dy_sametype
end
