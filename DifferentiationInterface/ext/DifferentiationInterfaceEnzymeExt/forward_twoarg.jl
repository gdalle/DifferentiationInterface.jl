## Pushforward

function DI.prepare_pushforward(f!, y, ::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}}, x, dx)
    return NoPushforwardExtras()
end

function DI.value_and_pushforward(
    f!,
    y,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    dx,
    ::NoPushforwardExtras,
)
    dx_sametype = convert(typeof(x), dx)
    dy_sametype = zero(y)
    autodiff(
        forward_mode(backend),
        Const(f!),
        Const,
        Duplicated(y, dy_sametype),
        Duplicated(x, dx_sametype),
    )
    return y, dy_sametype
end
