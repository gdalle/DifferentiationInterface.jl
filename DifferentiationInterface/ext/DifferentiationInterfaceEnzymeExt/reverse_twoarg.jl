## Pullback

function DI.prepare_pullback(f!, y, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, dy)
    return NoPullbackExtras()
end

function DI.value_and_pullback(
    f!,
    y,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    dy,
    ::NoPullbackExtras,
)
    dy_sametype = convert(typeof(y), copy(dy))
    _, new_dx = only(
        autodiff(
            reverse_mode(backend), Const(f!), Const, Duplicated(y, dy_sametype), Active(x)
        ),
    )
    return y, new_dx
end

function DI.value_and_pullback(
    f!,
    y,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::AbstractArray,
    dy,
    ::NoPullbackExtras,
)
    dx_sametype = zero(x)
    dy_sametype = convert(typeof(y), copy(dy))
    autodiff(
        reverse_mode(backend),
        Const(f!),
        Const,
        Duplicated(y, dy_sametype),
        Duplicated(x, dx_sametype),
    )
    return y, dx_sametype
end
