## Pullback

function DI.prepare_pullback(
    f!, y, ::AutoEnzyme{<:Union{ReverseMode,Nothing}}, x, ty::Tangents
)
    return NoPullbackExtras()
end

function DI.value_and_pullback(
    f!,
    y,
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
)
    tx = map(ty) do dy
        only(DI.pullback(f!, y, extras, backend, x, Tangents(dy)))
    end
    f!(y, x)
    return y, tx
end

function DI.value_and_pullback(
    f!,
    y,
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::Tangents{1},
)
    f!_and_df! = get_f_and_df(f!, backend)
    dy_sametype = convert(typeof(y), copy(only(ty)))
    y_and_dy = Duplicated(y, dy_sametype)
    _, dx = only(
        autodiff(reverse_mode_noprimal(backend), f!_and_df!, Const, y_and_dy, Active(x))
    )
    return y, Tangents(dx)
end

function DI.value_and_pullback(
    f!,
    y,
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1},
)
    f!_and_df! = get_f_and_df(f!, backend)
    dx_sametype = make_zero(x)
    dy_sametype = convert(typeof(y), copy(only(ty)))
    x_and_dx = Duplicated(x, dx_sametype)
    y_and_dy = Duplicated(y, dy_sametype)
    autodiff(reverse_mode_noprimal(backend), f!_and_df!, Const, y_and_dy, x_and_dx)
    return y, Tangents(dx_sametype)
end
