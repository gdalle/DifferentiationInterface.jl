## Pullback

function DI.prepare_pullback(
    f!, y, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, ty::Tangents
)
    return NoPullbackExtras()
end

function DI.value_and_pullback(
    f!,
    y,
    extras::NoPullbackExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
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
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::Tangents{1},
)
    dy = only(ty)
    f!_and_df! = get_f_and_df(f!, backend)
    dy_sametype = convert(typeof(y), copy(dy))
    y_and_dy = Duplicated(y, dy_sametype)
    _, new_dx = if backend isa AutoDeferredEnzyme
        only(autodiff_deferred(reverse_mode(backend), f!_and_df!, Const, y_and_dy, Active(x)))
    else
        only(autodiff(reverse_mode(backend), f!_and_df!, Const, y_and_dy, Active(x)))
    end
    return y, Tangents(new_dx)
end

function DI.value_and_pullback(
    f!,
    y,
    ::NoPullbackExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::AbstractArray,
    ty::Tangents{1},
)
    dy = only(ty)
    f!_and_df! = get_f_and_df(f!, backend)
    dx_sametype = make_zero(x)
    dy_sametype = convert(typeof(y), copy(dy))
    y_and_dy = Duplicated(y, dy_sametype)
    x_and_dx = Duplicated(x, dx_sametype)
    if backend isa AutoDeferredEnzyme
        autodiff_deferred(reverse_mode(backend), f!_and_df!, Const, y_and_dy, x_and_dx)
    else
        autodiff(reverse_mode(backend), f!_and_df!, Const, y_and_dy, x_and_dx)
    end
    return y, Tangents(dx_sametype)
end
