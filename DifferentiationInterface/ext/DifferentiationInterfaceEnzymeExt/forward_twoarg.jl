## Pushforward

function DI.prepare_pushforward(
    f!, y, ::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}}, x, tx::Tangents
)
    return NoPushforwardExtras()
end

function DI.value_and_pushforward(
    f!,
    y,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
    ::NoPushforwardExtras,
)
    f!_and_dfs! = get_f_and_df(f!, backend, tx)
    conv_x, zero_y = Converter(x), Zero(y)
    dxs_sametype = map(conv_x, tx.d)
    dys_sametype = map(zero_y, tx.d)
    x_and_dxs = BatchDuplicated(x, dxs_sametype)
    y_and_dys = BatchDuplicated(y, dys_sametype)
    if backend isa AutoDeferredEnzyme
        autodiff_deferred(forward_mode(backend), f!_and_dfs!, Const, y_and_dys, x_and_dxs)
    else
        autodiff(forward_mode(backend), f!_and_dfs!, Const, y_and_dys, x_and_dxs)
    end
    return y, Tangents(dys_sametype)
end

function DI.pushforward(
    f!,
    y,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
    extras::NoPushforwardExtras,
)
    _, ty = DI.value_and_pushforward(f!, y, backend, x, tx, extras)
    return ty
end

function DI.value_and_pushforward!(
    f!,
    y,
    ty::Tangents,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
    extras::NoPushforwardExtras,
)
    # TODO: make more efficient
    y, new_ty = DI.value_and_pushforward(f!, y, backend, x, tx, extras)
    return y, copyto!(ty, new_ty)
end

function DI.pushforward!(
    f!,
    y,
    ty::Tangents,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
    extras::NoPushforwardExtras,
)
    return copyto!(ty, DI.pushforward(f!, y, backend, x, tx, extras))
end
