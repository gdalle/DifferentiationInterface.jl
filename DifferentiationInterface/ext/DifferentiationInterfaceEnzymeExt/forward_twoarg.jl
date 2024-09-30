## Pushforward

function DI.prepare_pushforward(
    f!::F,
    y,
    ::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return NoPushforwardPrep()
end

function DI.value_and_pushforward(
    f!::F,
    y,
    ::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f!_and_df! = get_f_and_df(f!, backend)
    dx_sametype = convert(typeof(x), only(tx))
    dy_sametype = make_zero(y)
    x_and_dx = Duplicated(x, dx_sametype)
    y_and_dy = Duplicated(y, dy_sametype)
    autodiff(
        forward_noprimal(backend),
        f!_and_df!,
        Const,
        y_and_dy,
        x_and_dx,
        map(translate, contexts)...,
    )
    return y, (dy_sametype,)
end

function DI.value_and_pushforward(
    f!::F,
    y,
    ::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f!_and_df! = get_f_and_df(f!, backend, Val(B))
    tx_sametype = map(Fix1(convert, typeof(x)), tx)
    ty_sametype = ntuple(_ -> make_zero(y), Val(B))
    x_and_tx = BatchDuplicated(x, tx_sametype)
    y_and_ty = BatchDuplicated(y, ty_sametype)
    autodiff(
        forward_noprimal(backend),
        f!_and_df!,
        Const,
        y_and_ty,
        x_and_tx,
        map(translate, contexts)...,
    )
    return y, ty_sametype
end

function DI.pushforward(
    f!::F,
    y,
    prep::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    _, ty = DI.value_and_pushforward(f!, y, prep, backend, x, tx, contexts...)
    return ty
end

function DI.value_and_pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    y, new_ty = DI.value_and_pushforward(f!, y, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return y, ty
end

function DI.pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    new_ty = DI.pushforward(f!, y, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return ty
end
